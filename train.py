import os
import time
import copy
import torch
import pytorch_msssim
import random
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import autograd
import numpy as np
import cv2
from tqdm import tqdm
import torchvision.transforms as T

from torchvision import models
from torch.nn import functional as F

from misc_function import processImage, recreate_image, PreidictLabel, AdvLoss




def run(config, dataset_path, IPF_path, image_list, idx,fileText, args):



	# Create a directory for saving adversarial images
	adv_path_no ='Results/{}/{}_{}_{}/'.format(args.method, str(args.strength), args.dataset, args.adv_model)
	if not os.path.isdir(adv_path_no):
		os.makedirs(adv_path_no)

		
	# Structure loss function
	l2_loss = nn.MSELoss()
	ssim_loss = pytorch_msssim.SSIM()
	
	
	# Using GPU
	if config.GPU >= 0:
		with torch.cuda.device(config.GPU):
			config.model.cuda()
			l2_loss.cuda()
			ssim_loss.cuda()
	
	
	# Setup optimizer
	optimizer = optim.Adam(config.model.parameters(), lr=config.LR)

	
	# Load the classifier for attacking 
	if args.adv_model ==  'resnet18':
		classifier = models.resnet18(pretrained=True)
	elif args.adv_model ==  'resnet50':
		classifier = models.resnet50(pretrained=True)
	elif args.adv_model ==  'alexnet':
		classifier = models.alexnet(pretrained=True)
	
	classifier.cuda()
	classifier.eval()


	# Freeze the parameters of the classifier under attack to not be updated
	for param in classifier.parameters():
		param.requires_grad = False



		
	# The name of the chosen image
	img_name = image_list[idx].split('/')[-1]
		

	
	# Load and Pre-processing the clean and filtered image 
	x= processImage(dataset_path,img_name)			
	gt_enh = processImage(IPF_path,img_name)


	# Compute the residual perturbation
	gt_noise = gt_enh - x

	# Perform inference on the clean image
	class_x, prob_class_x, prob_x, logit_x, semantic_vec, super_x = PreidictLabel(x, classifier)

	maxIters = 3000
	



	for it in tqdm(range(maxIters)): 
				
		with autograd.detect_anomaly():


			noise= config.forward(x,gt_noise, config)
			
			# Enhance adversarial image			
			enh = (x+noise).clamp(min=0, max=1)
			

			# Perform inference on the generated adversarial image
			class_enh, prob_class_enh, prob_enh, logit_enh , _, super_adv= PreidictLabel(enh, classifier)

			
			# Computing structure and semantic adversarial losses
			loss0 = l2_loss(noise, gt_noise)
			loss1 = 1-ssim_loss((noise+1)/2., (gt_noise+1)/2.)
			loss2 = AdvLoss(logit_enh, class_x,semantic_vec, is_targeted=False, num_classes=1000)
			

			# Normalized MSE
			loss3 = loss0.cpu().data.numpy().item(0)/l2_loss(gt_noise, torch.zeros(1,3,224,224).cuda())
			

			loss = loss0 + 0.01*loss1 + loss2
			
			# backward
			optimizer.zero_grad()
			loss.backward()
			if config.clip is not None:
				torch.nn.utils.clip_grad_norm(config.model.parameters(), config.clip)
			optimizer.step()

			# Save the generated adversarial image	
			cv2.imwrite('{}{}'.format(adv_path_no,img_name), recreate_image(enh))
			
			adv_img= processImage(adv_path_no,img_name)
			class_adv, _, _, _, _, super_adv = PreidictLabel(adv_img, classifier)	

			if args.method == 'Nonlinear_Detail':
				if (super_x != super_adv and class_x != class_adv and loss3<0.04 and it>2500):
					break
			elif args.method == 'Log':
				if (super_x != super_adv and class_x != class_adv and loss3<0.003 and it>2500):
					break	
			elif args.method == 'Linear_Detail':
				if (super_x != super_adv and class_x != class_adv and loss3<0.04 and it>2500):
					break
			elif args.method == 'Gamma':
				if (super_x != super_adv and class_x != class_adv and loss3<0.0005 and it>2500):
					break
	 		
			#print(img_name, it+1, super_x, super_adv, class_x.cpu().data.numpy().item(0), class_enh.cpu().data.numpy().item(0), class_adv.cpu().data.numpy().item(0),  loss0.cpu().data.numpy().item(0), loss3.cpu().data.numpy().item(0), loss1.cpu().data.numpy().item(0), loss2.cpu().data.numpy().item(0))


	text = '{}\tItrs:{}\tSemantic labels, Clean:{}\t Adversarial:{}\t Categorical labels, Clean:{}\t Adversarial:{}\t L_2 loss:{:.5f}\t SSIM loss{:.5f}\t Adv loss:{:.5f}\n'.format(img_name, it+1, super_x, super_adv, class_x,  class_adv,  loss0.cpu().data.numpy().item(0),  loss1.cpu().data.numpy().item(0), loss2.cpu().data.numpy().item(0))

	fileText.write(text)
	return adv_img
