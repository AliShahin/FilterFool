import cv2
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import copy


def processImage(dataset_path,img_name):

		x = cv2.imread(dataset_path+img_name, 1)/255.0
		# Have RGB images
		x = x[:, :, (2, 1, 0)]
		x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_LINEAR)
		x = x.transpose(2, 0, 1)  # Convert array to C,W,H
		x = torch.from_numpy(x).float()
		# Add one more channel to the beginning. Tensor shape = 1,3,224,224
		x.unsqueeze_(0)
		# Convert to Pytorch variable
		#x = Variable(x.cuda())
		return x.cuda()




def recreate_image(im_as_var):
	
	if im_as_var.shape[0] == 1:
		recreated_im = copy.copy(im_as_var.cpu().data.numpy()[0]).transpose(1,2,0)
	else:	
		recreated_im = copy.copy(im_as_var.cpu().data.numpy())
	recreated_im[recreated_im > 1] = 1
	recreated_im[recreated_im < 0] = 0
	recreated_im = np.uint(recreated_im * 255.)
	# Convert RBG to GBR
	recreated_im = recreated_im[..., ::-1]
	return recreated_im	

def FindSemanticClass(class_x):

		# Find superclass
		W = np.load('Categorical_2_Semantic_Mapping.npy', allow_pickle=True).item()
		Dogs=np.array(W['Dogs'])
		Mammals=np.array(W['Other mammals'])
		Birds=np.array(W['Birds'])
		Fish=np.array(W['Reptiles, fish, amphibians'])
		Inverterbrates=np.array(W['inverterbrates'])
		Food=np.array(W['Food, plants, fungi'])
		Devices=np.array(W['Devices'])
		Structures=np.array(W['Structures, furnishing'])
		Clothes=np.array(W['Clothes, covering'])
		Containers=np.array(W['Implements, containers, misc. objects'])
		Vehicles=np.array(W['vehicles'])
		Semanticclasses_Labels=[Dogs,Mammals,Birds,Fish,Inverterbrates,Food,Devices,Structures,Clothes,Containers,Vehicles]

		for i in range(11):
			if np.sum(Semanticclasses_Labels[i]==class_x.cpu().data.numpy().item(0)):
				superclass=i
		mappingSuper=np.load('MappingMatrix.npz')
		mappingTorch =torch.tensor(mappingSuper['MappingMatrix'], device="cuda:0").float()
		semantic_classes=mappingTorch[:,superclass]	
		return superclass, semantic_classes

def PreidictLabel(x, classifier):
 
 		#x=torch.round(x * 255)/255.
		mean = torch.zeros(x.shape).float().cuda()
		mean[:,0,:,:]=0.485
		mean[:,1,:,:]=0.456
		mean[:,2,:,:]=0.406

		std = torch.zeros(x.shape).float().cuda()
		std[:,0,:,:]=0.229
		std[:,1,:,:]=0.224
		std[:,2,:,:]=0.225


		# Standarise
		x = (x - mean) / std
  
		logit_x = classifier.forward(x)
		h_x = F.softmax(logit_x).data.squeeze()
		probs_x, idx_x = h_x.sort(0, True)
		class_x = idx_x[0]
		class_x_prob = probs_x[0]


		superclass, semantic_classes = FindSemanticClass(class_x) 
	
		return class_x, class_x_prob, probs_x, logit_x, semantic_classes, superclass


def AdvLoss(logits, target, semantic_idxs, is_targeted, num_classes=1000, kappa=0):
	# inputs to the softmax function are called logits.
	# https://arxiv.org/pdf/1608.04644.pdf
	target_one_hot = semantic_idxs
        
	# subtract large value from target class to find other max value
	# https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
	real = torch.sum(F.relu(target_one_hot*logits), 1)
	other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
	kappa = torch.zeros_like(other).fill_(kappa)

	if is_targeted:
		return torch.sum(torch.max(other-real, kappa))
	return torch.sum(torch.max(real-other, kappa))

def forward(imgs,gt, config):
	x_hr= imgs
	gt_hr=gt
	return config.model(x_hr, x_hr)

class Config(object):
    def __init__(self, **params):
        for k, v in params.items():
            self.__dict__[k] = v	