import copy
import argparse
import sys
from os.path import join,isfile
from tqdm import tqdm
from os import listdir
import os
from train import run
from module import DeepGuidedFilter as FCNN
from misc_function import forward,Config

parser = argparse.ArgumentParser(description='Train FCNN to Generate Structure-Aware Adversarial Images ')
parser.add_argument('--method', type=str, required=True)
parser.add_argument('--adv_model', type=str,  help='adversarial model')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--strength', required=True, type=float,help='strength of filters')
args = parser.parse_args()


# Path to the directory that contains images
dataset_path='./CleanImgs/{}/'.format(args.method)
	
# Path to the directory that contains filtered images
IPF_path = './FilteredImages/{}/'.format(args.method)


# List of the name of all the filtered images in the selected IPF_path
image_list = [f for f in listdir(IPF_path) if not f.startswith('.')] 
NumImg=len(image_list)


default_config = Config(
	GPU = 0,
	LR = 0.001,
	# clip
	clip = None,
	# model
	model = None,
	# forward
	forward = None,
)

# Configuration
config = copy.deepcopy(default_config)
# model
config.model = FCNN()
config.forward = forward
config.clip = 0.01


log_path = 'Results/{}/'.format(args.method)
if not os.path.isdir(log_path):
	os.makedirs(log_path)
fileText_name = log_path+'log_{}_{}_{}.txt'.format(args.dataset,args.adv_model,str(args.strength))	

for idx in range(NumImg):
	fileText = open(fileText_name, 'a+')
	adv_img=run(config, dataset_path, IPF_path, image_list, idx, fileText, args)
	fileText.close()


