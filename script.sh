#!/bin/bash

clear


# Select the filter, Options: Nonlinear_Detail, Log, Gamma 
METHODS=(Nonlinear_Detail)


# Select the Classifier under attack, Options: resnet50, resnet18, alexnet
ADVMODElS=(resnet50)


DATASET=(ImageNet) 


for method in "${METHODS[@]}"
do
	for advmodel in "${ADVMODElS[@]}"
	do	
		if [ $method == 'Nonlinear_Detail' ] || [ $method == 'Log' ]
		then
			Strengths=(1)
		fi	
		if [ $method == 'Linear_Detail' ]
		then
			Strengths=(2.0)
		fi
		if [ $method == 'Gamma' ]
		then
			Strengths=(0.5)
		fi
		for strength in "${Strengths[@]}"
		do
		        echo FilterFool generates $method \($strength\) adversarial image to attack $advmodel 
		        python -W ignore main.py --method=$method --adv_model=$advmodel --dataset=$DATASET --strength=$strength 
		done
	done
done
