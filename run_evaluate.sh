#!/bin/bash

dataset="cifar100"
S_les_model="mobilenet_v2"
Sv_model="mobilenet_v2"
detector_model="./models/task_train_with_${dataset}_sles_${S_les_model}_out1_8_stage_2" #"./cifar100_resnet/no_bn_momentum0.6_1000_layer3_resnet_od_cifar100_8_4_10.pth"
model_adv_generator="./clean_trained_${Sv_model}_ckpt.pth"
save_path="./energy_save_pth"
calc_error=1
percentile=95 #0.3077190101146695
n_batches=5
mkdir $save_path

python evaluate_different_attacks.py --type=$dataset --a_type=pgd --pgd_param=0.04,0.016,10 --energy_path=$save_path --model_adv_gen=$model_adv_generator --model_inference=$detector_model --clean=1 --baseline=0 --calc_err=$calc_error --save_energy_dist=0 --aname=pgd4 --percentile=$percentile --sles=$S_les_model --model=$Sv_model --n_batches=$n_batches --class_model_pth=$model_adv_generator
#