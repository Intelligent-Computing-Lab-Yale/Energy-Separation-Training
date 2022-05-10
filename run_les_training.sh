#!/bin/bash

epch=500
dataset="cifar100"
sles_model="mobilenet_v2"
out1=8
task="train_with_${dataset}_sles_${sles_model}_out1_${out1}"
dataset_path="${dataset}_task_${dataset}_sles_${sles_model}"
train_set="./${dataset}/adv_train_set_${dataset_path}"
test_set="./${dataset}/adv_test_set_${dataset_path}"
saved_model="./models"
log_pth="./logs"

mkdir $log_pth
mkdir $saved_model

python les_training.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=8,16,32 --stage=0
python les_training.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=8,16,32 --stage=1
python les_training.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=8,16,32 --stage=2
