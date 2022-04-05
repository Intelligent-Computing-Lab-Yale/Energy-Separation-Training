#!/bin/bash

epch=500
dataset="cifar100"
sles_model="mobilenet_v2"
out1=16
task="train_with_${dataset}_sles_${sles_model}_out1_${out1}"
train_set="../${dataset}/adv_train_set_${task}"
test_set="../${dataset}/adv_test_set_${task}"
saved_model="./models"
log_pth="./logs"

python phase1_gen_sois.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=16,32,64 --stage=0
python phase1_gen_sois.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=16,32,64 --stage=1
python phase1_gen_sois.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=16,32,64 --stage=2
