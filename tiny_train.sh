#!/bin/bash

epch=500
task="sles_mobilenet_v2"
test_set="./tinyimagenet/task_${task}_test_set"
dataset="tinyimagenet"
lambda_a=0.9
log_dir="./logs"
save_pth="./saved_models"
model_pth="./clean_trained_tinyimagenet_netmobilenet_v2_ckpt.pth"
S_les_model="mobilenet_v2"

mkdir tinyimagenet
mkdir $log_dir
mkdir $save_pth

python adv_train_set_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$S_les_model --task=$task --per=100
python adv_dataset_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$S_les_model --task=$task

python tiny_creator.py --task $task
python tiny_train_param.py --type=$dataset --batch_size=200 --epochs=$epch --la=$lambda_a --lc=0.1 --train_pth=$train_set --test_pth=$test_set --save_pth=$save_pth --log_pth=$log_dir --task=$task --stage=0 --p=100
python tiny_train_param.py --type=$dataset --batch_size=200 --epochs=$epch --la=$lambda_a --lc=0.1 --train_pth=$train_set --test_pth=$test_set --save_pth=$save_pth --log_pth=$log_dir --task=$task --stage=1 --p=100
python tiny_train_param.py --type=$dataset --batch_size=200 --epochs=$epch --la=$lambda_a --lc=0.1 --train_pth=$train_set --test_pth=$test_set --save_pth=$save_pth --log_pth=$log_dir --task=$task --stage=2 --p=100