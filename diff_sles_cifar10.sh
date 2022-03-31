#!/bin/bash


#epch=500
#dataset="cifar10"
#model="resnet18"
#task="train_with_${dataset}_sles_${model}"
#model_pth="./cifar10/clean_trained_${model}_ckpt.pth"
#train_set="./cifar10/adv_train_set_cifar10_task_${task}"
#test_set="./cifar10/adv_test_set_cifar10_task_${task}"
##python adv_train_set_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$model --task=$task
#python adv_dataset_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$model --task=$task
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --stage=0
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --stage=1
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --stage=2
#
#epch=500
#model="vgg16"
#task="train_with_${dataset}_sles_${model}"
#model_pth="./cifar10/clean_trained_${model}_ckpt.pth"
#train_set="./cifar10/adv_train_set_cifar10_task_${task}"
#test_set="./cifar10/adv_test_set_cifar10_task_${task}"
##python adv_train_set_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$model --task=$task
#python adv_dataset_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$model --task=$task
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --stage=0
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --stage=1
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --stage=2

epch=500
dataset="cifar100"
model="mobilenet_v2"
out1=16
task="train_with_${dataset}_sles_${model}_out1_${out1}"
model_pth="../cifar100/clean_trained_${model}_ckpt.pth"
train_set="../cifar100/adv_train_set_cifar100_task_${task}"
test_set="../cifar100/adv_test_set_cifar100_task_${task}"
saved_model="./models"
log_pth="./logs"
#python adv_train_set_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$model --task=$task --per=100
#python adv_dataset_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$model --task=$task
python phase1_gen_sois.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=16,32,64 --stage=0
python phase1_gen_sois.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=16,32,64 --stage=1
python phase1_gen_sois.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=$saved_model --log_pth=$log_pth --task=$task --out=16,32,64 --stage=2

#out1=32
#task="train_with_${dataset}_sles_${model}_out1_${out1}"
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=$log_pth --task=$task --out=32,64,64 --stage=0
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=./ --task=$task --out=32,64,64 --stage=1
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=./ --task=$task --out=32,64,64 --stage=2

#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --out=16, --stage=0
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --out=16,32,64 --stage=1
#python phase1_train_mod.py --type=$dataset --batch_size=200 --epochs=$epch --pgd_params=8_4_10 --train_pth=$train_set --test_pth=$test_set --save_pth=./ECCV_models --log_pth=ECCV_log --task=$task --out=16,32,64 --stage=2
