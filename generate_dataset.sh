dataset="cifar100"
mkdir $dataset

S_les_model="mobilenet_v2"
task="${dataset}_sles_${S_les_model}"
model_pth="./cifar100/clean_trained_${S_les_model}_ckpt.pth"
train_set="./cifar100/adv_train_set_${task}"
test_set="./cifar100/adv_test_set_${task}"

python adv_train_set_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$S_les_model --task=$task --per=100
python adv_dataset_gen.py --type=$dataset --batch_size=200 --atype='pgd' --eps=0.04 --alpha=0.016 --steps=10 --path="./${dataset}" --model_pth=$model_pth --model=$S_les_model --task=$task
