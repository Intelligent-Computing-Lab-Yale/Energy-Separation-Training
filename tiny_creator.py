import torch
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--task', default='any_task', help='dataset for training')
args = parser.parse_args()

l_test = ['test_set_task_'+args.task+'_4', 'test_set_task_'+args.task+'_9', 'test_set_task_'+args.task+'_14', 'test_set_task_'+args.task+'_19', 'test_set_task_'+args.task+'_24', 'test_set_task_'+args.task+'_29', 'test_set_task_'+args.task+'_34', 'test_set_task_'+args.task+'_39', 'test_set_task_'+args.task+'_44']

data_test = []
for i in l_test:

    a = torch.load('./tinyimagenet/' + i, map_location='cpu')
    # print(len(j))
    data_test.append(a)

torch.save(data_test, './tinyimagenet/task_'+args.task+'_test_set')
print(len(data_test))