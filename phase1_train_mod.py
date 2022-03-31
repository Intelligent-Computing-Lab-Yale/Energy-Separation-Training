import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset
from datetime import datetime
import random
import numpy as np
import copy
import torchvision.models as models
import detector_net
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os.path

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=257, help='number of epochs to train (default: 10)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--la', type=float, default = 0.6)
parser.add_argument('--lc', type=float, default = 0.1)
parser.add_argument('--lr', type=float, default = 0.01)
parser.add_argument('--pgd_params', default = '8_4_10')
parser.add_argument('--baseline_model_pth', default = '.')
parser.add_argument('--train_pth', default = '.')
parser.add_argument('--test_pth', default = '.')
parser.add_argument('--save_pth', default = '.')
parser.add_argument('--log_pth', default = '.')
parser.add_argument('--task', default = 'PGD50_training')
parser.add_argument('--stage', type=int, default=0,  help='how many stages')
parser.add_argument('--p', type=float, default = 100)
parser.add_argument('--out', default = '8,16,32')

args = parser.parse_args()
out1, out2, out3 = map(int, args.out.split(','))

print(f' log file path {args.log_pth}/{args.task}_log')
if (args.stage==0 and os.path.exists(args.log_pth+'/'+args.task+'_log') == 1) or (os.path.exists(args.log_pth+'/'+args.task+'_log') == 0):
    f = open(args.log_pth+'/'+args.task+'_log', 'w')
else:
    f = open(args.log_pth + '/' + args.task + '_log', 'a')

f.write(f'############################### TASK {args.task} STAGE {args.stage} ######################### \n')

if args.type == 'cifar10':
    trainloader, test_loader = dataset.get10(batch_size=args.batch_size)
    model = detector_net.Net()

if args.type == 'cifar100':
    trainloader, test_loader = dataset.get100(batch_size=args.batch_size)
    model = detector_net.Net()

    model.conv1 = nn.Conv2d(3, out1, kernel_size=3, padding=1,bias=True)
    model.conv2 = nn.Conv2d(out1, out2, kernel_size=3, padding=1, bias=True)
    model.conv3 = nn.Conv2d(out2, out3, kernel_size=3, padding=1, bias=True)
    model.conv4 = nn.Conv2d(out3, 32, kernel_size=3, padding=1, bias=True)

print(model)

la_list = [args.la, args.la+0.4, args.la+1.4, 2.5, 3.0]
lr_list = [0.002, 0.002, 0.0005, 2.5, 3.0]

args.la = la_list[args.stage]
args.lr = lr_list[args.stage]

model = torch.nn.DataParallel(model)
if args.stage >0:
    print(str(args.save_pth)+'/task_'+str(args.task)+'_stage_'+str((args.stage)-1))
    load_file = torch.load(str(args.save_pth)+'/task_'+str(args.task)+'_stage_'+str(args.stage-1), map_location='cpu')
    try:
        model.load_state_dict(load_file.state_dict())
    except:
        model.load_state_dict(load_file)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.6)

if args.stage >= 1:
    model.module.conv1.weight.requires_grad = False
    model.module.conv1.bias.requires_grad = False
if args.stage >= 2:
    model.module.conv2.weight.requires_grad = False
    model.module.conv2.bias.requires_grad = False

if args.stage >= 3:
    model.module.conv3.weight.requires_grad = False
    model.module.conv3.bias.requires_grad = False

if args.stage >= 4:
    model.module.conv4.weight.requires_grad = False
    model.module.conv4.bias.requires_grad = False


best_acc, old_file = 0, None
best_loss = 1000
best_sep = 0
lambda_adv = torch.tensor([args.la]).cuda()
lambda_clean = torch.tensor([args.lc]).cuda()
t_begin = time.time()
mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()

print(args.train_pth)
test_loader_adv = torch.load(args.test_pth, map_location='cpu')
train_loader = torch.load(args.train_pth, map_location='cpu')
n_batches = int(250 * args.p * 0.01)

f.write(f'lambda_clean {lambda_clean}, lambda_adv {lambda_adv}, lr {args.lr}, n_batches {n_batches}')
print(f'lambda_clean {lambda_clean}, lambda_adv {lambda_adv}, lr {args.lr}, n_batches {n_batches}')
for epoch in range(args.epochs):
    print(f'epoch : {epoch}')
    model.train()
    random.shuffle(train_loader)
    # print("training phase")

    for batch_idx, (data, target, y) in enumerate(train_loader):
        # print(batch_idx)

        if batch_idx < n_batches:
            indx_target = target.clone()
            data, target, y = data.cuda(), target.cuda(), y.cuda()
            data, target, y  = Variable(data), Variable(target), Variable(y)
            optimizer.zero_grad()
            # output = model(data)
            # x = data
            # for i in range(8):
            #     x = model.module.features[i](x)
            sum_of_I = model(data, args.stage)
            # sum_of_I = x.abs().mean(dim=1).mean(dim=1).mean(dim=1)
            # sum_of_I = model.module.features[3](data).abs().mean(dim=1).mean(dim=1).mean(dim=1)
            # print(sum_of_I.size())
            loss =  ((y)*mseloss(sum_of_I,lambda_adv))+((1-y)*mseloss(sum_of_I,lambda_clean))
            loss.sum().backward()

            optimizer.step()

        # if batch_idx % args.log_interval == 0 and batch_idx > 0:

            # print(f'adv_energy_loss : {(y*mseloss(sum_of_I,lambda_adv)).sum()} clean_energy_loss : {((1-y)*mseloss(sum_of_I,lambda_clean)).sum()}')

    elapse_time = time.time() - t_begin
    speed_epoch = elapse_time / (epoch + 1)
    speed_batch = speed_epoch / 50000
    eta = speed_epoch * args.epochs - elapse_time

    if epoch % args.test_interval == 0:
        model.eval()
        test_loss = 0
        correct = 0
        energy_a_loss = 0
        energy_avg = 0
        energy_mean_a = 0
        # print("testing phase adv")
        soi_list_a = []
        # print(len(test_loader_adv), len(test_loader_adv[0]))
        for i, (data, target) in enumerate(test_loader_adv):
            indx_target = target.clone()
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                energy_a = model(data, args.stage)
                # x = data
                # for i in range(8):
                #     x = model.module.features[i](x)
                # energy_a = x.abs().mean(dim=1).mean(dim=1).mean(dim=1)
                # energy_a = model.module.features[3](data).abs().mean(dim=1).mean(dim=1).mean(dim=1)
                energy_mean_a += energy_a.mean()
                energy_a_loss += mseloss(energy_a,lambda_adv)

            tuple_da = (energy_a, target)

            soi_list_a.append(tuple_da)
        energy_a_loss = energy_a_loss/len(test_loader_adv)
        mean_soi_a = energy_mean_a/len(test_loader_adv)
        # path_a = args.save_pth+'/soi_adv_'+args.type+'_epoch'+str(epoch)
        # print(f'saving adv soi at {path_a}')
        # torch.save(soi_list_a, path_a)
        f.write(f'energy loss a : {energy_a_loss} mean_soi_a {mean_soi_a} \n')

        energy_c = 0
        test_loss = 0
        correct = 0
        energy_c_loss = 0
        energy_mean_c = 0
        soi_list_c = []
        # print("testing phase clean")
        for i, (data, target) in enumerate(test_loader):
            indx_target = target.clone()
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
                energy_c = model(data, args.stage)
                # x = data
                # for i in range(8):
                #     x = model.module.features[i](x)
                # energy_c = x.abs().mean(dim=1).mean(dim=1).mean(dim=1)
                # energy_c = model.module.features[3](data).abs().mean(dim=1).mean(dim=1).mean(dim=1)
                energy_mean_c += energy_c.mean()
                energy_c_loss += mseloss(energy_c, lambda_clean)
            tuple_dc = (energy_c, target)
            soi_list_c.append(tuple_dc)
        energy_c_loss = energy_c_loss/len(test_loader)
        mean_soi_c = energy_mean_c / len(test_loader)
        # path_c = args.save_pth + '/soi_clean_' + args.type + '_epoch' + str(epoch)
        # print(f'saving clean soi at {path_c}')
        # torch.save(soi_list_c, path_c)
        f.write(f'energy loss c : {energy_c_loss} mean_soi_c {mean_soi_c} \n')

        loss_energy = energy_c_loss + energy_a_loss
        # print(f'total loss: {loss_energy}')
        f.write(f'sep: {mean_soi_a - mean_soi_c} \n')
        if (mean_soi_a - mean_soi_c) > best_sep:
        # if loss_energy < best_loss:
        #     new_file = args.save_pth+'/no_bn_momentum0.6_500_layer3_la='+str(args.la)+'_od_'+args.type+'_'+args.pgd_params+'.pth'
            new_file = args.save_pth+'/task_'+str(args.task)+'_stage_'+str(args.stage)+'_out_'+str(out1)
            f.write(f'saving at {new_file} \n')
            torch.save(model, new_file)
            best_loss = loss_energy
            best_sep = mean_soi_a - mean_soi_c #mean_soi_a - mean_soi_c
        f.write(f'best sep {best_sep}')
        f.write(f'best loss {best_loss} \n')

f.close()


