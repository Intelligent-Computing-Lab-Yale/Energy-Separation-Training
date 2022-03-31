import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import adv_attacks

import torchvision
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
import dataset
from datetime import datetime
import torchvision.models as models
import os
import time
import numpy as np
import copy
# import torchattacks

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar10', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 200)')
parser.add_argument('--atype', default='fgsm', help='fgsm, pgd')
parser.add_argument('--eps', type=float, default=0, help='epsilon value')
parser.add_argument('--alpha', type=float, default=0, help='alpha value')
parser.add_argument('--steps', type=int, default=0, help='steps value')
parser.add_argument('--path', default='.', help='path where the adversarial test set should be stored')
parser.add_argument('--model_pth', default='.', help='model checkpoint file to use for adversarial data generation')
parser.add_argument('--cw_param', default='0,1e-4,100')
parser.add_argument('--task', default='any_task')
parser.add_argument('--model', default='vgg16')
parser.add_argument('--per', type=float, default=1, help='alpha value')

args = parser.parse_args()

if args.type == 'cifar10':
    print('loading cifar10 dataset')
    train_loader, test_loader = dataset.get10(batch_size=args.batch_size, num_workers=1, train= True)
elif args.type == 'cifar100':
    print('loading cifar100 dataset')
    train_loader, test_loader = dataset.get100(batch_size=args.batch_size, num_workers=1, train=True)
elif args.type == 'tinyimagenet':
    print('loading tinyimagenet dataset')
    train_loader, test_loader = dataset.tinyimagenet(batch_size=args.batch_size)

model_pth =  args.model_pth
print(f'model path for adversarial dataset generation: {model_pth}')

if args.type == 'cifar10':
    if args.model == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if args.model == 'vgg16':
        net = torchvision.models.vgg16(pretrained=False, num_classes=10)
    if args.model == 'mobilenet_v2':
        net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)

if args.type == 'cifar100':
    if args.model=='resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=100)
    if args.model == 'vgg16':
        net = torchvision.models.vgg16(pretrained=False, num_classes=100)
    if args.model == 'mobilenet_v2':
        net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=100)

if args.type == 'tinyimagenet':
    if args.model=='resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=200)
    if args.model == 'vgg16':
        net = torchvision.models.vgg16(pretrained=False, num_classes=200)
    if args.model == 'mobilenet_v2':
        net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=200)

net = torch.nn.DataParallel(net)
try:
    net.load_state_dict(torch.load(model_pth).state_dict())
except:
    net.load_state_dict(torch.load(model_pth)['state_dict'])
net = net.cuda()
# FGSM attack code
# def fgsm_attack(image, epsilon, data_grad):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     if epsilon!=0:
#         perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # Return the perturbed image
#     return perturbed_image

from torch.autograd import Variable
def pgd_attack(net, device, testloader, n_batches ):
    train_dataset = []
    for batch_idx, data in enumerate(testloader, 0):
        # get the inputs
        # print('hi there')
        if batch_idx < n_batches:
            inputs, labels_tru = data

            # wrap them in Variable
            inp_var, true_label = Variable(inputs.cuda(), requires_grad=True), Variable(labels_tru.cuda()
                                                                                        , requires_grad=False)
            inp_adv = adv_attacks.pgd_attack(net, inp_var, true_label, args.eps, args.alpha, args.steps)
            ifadv = torch.ones(200)
            ifnotadv = torch.zeros(200)
            adv_da_tuple = (inp_adv, labels_tru, ifadv)
            clean_da_tuple = (inp_var, labels_tru, ifnotadv)

            train_dataset.append(adv_da_tuple)
            train_dataset.append(clean_da_tuple)
            print(batch_idx)
            if args.type == 'tinyimagenet' and (batch_idx+1) % 100 == 0:
                # print(batch_idx)
                print(args.path+'/set_task_'+args.task+'_'+str(batch_idx))
                print(len(train_dataset))
                torch.save(train_dataset, args.path+'/set_task_'+args.task+'_'+str(batch_idx))
                train_dataset = []

        # if batch_idx == len(testloader)-1:
        #     torch.save(train_dataset, args.path + 'set' + str(count))
    return train_dataset



def test_attack( model, device, testloader ):
  model.eval()
  train_dataset = []

  # Loop over all examples in test set
  for i, (data, target) in enumerate(testloader):

      # Send the data and label to the device
      data, target = data.to(device), target.to(device)

      # Set requires_grad attribute of tensor. Important for Attack
      data.requires_grad = True

      # Forward pass the data through the model
      output = model(data)

      # Calculate the loss
      loss = F.cross_entropy(output, target)

      # Zero all existing gradients
      model.zero_grad()

      # Calculate gradients of model in backward pass
      loss.backward()

      # Collect datagrad
      data_grad = data.grad.data

      # Call FGSM Attack
      perturbed_data = adv_attacks.fgsm_attack(data, args.eps, data_grad)
      # pert = (perturbed_data-data).abs().mean()
      # print(pert)

      ifadv = torch.ones(200).cpu()
      ifnotadv = torch.zeros(200).cpu()
      adv_da_tuple = (perturbed_data.cpu(), target.cpu(),ifadv)

      clean_da_tuple = (data.cpu(), target.cpu(), ifnotadv)
      # print('hello')
      train_dataset.append(adv_da_tuple)

      train_dataset.append(clean_da_tuple)

  return train_dataset

if args.atype == 'fgsm':
    print('doing FGSM')
    train_dataset = test_attack( net, 'cuda', train_loader)
    file = args.path+'/adv_train_set_'+args.type+'_fgsm_e=' + str(args.eps)
    print(f'file saved at {file}')
    torch.save(train_dataset, file)

elif args.atype == 'pgd':
    print('doing PGD')
    # n_batches = args.percent_dataset * 0.01 * len(train_loader)
    # print(f'length n_batches {n_batches}')
    data_batches = len(train_loader)
    n_batches = int(data_batches*args.per*0.01)
    train_dataset = pgd_attack(net, 'cuda', train_loader, n_batches)

    # file = args.path+'/adv_train_set_'+args.type+'_pgd_e=' + str(args.eps) + '_a='+str(args.alpha)+'_n='+str(args.steps)+'_p_'+str(args.percent_dataset)
    file = args.path+'/adv_train_set_'+args.type+'_task_'+args.task

    print(f'file saved at {file}')
    torch.save(train_dataset, file)

elif args.atype == 'cw':
    kappa, c, steps = map(float, args.cw_param.split(','))
    steps = int(steps)
    attack = torchattacks.CW(net, c=c, kappa=kappa, steps=steps)
    file = args.path+'/adv_train_set_'+args.type+'_cw_k=' + str(kappa) + '_c='+str(c)+'_n='+str(steps)

    train_dataset = []
    for i, (data, labels_tru) in enumerate(train_loader):

        inp_adv = attack(data.cuda(), labels_tru.cuda())
        ifadv = torch.ones(200).cpu()
        ifnotadv = torch.zeros(200).cpu()
        adv_da_tuple = (inp_adv.cpu(), labels_tru.cpu(), ifadv)

        clean_da_tuple = (data.cpu(), labels_tru.cpu(), ifnotadv)
        train_dataset.append(adv_da_tuple)
        train_dataset.append(clean_da_tuple)

    print(f'file saved at {file}')
    torch.save(train_dataset, file)
elif args.atype == 'square':
    # kappa, c, steps = map(float, args.cw_param.split(','))
    # steps = int(steps)
    eps = 2.5
    attack = torchattacks.Square(net, eps=2.5, n_queries=10, n_restarts=1, loss='ce', p_init=0.8)

    # attack = torchattacks.(net, c=c, kappa=kappa, steps=steps)
    file = args.path+'/adv_train_set_'+args.type+'_square_e=' + str(2.5)

    train_dataset = []
    for i, (data, labels_tru) in enumerate(train_loader):
        print(i)
        inp_adv = attack(data.cuda(), labels_tru.cuda())
        ifadv = torch.ones(200).cpu()
        ifnotadv = torch.zeros(200).cpu()
        adv_da_tuple = (inp_adv.cpu(), labels_tru.cpu(), ifadv)

        clean_da_tuple = (data.cpu(), labels_tru.cpu(), ifnotadv)
        train_dataset.append(adv_da_tuple)
        train_dataset.append(clean_da_tuple)

    print(f'file saved at {file}')
    torch.save(train_dataset, file)


