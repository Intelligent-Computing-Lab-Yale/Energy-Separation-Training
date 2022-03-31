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

model_pth = args.model_pth
print(f'model path for adversarial dataset generation: {model_pth}')
if args.type == 'cifar10':
    if args.model == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if args.model == 'vgg16':
        net = torchvision.models.vgg16(pretrained=False, num_classes=10)
    if args.model == 'mobilenet_v2':
        net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=10)

if args.type == 'cifar100':
    # net = torchvision.models.resnet18(pretrained=False, num_classes=100).cuda()
    if args.model == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=100)
    if args.model == 'vgg16':
        net = torchvision.models.vgg16(pretrained=False, num_classes=100)
    if args.model == 'mobilenet_v2':
        net = torchvision.models.mobilenet_v2(pretrained=False, num_classes=100)

elif args.type == 'tinyimagenet':
    if args.model == 'resnet18':
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
#net.load_state_dict(torch.load(model_pth)['state_dict']) #.state_dict())

net = net.cuda()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    if epsilon!=0:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

from torch.autograd import Variable
def pgd_attack(net, device, testloader):
    adv_data_batch = []
    adv_label_batch = []
    adv_test_set = []
    count = 0
    print(f'steps, alpha, epsilon= {args.steps, args.alpha, args.eps}')
    for batch_idx, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels_tru = data

        # wrap them in Variable
        inp_var, true_label = Variable(inputs.cuda(), requires_grad=True), Variable(labels_tru.cuda()
                                                                                    , requires_grad=False)

        inp_adv = adv_attacks.pgd_attack(net, inp_var, true_label, args.eps, args.alpha, args.steps)

        if count < 4:
            adv_data_batch.append(inp_adv)
            adv_label_batch.append(labels_tru)
            count += 1

        else:

            adv_data_batch.append(inp_adv)
            adv_label_batch.append(labels_tru)

            Tdata_batch = torch.cat(
                [adv_data_batch[0], adv_data_batch[1], adv_data_batch[2], adv_data_batch[3], adv_data_batch[4]], dim=0)
            Tlabel_batch = torch.cat(
                [adv_label_batch[0], adv_label_batch[1], adv_label_batch[2], adv_label_batch[3], adv_label_batch[4]],
                dim=0)
            da_tuple = (Tdata_batch, Tlabel_batch)
            if args.type == 'tinyimagenet':
                print(args.path + '/set' + str(batch_idx))
                torch.save(da_tuple, args.path + '/test_set_task_'+args.task+'_'+ str(batch_idx))
                # print(len(da_tuple))
                adv_test_set = []
            else:
                adv_test_set.append(da_tuple)
            adv_data_batch = []
            adv_label_batch = []
            count = 0

    return adv_test_set




def test_attack( model, device, testloader ):
  model.eval()
  # Accuracy counter
  correct, correct_adv = 0, 0
  adv_examples = []
  adv_test = []
  adv_data_batch = []
  adv_label_batch = []
  adv_test_set = []
  adv_data = []
  adv_label = []
  count = 0
  print(f'FGSM attack with epsilon: {args.eps}')
  # Loop over all examples in test set
  for data, target in testloader:

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
      # da_tuple = (perturbed_data, target)
      if count < 4:
          adv_data_batch.append(perturbed_data)
          adv_label_batch.append(target)
          count += 1

      else:

          adv_data_batch.append(perturbed_data)
          adv_label_batch.append(target)
          Tdata_batch = torch.cat([adv_data_batch[0], adv_data_batch[1], adv_data_batch[2], adv_data_batch[3], adv_data_batch[4]], dim=0)
          Tlabel_batch = torch.cat([adv_label_batch[0], adv_label_batch[1], adv_label_batch[2], adv_label_batch[3], adv_label_batch[4]], dim=0)
          da_tuple = (Tdata_batch, Tlabel_batch)
          adv_test_set.append(da_tuple)
          adv_data_batch = []
          adv_label_batch = []
          count = 0

  return adv_test_set

def random_noise(model, device, testloader):
    model.eval()
    train_dataset = []

    adv_data_batch = []
    adv_label_batch = []
    adv_test_set = []
    adv_data = []
    adv_label = []
    count = 0

    # Loop over all examples in test set
    for i, (data, target) in enumerate(testloader):
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        ifadv = torch.ones(200).cpu()
        ifnotadv = torch.zeros(200).cpu()
        perturbed_data = data+ torch.randn(size=(data.size())).cuda()

        if count < 4:
            adv_data_batch.append(perturbed_data)
            adv_label_batch.append(target)
            count += 1

        else:

            adv_data_batch.append(perturbed_data)
            adv_label_batch.append(target)
            print(f' shape_data: {len(adv_data_batch)} ; {adv_data_batch[0].size()}')
            Tdata_batch = torch.cat(
                [adv_data_batch[0], adv_data_batch[1], adv_data_batch[2], adv_data_batch[3], adv_data_batch[4]], dim=0)
            Tlabel_batch = torch.cat(
                [adv_label_batch[0], adv_label_batch[1], adv_label_batch[2], adv_label_batch[3], adv_label_batch[4]],
                dim=0)
            da_tuple = (Tdata_batch, Tlabel_batch)
            adv_test_set.append(da_tuple)
            adv_data_batch = []
            adv_label_batch = []
            count = 0

    return adv_test_set

if args.atype == 'fgsm':
    print('doing FGSM')
    adv_test_set = test_attack( net, 'cuda', test_loader)
    file = args.path+'/adv_test_set_'+args.type+'_fgsm_e=' + str(args.eps)
    print(f'file saved at {file}')
    torch.save(adv_test_set, file)

elif args.atype == 'pgd':
    print('doing PGD')
    adv_test_set = pgd_attack(net, 'cuda', test_loader)
    # file = args.path+'/adv_test_set_'+args.type+'_pgd_e=' + str(args.eps) + '_a='+str(args.alpha)+'_n='+str(args.steps)
    file = args.path+'/adv_test_set_'+args.type+'_task_'+args.task
    print(f'file saved at {file}')
    torch.save(adv_test_set, file)

elif args.atype == 'cw':
    kappa, c, steps = map(float, args.cw_param.split(','))
    steps = int(steps)
    attack = torchattacks.CW(net, c=c, kappa=kappa, steps=steps)
    file = args.path+'/adv_test_set_'+args.type+'_cw_k=' + str(kappa) + '_c='+str(c)+'_n='+str(steps)

    train_dataset = []
    for i, (data, labels_tru) in enumerate(test_loader):

        inp_adv = attack(data.cuda(), labels_tru.cuda())
        # ifadv = torch.ones(200).cpu()
        # ifnotadv = torch.zeros(200).cpu()
        adv_da_tuple = (inp_adv.cpu(), labels_tru.cpu())

        # clean_da_tuple = (data.cpu(), labels_tru.cpu(), ifnotadv)
        train_dataset.append(adv_da_tuple)
        # train_dataset.append(clean_da_tuple)

    print(f'file saved at {file}')
    torch.save(train_dataset, file)

elif args.atype == 'square':
    # kappa, c, steps = map(float, args.cw_param.split(','))
    # steps = int(steps)
    eps = 2.5
    attack = torchattacks.Square(net, eps=2.5, n_queries=10, n_restarts=1, loss='ce', p_init=0.8)

    file = args.path+'/adv_test_set_'+args.type+'_square_w=' + str(eps)

    train_dataset = []
    for i, (data, labels_tru) in enumerate(test_loader):

        inp_adv = attack(data.cuda(), labels_tru.cuda())
        # ifadv = torch.ones(200).cpu()
        # ifnotadv = torch.zeros(200).cpu()
        adv_da_tuple = (inp_adv.cpu(), labels_tru.cpu())

        # clean_da_tuple = (data.cpu(), labels_tru.cpu(), ifnotadv)
        train_dataset.append(adv_da_tuple)
        # train_dataset.append(clean_da_tuple)

    print(f'file saved at {file}')
    torch.save(train_dataset, file)
    
elif args.atype == 'random':
    print('doing random')
    adv_test_set = random_noise(net, 'cuda', test_loader)
    torch.save(adv_test_set, 'cifar10_random/adv_test_set_random')
