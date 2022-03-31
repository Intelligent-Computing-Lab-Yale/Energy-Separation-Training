import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import numpy as np
import matplotlib.pyplot as plt

size_1 = 64
size_2 = 64

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3 input channels for CIFAR10, VGG11 calls for 64 output channels from
        # the first conv layer, a batchnorm, then a ReLU
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1,bias=True)# 3 8
        # self.norm1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()

        self.mp1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1,bias=True)# 8 16
        # self.norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1,bias=True)# 16 32
        # self.norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1,bias=True)# 32 32
        # self.norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        #
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1,bias=True)# 32 64
        # self.norm5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()
        #
        # self.conv6 = nn.Conv2d(size_1, 64, kernel_size=3, padding=1)
        # self.norm6 = nn.BatchNorm2d(64)
        # self.relu6 = nn.ReLU()

    def forward(self, x0, stage):
        # print(x0.size())
        # self.conv1.weight = torch.nn.Parameter(self.conv1.weight.abs())
        x1 = self.conv1(x0)
        # feature = x1
        # x1,_ = BinActive2.apply(x1, 255)
        if stage == 0:

            soi = x1.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        # x1 = self.norm1(x1)
        # x1,_ = BinActive2.apply(x1, 255)
        x1 = self.relu1(x1)
        # x1,_ = BinActive2.apply(x1, 255)
        # x1 = self.mp1(x1)

        x2 = self.conv2(x1)
        # x2,_ = BinActive2.apply(x2, 255)
        if stage == 1:
            soi = x2.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        # x2 = self.norm2(x2)
        # x2,_ = BinActive2.apply(x2, 255)

        # soi = x2.abs().mean(dim=1).mean(dim=1).mean(dim=1)

        x2 = self.relu2(x2)
        # x2,_ = BinActive2.apply(x2, 255)

        # x2 = self.mp2(x2)
        x3 = self.conv3(x2)
        # x3,_ = BinActive2.apply(x3, 255)
        if stage == 2:
            soi = x3.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        # x3 = self.norm3(x3)
        # x3,_ = BinActive2.apply(x3, 255)

        # soi = x3.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        x3 = self.relu3(x3)
        # x3,_ = BinActive2.apply(x3, 255)

        # x3 = self.mp2(x3)
        x4 = self.conv4(x3)
        # x4,_ = BinActive2.apply(x4, 255)

        # soi = x4.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        # x4 = self.norm4(x4)
        # x4,_ = BinActive2.apply(x4, 255)

        x4 = self.relu4(x4)
        # x4,_ = BinActive2.apply(x4, 255)

        #
        x5 = self.conv5(x4)
        # x5,_ = BinActive2.apply(x5, 255)

        # soi = x5.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        # x5 = self.norm5(x5)
        # x5,_ = BinActive2.apply(x5, 255)

        # x5 = self.relu5(x5)
        # x5,_ = BinActive2.apply(x5, 255)

        #
        # x6 = self.conv6(x5)
        # # soi = x6.abs().mean(dim=1).mean(dim=1).mean(dim=1)
        # x6 = self.norm6(x6)
        # x6 = self.relu6(x6)

        return soi


