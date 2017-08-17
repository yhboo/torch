import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layer.convolution import Conv2D
from layer.dense import Dense

class VGG_CIFAR10_TORCH(nn.Module):
    def __init__(self, name = "VGG_CIFAR10_TORCH"):
        super(VGG_CIFAR10_TORCH, self).__init__()
        self.name = name
        self.c1 = Conv2D(3, 64, (3,3), (1,1), "same")
        self.c1_bn = nn.BatchNorm2d(64)
        self.c2 = Conv2D(128, 128, (3,3), (1,1), "same")
        self.c2_bn = nn.BatchNorm2d(128)
        #max-pooling
        self.c3 = Conv2D(128, 256, (3,3), (1,1), "same")
        self.c3_bn = nn.BatchNorm2d(256)
        self.c4 = Conv2D(256, 256, (3,3), (1,1), "same")
        self.c4_bn = nn.BatchNorm2d(256)
        #max-pooling
        self.c5 = Conv2D(256, 512, (3,3), (1,1), "same")
        self.c5_bn = nn.BatchNorm2d(512)
        self.c6 = Conv2D(512, 512, (3,3), (1,1), "same")
        self.c6_bn = nn.BatchNorm2d(512)
        #max-pooling
        self.d1 = Dense(8192, 1024)
        self.d2 = Dense(1024, 1024)
        self.d3 = Dense(1024, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(F.max_pool2d(self.c2(x), kernel_size = 3, stride = 2, padding = 1))
        x = F.relu(self.c3(x))
        x = F.relu(F.max_pool2d(self.c4(x),3, 2, 1))
        x = F.relu(self.c5(x))
        x = F.relu(F.max_pool2d(self.c6(x),3, 2, 1))
        x = x.view(-1,8192)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        return x

    def summary(self):
        n_params = 0
        print('-------',self.name,' summary------')
        for name, param in self.state_dict().items():
            print(name, param.size())
            n_params+=param.numel()
        print('total parameters : ', n_params)

class VGG9_CIFAR10(nn.Module):
    def __init__(self, name = "VGG9_CIFAR10"):
        super(VGG9_CIFAR10, self).__init__()
        self.name = name
        self.c1 = Conv2D(3, 128, (3,3), (1,1), "same")
        self.c2 = Conv2D(128, 128, (3,3), (1,1), "same")
        #max-pooling
        self.c3 = Conv2D(128, 256, (3,3), (1,1), "same")
        self.c4 = Conv2D(256, 256, (3,3), (1,1), "same")
        #max-pooling
        self.c5 = Conv2D(256, 512, (3,3), (1,1), "same")
        self.c6 = Conv2D(512, 512, (3,3), (1,1), "same")
        #max-pooling
        self.d1 = Dense(8192, 1024)
        self.d2 = Dense(1024, 1024)
        self.d3 = Dense(1024, 10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(F.max_pool2d(self.c2(x), kernel_size = 3, stride = 2, padding = 1))
        x = F.relu(self.c3(x))
        x = F.relu(F.max_pool2d(self.c4(x),3, 2, 1))
        x = F.relu(self.c5(x))
        x = F.relu(F.max_pool2d(self.c6(x),3, 2, 1))
        x = x.view(-1,8192)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        return x

    def summary(self):
        n_params = 0
        print('-------',self.name,' summary------')
        for name, param in self.state_dict().items():
            print(name, param.size())
            n_params+=param.numel()
        print('total parameters : ', n_params)
