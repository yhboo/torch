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
        self.c1 = Conv2D(3, 64, (3,3), (1,1), "same") #dropout 0.3
        self.c1_bn = nn.BatchNorm2d(64)
        self.c1_drop = nn.Dropout2d(0.3)
        self.c2 = Conv2D(64, 64, (3,3), (1,1), "same")
        self.c2_bn = nn.BatchNorm2d(64)
        #max-pooling
        self.c3 = Conv2D(64, 128, (3,3), (1,1), "same") #dropout 0.4
        self.c3_bn = nn.BatchNorm2d(128)
        self.c3_drop = nn.Dropout2d(0.4)
        self.c4 = Conv2D(128, 128, (3,3), (1,1), "same")
        self.c4_bn = nn.BatchNorm2d(128)
        #max-pooling
        self.c5 = Conv2D(128, 256, (3,3), (1,1), "same") #dropout 0.4
        self.c5_bn = nn.BatchNorm2d(256)
        self.c5_drop = nn.Dropout2d(0.4)
        self.c6 = Conv2D(256, 256, (3,3), (1,1), "same") #dropout 0.4
        self.c6_bn = nn.BatchNorm2d(256)
        self.c6_drop = nn.Dropout2d(0.4)
        self.c7 = Conv2D(256, 256, (3, 3), (1, 1), "same")
        self.c7_bn = nn.BatchNorm2d(256)
        #max-pooling
        self.c8 = Conv2D(256, 512, (3, 3), (1, 1), "same")  # dropout 0.4
        self.c8_bn = nn.BatchNorm2d(512)
        self.c8_drop = nn.Dropout2d(0.4)
        self.c9 = Conv2D(512, 512, (3, 3), (1, 1), "same")  # dropout 0.4
        self.c9_bn = nn.BatchNorm2d(512)
        self.c9_drop = nn.Dropout2d(0.4)
        self.c10 = Conv2D(512, 512, (3, 3), (1, 1), "same")
        self.c10_bn = nn.BatchNorm2d(512)
        #max-pooling
        self.c11 = Conv2D(512, 512, (3, 3), (1, 1), "same")  # dropout 0.4
        self.c11_bn = nn.BatchNorm2d(512)
        self.c11_drop = nn.Dropout2d(0.4)
        self.c12 = Conv2D(512, 512, (3, 3), (1, 1), "same")  # dropout 0.4
        self.c12_bn = nn.BatchNorm2d(512)
        self.c12_drop = nn.Dropout2d(0.4)
        self.c13 = Conv2D(512, 512, (3, 3), (1, 1), "same")
        self.c13_bn = nn.BatchNorm2d(512)
        # max-pooling
        # dropout 0.5
        self.d0_drop = nn.Dropout(0.5)
        self.d1 = Dense(512, 512)  #dropout 0.5
        self.d1_bn = nn.BatchNorm1d(512)
        self.d1_drop = nn.Dropout(0.5)
        self.d2 = Dense(512, 10)

    def forward(self, x):
        x = F.relu(self.c1_bn(self.c1(x))) #conv1
        x = self.c1_drop(x) #dropout
        x = F.relu(F.max_pool2d(self.c2_bn(self.c2(x)), kernel_size = 2, stride = 2)) #conv2, maxpooling

        x = F.relu(self.c3_bn(self.c3(x))) #conv3
        x = self.c3_drop(x) #dropout
        x = F.relu(F.max_pool2d(self.c4_bn(self.c4(x)),2, 2)) #conv4, maxpooling

        x = F.relu(self.c5_bn(self.c5(x))) #conv5
        x = self.c5_drop(x) #dropout
        x = F.relu(self.c6_bn(self.c6(x)))  # conv6
        x = self.c6_drop(x)  # dropout
        x = F.relu(F.max_pool2d(self.c7_bn(self.c7(x)),2, 2)) #conv7, maxpooling

        x = F.relu(self.c8_bn(self.c8(x)))  # conv8
        x = self.c8_drop(x)  # dropout
        x = F.relu(self.c9_bn(self.c9(x)))  # conv9
        x = self.c9_drop(x)  # dropout
        x = F.relu(F.max_pool2d(self.c10_bn(self.c10(x)), 2, 2))  # conv10, maxpooling

        x = F.relu(self.c11_bn(self.c11(x)))  # conv11
        x = self.c11_drop(x)  # dropout
        x = F.relu(self.c12_bn(self.c12(x)))  # conv12
        x = self.c12_drop(x)  # dropout
        x = F.relu(F.max_pool2d(self.c13_bn(self.c13(x)), 2, 2))  # conv13, maxpooling

        x = x.view(-1,512) #flatten

        x = self.d0_drop(x) #dropout
        x = F.relu(self.d1_bn(self.d1(x))) #dense1
        x = self.d1_drop(x) #dropout
        x = self.d2(x) #dense2

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
