import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from functions import train_vanilla
from models.cnns import VGG9_CIFAR10

"""
CNN example
data : CIFAR-10
"""

def main():
    batch_size = 32
    n_epoch = 50
    lr = 0.001


    model = VGG9_CIFAR10()
    model.summary()

    model_path = 'results/parameters/'
    log_path = 'results/logs/'
    model_name = 'CIFAR10_VGG9'
    data_path = '../data/cifar10'

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train = True, download=True,
                         transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.RandomHorizontalFlip(),
                             transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                         ])),
        batch_size = batch_size, shuffle = True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train = False, download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.RandomHorizontalFlip(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                         ])),
        batch_size = batch_size, shuffle = False, **kwargs
    )

    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.size())
        print(target.size())
        if batch_idx == 1:
            break

    print(' --- data summary ---')
    print('n_train : ', len(train_loader.dataset))
    print('n_test : ', len(test_loader.dataset))
    print('len_loader : ',len(train_loader))
    print(' ---------------------')


    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum= 0.9)

    train_vanilla(model, optimizer, train_loader, test_loader,
                  model_path = model_path, model_name = model_name, log_path = log_path,
                  n_epoch = n_epoch)



if __name__ == '__main__':
    main()