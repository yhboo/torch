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

from functions import *
from models.cnns import VGG9_CIFAR10

"""
CNN example
data : CIFAR-10
"""

def main():
    batch_size = 64
    n_epoch = 30
    lr = 0.001
    n_worker = 2
    opt_type = "sgd"
    print(' --- parameters summary ---')
    print('batch_size : ', batch_size)
    print('n_epoch    : ', n_epoch)
    print('lr         : ', lr)
    print('n_worker   : ', n_worker)
    print('optimizer  : ', opt_type)

    model = VGG9_CIFAR10()
    model.summary()

    model_path = '../results/parameters/'
    log_path = '../results/logs/'
    exp_name = 'cifar10_vgg9_vanilla'
    data_path = '../data/cifar10'

    kwargs = {'num_workers': n_worker, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train = True, download=True,
                         transform = transforms.Compose([
                             transforms.ToTensor(),
                             #transforms.RandomHorizontalFlip(),
                             transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
                         ])),
        batch_size = batch_size, shuffle = True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train = False, download=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             #transforms.RandomHorizontalFlip(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                         ])),
        batch_size = batch_size, shuffle = False, **kwargs
    )

    print(' --- data summary ---')
    print('n_train : ', len(train_loader.dataset))
    print('n_test : ', len(test_loader.dataset))
    print('len_loader : ',len(train_loader))
    print(' ---------------------')


    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum= 0.9)

    """
    train_vanilla(model, optimizer, train_loader, test_loader,
                  model_path = model_path, exp_name = exp_name, log_path = log_path,
                  n_epoch = n_epoch)
    """
    lr_list = np.logspace(np.log10(lr), np.log10(lr*0.01), n_epoch)

    print('lr list : ', lr_list)
    train_lr_per_epoch(model, optimizer, train_loader, test_loader, lr = lr_list,
                       model_path = model_path, exp_name = exp_name, log_path = log_path,
                       n_epoch = n_epoch)


if __name__ == '__main__':
    main()