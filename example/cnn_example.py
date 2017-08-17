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
from models.cnns import *

"""
CNN example
data : CIFAR-10
"""

def main():
    batch_size = 128
    n_epoch = 250
    lr = 0.1
    lr_decay_step = 25
    lr_decay_factor = 0.5
    n_worker = 8
    opt_type = "sgd"
    print(' --- parameters summary ---')
    print('batch_size : ', batch_size)
    print('n_epoch    : ', n_epoch)
    print('lr         : ', lr)
    print('decay_step : ', lr_decay_step)
    print('decay_ratio: ', lr_decay_factor)
    print('n_worker   : ', n_worker)
    print('optimizer  : ', opt_type)

    model = VGG_CIFAR10_TORCH()
    model.summary()

    model_path = '../results/parameters/'
    log_path = '../results/logs/'
    exp_name = 'cifar10_vgg_from_torch'
    data_path = '../data/cifar10'

    kwargs = {'num_workers': n_worker, 'pin_memory': True}



    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train = True, download=True,
                         transform = transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
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
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum= 0.9, weight_decay= 0.0005)

    """
    train_vanilla(model, optimizer, train_loader, test_loader,
                  model_path = model_path, exp_name = exp_name, log_path = log_path,
                  n_epoch = n_epoch)
    """

    #lr scheduling
    lr_list = np.ones(n_epoch)
    cur_lr = lr
    for i in range(n_epoch):
        lr_list[i] = cur_lr
        if (i + 1) % lr_decay_step == 0:
            cur_lr *=lr_decay_factor

    #print('lr list : ', lr_list)

    train_lr_per_epoch(model, optimizer, train_loader, test_loader, lr = lr_list,
                       model_path = model_path, exp_name = exp_name, log_path = log_path,
                       n_epoch = n_epoch)


if __name__ == '__main__':
    main()