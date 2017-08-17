import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import torch

import torch.optim as optim
from torchvision import datasets, transforms

from functions import train_vanilla
from models.mlps import MLP_3LAYER_MNIST


def main():
    batch_size = 100
    n_epoch = 30
    lr = 0.001

    model = MLP_3LAYER_MNIST()
    model.summary()

    model_path = '../results/parameters/'
    log_path = '../results/logs/'
    model_name = 'mlp_mnist_3layers'
    data_path = '../data/mnist'

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs
    )

    print(' --- data summary ---')
    print('n_train : ', len(train_loader.dataset))
    print('n_test : ', len(test_loader.dataset))
    print('len_loader : ', len(train_loader))
    print(' ---------------------')

    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_vanilla(model, optimizer, train_loader, test_loader,
                  model_path = model_path, model_name = model_name, log_path = log_path,
                  n_epoch = n_epoch)



if __name__ == '__main__':
    main()