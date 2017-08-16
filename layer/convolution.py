import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class Conv2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, mode = "same", dilation = 1, groups = 1, use_bias = True):
        """
        :param C_in:        (int), input channel 
        :param C_out:       (int), output channel
        :param kernel_size: (tuple), (H_kernel, W_kernel)
        :param stride:      (tuple), (H_stride, W_stride)
        :param mode:        (string), "same", "full", or "valid", (default : same)
        :param dilation:    (do not touch yet)
        :param groups:      (do not touch yet)
        :param use_bias:    (bool), if true, bias will be added
        """
        super(Conv2D, self).__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size ##tuple
        self.stride = stride ##tuple
        self.mode = mode #same, valid, full
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

        ##if mode == valid : 0, same : floor(n_h / 2), full : n_h - 1

        if mode == "valid":
            p_h = 0
            p_w = 0
        elif mode == "same":
            p_h = np.floor(kernel_size[0] / 2).astype('int')
            p_w = np.floor(kernel_size[1] / 2).astype('int')
        elif mode == "full":
            p_h = kernel_size[0]-1
            p_w = kernel_size[1]-1
        else:
            print('convolution type error!')
            raise NotImplementedError

        self.padding = (p_h, p_w)


        self.weight = Parameter(torch.Tensor(C_in, C_out, kernel_size[0], kernel_size[1]))
        if use_bias:
            self.bias = Parameter(torch.Tensor(C_out))
        else:
            self.register_parameter('bias', None)

        #initialize
        variance = np.sqrt(2.0 / (C_in*kernel_size[0]*kernel_size[1] + C_out*kernel_size[0]*kernel_size[1]))
        self.weight.data.normal_(0.0, variance)
        if use_bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)