import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


class Conv2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, mode, dilation = 1, groups = 1, use_bias = True):
        super(Conv2D, self).__init__()

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size ##tuple
        self.stride = stride ##tuple
        self.mode = mode #same, valid, full
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

        self.pading = 0 ##if mode == valid : 0, same : floor(n_h / 2), full : n_h - 1?? check

        self.weight = torch.Tensor(C_in, C_out, kernel_size[0], kernel_size[1])
        if use_bias:
            self.bias = torch.Tensror(C_out)
        else:
            self.register_parameter('bias', None)

        #initialize
        variance = np.sqrt(2.0 / (C_in + C_out))
        self.weight.data.normal_(0.0, variance)
        if use_bias:
            self.bias.data.fill_(0.0)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)