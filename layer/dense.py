import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class Dense(nn.Module):
    def __init__(self, D_in, D_out, use_bias = True):
        super(Dense, self).__init__()
        #self.d = nn.Linear(D_in, D_out, use_bias)

        self.D_in = D_in
        self.D_out = D_out
        self.use_bias = use_bias

        self.weight = Parameter(torch.Tensor(D_in, D_out))
        if use_bias:
            self.bias = Parameter(torch.Tensor(D_out))
        else:
            self.register_parameter('bias', None)

        ##initialization
        variance = np.sqrt(2.0/(D_in + D_out))
        self.weight.data.normal_(0.0, variance)
        if use_bias:
            self.bias.data.fill_(0.0)


    def forward(self, x):
        if self.use_bias:
            return torch.mm(x, self.weight) + self.bias
        else:
            return torch.mm(x, self.weight)




if __name__ == '__main__':
    d1 = Dense()
    print(d1.a)