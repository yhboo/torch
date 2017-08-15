import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

class Dense(torch.nn.Module):
    def __init__(self, D_in, D_out, use_bias = True):
        super(Dense, self).__init__()
        self.d = torch.nn.Linear(D_in, D_out, use_bias)


    def forward(self, x):
        return self.d(x)


    


if __name__ == '__main__':
    d1 = Dense()
    print(d1.a)