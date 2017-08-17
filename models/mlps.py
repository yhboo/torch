import torch.nn as nn
import torch.nn.functional as F
from layer.dense import Dense





class MLP_3LAYER_MNIST(nn.Module):
    def __init__(self, name = "MLP_MNIST"):
        self.name = name
        super(MLP_3LAYER_MNIST, self).__init__()
        self.d1 = Dense(784, 1024)
        self.d2 = Dense(1024, 1024)
        self.d3 = Dense(1024, 10)


    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x)
        return x



    def summary(self):
        n_params = 0
        print('-------',self.name,' summary------')
        for name, param in self.state_dict().items():
            print(name, param.size())
            n_params+=param.numel()
        print('total parameters : ', n_params)