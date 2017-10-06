import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class Model(nn.Module):

    def __init__(self, tied=False):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(30, 512)
        self.rnn = nn.LSTM(512, 512, 1, dropout=0.5)
        self.fc = nn.Linear(512, 30)

        self.init_weights()

        if tied:
            self.fc.weight = self.embedding.weight

    def init_weights(self):
        self.fc.bias.data.fill_(0)

    def forward(self, input_, state_):
        """
        input_: (sequence_length, batch_size)
        output: (sequence_length, batch_size, hidden_dim)
        logit:  (sequece_length, batch_size, output_dim)
        """
        emb = F.dropout(self.embedding(input_), 0.5, training=self.training)
        output, state = self.rnn(emb, state_)
        output = F.dropout(output, 0.5, training=self.training)
        output_reshape = output.view(output.size(0) * output.size(1), output.size(2))
        logit = self.fc(output_reshape)
        logit = logit.view(output.size(0), output.size(1), logit.size(1))
        return logit, state

    def zero_state(self, batch_size):
        return (Variable(torch.cuda.FloatTensor(1, batch_size, 512).zero_()),
                Variable(torch.cuda.FloatTensor(1, batch_size, 512).zero_()))
