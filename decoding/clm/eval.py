import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import torch.backends.cudnn

from data.parser import parse_wsj
from data.dataset import WSJDataset
from model import Model


def repackage_state(s):
    return Variable(s[0].data), Variable(s[1].data)


def train(model, optimizer, dataset, clip_norm=0.25):
    model.train()
    loss_sum = 0
    state = model.zero_state(dataset.batch_size)
    # state = model.module.zero_state(dataset.batch_size)
    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print('{} / {}'.format(idx, len(dataset)))
        data, target = dataset[idx]
        data, target = Variable(data), Variable(target)
        state = repackage_state(state)
        output, state = model(data, state)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output.view(-1, 30), target.view(-1))
        loss_sum += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(model.parameters(), clip_norm)
        optimizer.step()
    return loss_sum / len(dataset)


def evaluate(model, dataset):
    model.eval()
    loss_sum = 0
    state = model.zero_state(dataset.batch_size)
    for idx in range(len(dataset)):
        data, target = dataset[idx]
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        state = repackage_state(state)
        output, state = model(data, state)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output.view(-1, 30), target.view(-1))
        loss_sum += loss.data[0]
    return loss_sum / len(dataset)


def main(cfg):
    torch.manual_seed(cfg.torch_seed)

    """Prepare data"""
    train_data, valid_data = parse_wsj(cfg.wsj_path)
    train_dataset = WSJDataset(train_data, cfg.batch_size, cfg.sequence_length)
    valid_dataset = WSJDataset(valid_data, cfg.eval_batch_size, cfg.sequence_length)



    """Set model"""
    model = Model(cfg.tied)
    model.cuda()
    model.eval()

    data, target = valid_dataset[0]
    state = model.zero_state(valid_dataset.batch_size)
    state = repackage_state(state)

    print('data type : ', type(data))
    print('data size : ', data.size())
    print('state type : ', type(state[0]))
    print('state size : ', state[0].size())
    exit()

    """eval"""
    start_time = time.time()
    model.load_state_dict(torch.load(cfg.save_path))

    train_loss = evaluate(model, train_dataset)
    valid_loss = evaluate(model, valid_dataset)
    print('...... Train BPC', train_loss / np.log(2))

    print('...... Valid BPC', valid_loss / np.log(2))

    end_time = time.time()
    print('...... Time:', end_time - start_time)


if __name__ == '__main__':
    from config import Config
    config = Config()
    main(config)