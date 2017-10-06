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
    model.cuda(device_id=0)
    # model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(model.parameters(), cfg.initial_lr)
    # optimizer = optim.SGD(model.parameters(), cfg.initial_lr, cfg.momentum, nesterov=True)

    """Train"""
    best_valid_loss = 1000000
    patience = 0
    change = 0
    status = 'keep_train'

    for epoch in range(cfg.max_epoch):
        print('... Epoch', epoch, status)
        start_time = time.time()
        if status == 'end_train':
            time.sleep(1)
            torch.save(model.state_dict(), cfg.save_path)
            break
        elif status == 'change_lr':
            time.sleep(1)
            model.load_state_dict(torch.load(cfg.save_path))
            lr = cfg.initial_lr * np.power(0.1, change)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        elif status == 'save_param':
            torch.save(model.state_dict(), cfg.save_path)
        else:
            pass

        train_loss = train(model, optimizer, train_dataset, cfg.clip_norm)
        valid_loss = evaluate(model, valid_dataset)
        print('...... Train BPC', train_loss / np.log(2))
        print('...... Valid BPC, best BPC', valid_loss / np.log(2), best_valid_loss / np.log(2))

        if valid_loss > best_valid_loss:
            patience += 1
            print('......... Current patience', patience)
            if patience >= cfg.max_patience:
                change += 1
                patience = 0
                print('......... Current lr change', change)
                if change >= cfg.max_change:
                    status = 'end_train'  # (load param, stop training)
                else:
                    status = 'change_lr'  # (load param, change learning rate)
            else:
                status = 'keep_train'  # (keep training)
        else:
            best_valid_loss = valid_loss
            patience = 0
            print('......... Current patience', patience)
            status = 'save_param'  # (save param, keep training)

        end_time = time.time()
        print('...... Time:', end_time - start_time)

    train_loss = evaluate(model, train_dataset)
    print('...... Train BPC', train_loss / np.log(2))
    valid_loss = evaluate(model, valid_dataset)
    print('...... Valid BPC', valid_loss / np.log(2))


if __name__ == '__main__':
    from config import Config
    config = Config()
    main(config)
