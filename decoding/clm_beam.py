import pickle
import numpy as np
from decoding import *
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from clm.model import Model
from clm.config import Config

def repackage_state(s):
    return Variable(s[0].data), Variable(s[1].data)

def char_to_label(s, charset):
    s = s.replace('<EOS>', charset[-1])
    n_seq = len(s)
    label = np.zeros((n_seq,1), dtype= 'int32')
    for i in range(n_seq):
        label[i] = charset.find(s[i])

    return label



def calc_clm(model, s):
    """
    :param model: clm.model.Model() 
    :param s: string, input
    :return: next char probs
    """
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '.>"
    data = char_to_label(s, charset)  #data shape : (n_seq, 1)

    state = model.zero_state(1)
    # data, target = dataset[idx]
    # data, target = Variable(data, volatile=True), Variable(target, volatile=True)
    state = repackage_state(state)

    output, state = model(data, state)

    ls = F.log_softmax(output.view(-1, 30))
    return ls[-1]


def main():
    #params
    am_out_file = "test_uni_probs.pickle"
    label_file = "test_eval92.trans"
    arpa_file = "data/WSJ_impkn.arpa"
    clm_weight = 0.4
    char_bonus = 2.0
    beam_width = 128

    #model setting
    cfg = Config()
    model = Model(cfg.tied)

    print("---------configuration--------")
    print('clm weight : ',clm_weight)
    print('char bonus: ', char_bonus)
    print('beam width: ', beam_width)

    with open(am_out_file, 'rb') as f:
        am_out = pickle.load(f)

    with open(label_file, 'r') as f:
        label_list = f.readlines()

    n_sentence = len(label_list)
    n_total_word = 0
    n_total_char = 0
    n_total_word_err = 0
    n_total_char_err = 0
    n_total_word_beam = 0
    n_total_char_beam = 0
    n_total_word_err_beam = 0
    n_total_char_err_beam = 0
    total_beamsearch_time = 0
    total_greedy_time = 0

