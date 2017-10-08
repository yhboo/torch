import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import pickle
import numpy as np
from decoding import ctc_greedy, CER, WER
from decoding_clm import ctc_beamsearch_clm
import time
import torch


from clm.model import Model
from clm.config import Config
from utils import *



def frame_to_char(am_out, blank_pos = 0):
    """
    this function returns character level probability distribution
    :param am_out: (n_frame, n_label) np array
    :return: (n_char, n_charset) np array (n_charset = n_label - 1)
    """
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '.>"
    idx_eos = charset.find('>')
    T = am_out.shape[0]
    n_label = am_out.shape[1]
    if blank_pos == 0:
        idx_char_offset = 1
    elif blank_pos == n_label:
        idx_char_offset = 0
    else:
        raise NotImplementedError

    max_labels = np.argmax(am_out, axis = 1)
    am_softmax = softmax(am_out)
    seg_char = np.zeros(T, dtype = 'int')

    #avg of continuous argmax chars
    idx_seg = 0
    prev_label = -1
    for t in range(T):
        if max_labels[t] == blank_pos:
            prev_label = -1
        else:
            cur_label = max_labels[t] - idx_char_offset
            if prev_label == cur_label:
                seg_char[t] = idx_seg
            else:
                idx_seg += 1
                seg_char[t] = idx_seg
                prev_label = cur_label

    n_char = idx_seg
    char_probs = np.zeros((n_char,n_label - 1), dtype = 'float32')

    #remove blank
    for i in range(n_char):
        probs = am_softmax[seg_char == i+1]
        p = np.mean(probs, axis = 0)
        char_probs[i] = p[1:]

    # #same with greedy result
    # char_probs_max = np.argmax(char_probs, axis = 1)
    # predict = ""
    # for i in range(n_char):
    #     predict+=charset[char_probs_max[i]]

    return char_probs

def main():
    #params
    am_out_file = "test_uni_probs.pickle"
    label_file = "test_eval92.trans"
    arpa_file = "data/WSJ_impkn.arpa"
    clm_weight = 0.6
    char_bonus = 0.4
    beam_width = 128

    '''
    #model setting
    cfg = Config()
    model = Model(cfg.tied)
    model.cuda()
    model.load_state_dict(torch.load(cfg.save_path))
    '''

    print("---------configuration--------")
    print('clm weight : ',clm_weight)
    print('char bonus: ', char_bonus)
    print('beam width: ', beam_width)

    with open(am_out_file, 'rb') as f:
        am_out = pickle.load(f)

    with open(label_file, 'r') as f:
        label_list = f.readlines()

    n_sentence = len(am_out)
    for i in range(1):
        char_probs = frame_to_char(am_out[i])
        print(np.sum(char_probs, axis = 1))


if __name__ =='__main__':
    main()