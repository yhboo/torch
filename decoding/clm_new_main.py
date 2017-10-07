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


def frame_to_char(am_out, blank_pos = 0):
    """
    this function returns character level probability distribution
    :param am_out: (n_frame, n_label) np array
    :return: (n_char, n_label) np array
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
    blank_flag = True
    predict = ""

    for t in range(T):
        if blank_flag:
            # ab_ + b, ab_ + c case
            if max_labels[t] != blank_pos:
                if (max_labels[t] - idx_char_offset) == idx_eos:
                    predict += ' </s>'
                else:
                    predict += charset[max_labels[t] - idx_char_offset]
                blank_flag = False
            # ab_ + _ case
            else:
                pass
        else:
            # ab+ _ case
            if max_labels[t] == blank_pos:
                blank_flag = True
            # ab+ b case
            elif charset[max_labels[t] - idx_char_offset] == predict[-1]:
                pass
            # ab+c case
            else:
                if (max_labels[t] - idx_char_offset) == idx_eos:
                    predict += ' </s>'
                else:
                    predict += charset[max_labels[t] - idx_char_offset]

    return predict


def main():
    #params
    am_out_file = "test_uni_probs.pickle"
    label_file = "test_eval92.trans"
    arpa_file = "data/WSJ_impkn.arpa"
    clm_weight = 0.6
    char_bonus = 0.4
    beam_width = 128

    #model setting
    cfg = Config()
    model = Model(cfg.tied)
    model.cuda()
    model.load_state_dict(torch.load(cfg.save_path))


    print("---------configuration--------")
    print('clm weight : ',clm_weight)
    print('char bonus: ', char_bonus)
    print('beam width: ', beam_width)

    with open(am_out_file, 'rb') as f:
        am_out = pickle.load(f)

    with open(label_file, 'r') as f:
        label_list = f.readlines()


if __name__ =='__main__':
    main()