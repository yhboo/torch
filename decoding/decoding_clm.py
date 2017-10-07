import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from decoding import logsoftmax, beam_scoring


def repackage_state(s):
    return Variable(s[0].data), Variable(s[1].data)
    #return s[0].data, s[1].data

def char_to_label(s, charset):
    s = s.replace('</s>', charset[-1])
    n_seq = len(s)
    label = np.zeros((n_seq,1), dtype = 'long')
    for i in range(n_seq):
        label[i] = charset.find(s[i])

    #print(label.shape)
    #print(label)
    return label



def calc_clm(model, s, charset):
    """
    :param model: clm.model.Model() 
    :param s: string, input
    :return: next char probs
    """

    data = torch.from_numpy(char_to_label(s, charset))  #data shape : (n_seq, 1)
    data = data.cuda().contiguous()

    data = Variable(data)
    state = model.zero_state(1)
    state = repackage_state(state)

    # print('data type : ', type(data))
    # print('data size : ', data.size())
    # print('state type : ', type(state[0]))
    # print('state size : ', state[0].size())

    output, state = model(data, state)
    #print('output : ', output.size())
    ls = F.log_softmax(output.view(-1, 30))
    #print('clm result in func: ', ls.size())
    result = ls[-1]
    #print('ls[-1] : ', result)
    result = result.data.cpu()
    #print('result.data.cpu() : ', result)
    return result



def ctc_beamsearch_clm(am_out, model, clm_weight = 0.6, char_bonus = 2, beam_width = 128, blank_pos = 0):
    """
    See decoding.ctc_beamsearch_tri_gram 
    """
    am_logsoftmax = logsoftmax(am_out)

    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '.>"
    idx_space = charset.find(' ')  # 26
    idx_eos = charset.find('>')
    if idx_space == -1 or idx_eos == -1:
        print("charset doesn't contain space or eos!")
        raise NotImplementedError
    clm_buffer = dict()

    T = am_out.shape[0]  # time T
    n_label = am_out.shape[1]  # label dimension (n_charset)
    A_prev = OrderedDict()
    pnb_ = -np.inf  # log0
    pb_ = 0.0  # log1
    A_prev[''] = np.asarray([pnb_, pb_], dtype='float32')
    clm_buffer[''] = np.log(np.ones(30, dtype = 'float32')/30)

    if blank_pos == 0:
        idx_char_offset = 1
    elif blank_pos == n_label:
        idx_char_offset = 0

    for t in range(T):
        # if t == 100:
        #    print("t : 100")
        # print("frame : ",t)
        # print(A_prev)
        A_next = OrderedDict()
        for l in A_prev:
            p_prev = A_prev[l]  # p_prev[0] : pnb_, p_prev[1] = pb_
            p_prev_ = np.logaddexp(p_prev[0], p_prev[1])

            #calc next char prior prob
            if l not in clm_buffer.keys():
                clm_buffer[l] = calc_clm(model, l, charset)

            # keep last label
            if len(l) > 0:
                last_label = l[-1]
            else:
                last_label = ""

            for c in range(n_label):
                if c == blank_pos:
                    pb_ = p_prev_ + am_logsoftmax[t, c]
                    pnb_ = -np.inf
                    if l in A_next.keys():
                        p_ = A_next[l]
                        pb_ = np.logaddexp(p_[1], pb_)
                        pnb_ = p_[0]
                    A_next[l] = np.asarray([pnb_, pb_], dtype='float32')
                else:
                    clm_score = clm_buffer[l][c-idx_char_offset]
                    #print('clm buffer')
                    #for kk in clm_buffer.keys():
                    #    print(kk)
                    #print('clm result type : ', type(clm_score))
                    #print(clm_score)
                    if (c - idx_char_offset) == idx_eos:
                        # l_ = l + ' ' + charset[c - idx_char_offset]
                        l_ = l + ' </s>'
                    else:
                        l_ = l + charset[c-idx_char_offset]  # l_ : new , l : old
                    pb_ = -np.inf



                    if charset[c - idx_char_offset] == last_label:  # ab + b or ab_ + b case
                        # for ab+b case
                        pnb_ = p_prev[0] + am_logsoftmax[t, c]
                        pb_ = -np.inf
                        if l in A_next.keys():
                            p_ = A_next[l]
                            pb_ = p_[1]
                            pnb_ = np.logaddexp(pnb_, p_[0])
                        A_next[l] = np.asarray([pnb_, pb_], dtype='float32')

                        # for ab_ + b case
                        pnb_ = p_prev[1] + am_logsoftmax[t, c]
                        pb_ = -np.inf

                    else:
                        pnb_ = p_prev_ + am_logsoftmax[t, c]

                    #apply lm score
                    pnb_ += clm_score*clm_weight + char_bonus

                    if l_ in A_next.keys():
                        p_ = A_next[l_]
                        pnb_ = np.logaddexp(pnb_, p_[0])
                        pb_ = np.logaddexp(pb_, p_[1])

                    if len(l_.split(' ')[-1]) > 15:
                        pnb_ += - 10.0
                        pb_ += -10.0
                    A_next[l_] = np.asarray([pnb_, pb_], dtype='float32')

        A_prev = beam_scoring(A_next, beam_width)

    print('final buffer size : ', len(clm_buffer))
    #for k in clm_buffer.keys():
    #    print(k)
    return A_prev, len(clm_buffer)