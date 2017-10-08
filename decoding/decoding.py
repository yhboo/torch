import numpy as np
from collections import OrderedDict
from arpa_ctrl import calc_otf_trigram

from utils import *

def ctc_greedy(am_out, blank_pos = 0):
    """
    :param am_out: am output before softmax, type : numpy array (n_frame, n_label)
    :param blank_pos: position of ctc_blank, type : int(should be 0 or n_label)
    :return: hypothesis, type : string
    """
    predict = ""
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '.>"
    idx_space = charset.find(' ')  # 26
    idx_eos = charset.find('>')
    T = am_out.shape[0]  # time T
    n_label = am_out.shape[1]  # label dimension (n_charset)
    if blank_pos == 0:
        idx_char_offset = 1
    elif blank_pos == n_label:
        idx_char_offset = 0


    max_labels = np.argmax(am_out, axis = 1)
    blank_flag = True
    for t in range(T):
        if blank_flag:
            #ab_ + b, ab_ + c case
            if max_labels[t] != blank_pos:
                if (max_labels[t] - idx_char_offset) == idx_eos:
                    predict += ' </s>'
                else:
                    predict += charset[max_labels[t]-idx_char_offset]
                blank_flag = False
            #ab_ + _ case
            else:
                pass
        else:
            #ab+ _ case
            if max_labels[t] == blank_pos:
                blank_flag = True
            #ab+ b case
            elif charset[max_labels[t]-idx_char_offset] == predict[-1]:
                pass
            #ab+c case
            else:
                if (max_labels[t] - idx_char_offset) == idx_eos:
                    predict += ' </s>'
                else:
                    predict+= charset[max_labels[t] - idx_char_offset]


    return predict

def ctc_beamsearch_tri_gram(am_out, lm, lm_weight = 0.6, word_bonus = 2, beam_width = 128, blank_pos = 0):
    '''   
    :param am_out: am output before softmax, type : numpy array (n_frame, n_label)
    :param lm: list(5,1), [mono_prob, mono_backoff, bi_prob, bi_backoff, tri_prob]
    :param mono_prob: monogram prob, type : dict
    :param mono_backoff: monogram backoff weight, type : dict
    :param bi_prob: bigram prob, type : dict
    :param bi_backoff: bigram backoff, type : dict
    :param tri_prob: trigram prob, type : dict
    :param beam_width: beamsearch width, type : int 
    :param blank_pos: position of ctc_blank, type : int(should be 0 or n_label) 
    :return: hypothesis with width of beam, type : list of string 
    '''
    am_logsoftmax = logsoftmax(am_out)

    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '.>"  # '\n -> ^
    idx_space = charset.find(' ')  # 26
    idx_eos = charset.find('>')
    if idx_space == -1 or idx_eos == -1:
        print("charset doesn't contain space or eos!")
        raise NotImplementedError

    T = am_out.shape[0]  # time T
    n_label = am_out.shape[1]  # label dimension (n_charset)
    A_prev = OrderedDict()
    pnb_ = -np.inf  # log0
    pb_ = 0.0  # log1
    A_prev[''] = np.asarray([pnb_, pb_], dtype='float32')

    if blank_pos == 0:
        idx_char_offset = 1
    elif blank_pos == n_label:
        idx_char_offset = 0

    for t in range(T):
        #if t == 100:
        #    print("t : 100")
        # print("frame : ",t)
        # print(A_prev)
        A_next = OrderedDict()
        for l in A_prev:
            p_prev = A_prev[l]  # p_prev[0] : pnb_, p_prev[1] = pb_
            p_prev_ = np.logaddexp(p_prev[0], p_prev[1])
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
                    if (c - idx_char_offset) == idx_eos:
                        #l_ = l + ' ' + charset[c - idx_char_offset]
                        l_ = l + ' </s>'
                    else:
                        l_ = l + charset[c - idx_char_offset]  # l_ : new , l : old
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

                    elif c - idx_char_offset == idx_space:  # ab + ' ' case (for WLM)
                        if l_ in A_next.keys():
                            pnb_ = p_prev_ + am_logsoftmax[t,c]
                        else:
                            lm_score = calc_otf_trigram('<s> ' + l, lm)
                            pnb_ = p_prev_ + am_logsoftmax[t, c] + lm_score*lm_weight + word_bonus  # 0 is WLM prior prob. need to be edited
                            #print('|', l, '|')
                            #print('p_prev_ : ', p_prev_)
                            #print('softmax : ', am_logsoftmax[t,c])
                            #print('lm score: ', lm_score)
                            #print('pnb : ', pnb_)

                    else:  # ab + c case
                        pnb_ = p_prev_ + am_logsoftmax[t, c]

                    if l_ in A_next.keys():
                        p_ = A_next[l_]
                        pnb_ = np.logaddexp(pnb_, p_[0])
                        pb_ = np.logaddexp(pb_, p_[1])

                    if len(l_.split(' ')[-1]) > 15:
                        pnb_ += - 10.0
                        pb_ += -10.0
                    A_next[l_] = np.asarray([pnb_, pb_], dtype='float32')

        A_prev = beam_scoring(A_next, beam_width)

    return A_prev



def ctc_beamsearch(am_out, beam_width = 128, blank_pos = 0):
    '''
    :param am_out: am output before softmax, type : numpy array (n_frame, n_label)
    :param beam_width: beamsearch width, type : int
    :param blank_pos: position of ctc_blank, type : int(should be 0 or n_label)
    :return: hypothesis with width of beam, type : list of string
    
    Disctirption
     ctc beam search without prior probs
     algorithm is based on "Awni Y.Hannun(2014)"
    '''

    am_logsoftmax = logsoftmax(am_out)

    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ '.>" # '\n -> ^
    idx_space = charset.find(' ') #26
    idx_eos = charset.find('>')
    if idx_space == -1 or idx_eos == -1:
        print("charset doesn't contain space or eos!")
        raise NotImplementedError


    T = am_out.shape[0] # time T
    n_label = am_out.shape[1] #label dimension (n_charset)
    A_prev = OrderedDict()
    pnb_ = -np.inf #log0
    pb_ = 0.0 #log1
    A_prev[''] = np.asarray([pnb_, pb_], dtype = 'float32')

    if blank_pos == 0:
        idx_char_offset = 1
    elif blank_pos == n_label:
        idx_char_offset = 0


    for t in range(T):
        #if t == 100:
        #    print("t : 100")
        #print("frame : ",t)
        #print(A_prev)
        A_next = OrderedDict()
        for l in A_prev:
            p_prev = A_prev[l] #p_prev[0] : pnb_, p_prev[1] = pb_
            p_prev_ = np.logaddexp(p_prev[0], p_prev[1])
            #keep last label
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
                    if (c - idx_char_offset) == idx_eos:
                        l_ = l + ' </s>'
                    else:
                        l_ = l + charset[c - idx_char_offset] # l_ : new , l : old
                    pb_ = -np.inf

                    if charset[c - idx_char_offset] == last_label: #ab + b or ab_ + b case
                        #for ab+b case
                        pnb_ = p_prev[0] + am_logsoftmax[t, c]
                        pb_ = -np.inf
                        if l in A_next.keys():
                            p_ = A_next[l]
                            pb_ = p_[1]
                            pnb_ = np.logaddexp(pnb_, p_[0])
                        A_next[l] = np.asarray([pnb_, pb_], dtype='float32')

                        #for ab_ + b case
                        pnb_ = p_prev[1] + am_logsoftmax[t, c]
                        pb_ = -np.inf

                    elif c == idx_space: #ab + ' ' case (for WLM)
                        pnb_ = p_prev_ + am_logsoftmax[t, c] + 0 # 0 is WLM prior prob. need to be edited

                    else: #ab + c case
                        pnb_ = p_prev_ + am_logsoftmax[t, c]

                    if l_ in A_next.keys():
                        p_ = A_next[l_]
                        pnb_ = np.logaddexp(pnb_, p_[0])
                        pb_ = np.logaddexp(pb_, p_[1])


                    A_next[l_] = np.asarray([pnb_, pb_], dtype='float32')


        A_prev = beam_scoring(A_next, beam_width)

    return A_prev


def beam_scoring(A, beam_width, normalize = False):
    """
    :param A : OrderedDict, (key : string, value : list with 2 elem [pnb, pb]) 
    :param beam_width : int, the number of remaining beams.
    :return: OrderedDict, length : beam_width
    """
    d_beam = OrderedDict()

    d_full = OrderedDict(sorted(A.items(), key=lambda t: (np.logaddexp(t[1][0],t[1][1])), reverse=True))

    counter = 0
    for k in d_full.keys():
        #for debug
        if counter == -1:
            print('key : ', k, 'value : ', d_full[k])
            print('full dict length : ',len(d_full))
        d_beam[k] = d_full[k]
        counter += 1
        if counter == beam_width:
            break

    if normalize:
        max_item = list(d_beam.items())[0]
        max_p = np.logaddexp()

    return d_beam





if __name__ == '__main__':
    ctc_beamsearch()