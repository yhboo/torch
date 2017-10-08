import numpy as np
from torch.autograd import Variable


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


def WER(predict, label):
    """
    :param predict: string, decoding result
    :param label: string, true labels
    :return: float, int, int, (wer, n_wrong, n_full)
    """

    pred_list = predict.split(' ')
    label_list = label.split(' ')
    n_word = len(label_list)

    d = np.zeros((len(label_list), len(pred_list)), dtype = 'int')
    sub_in_del = np.zeros(3, dtype = 'int')
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            if label_list[i-1] == pred_list[j-1] :
                d[i,j] = d[i-1,j-1]
            else:
                sub_in_del[0] = d[i-1,j-1] + 1
                sub_in_del[1] = d[i,j-1] + 1
                sub_in_del[2] = d[i-1,j] + 1
                d[i,j] = np.min(sub_in_del)

    n_err = d[-1,-1]
    print('this n_err : ', n_err)
    print('this n_word : ', n_word)
    return n_err/n_word, n_err, n_word

def CER(predict, label):
    """
        :param predict: string, decoding result
        :param label: string, true labels
        :return: float, int, int, (wer, n_wrong, n_full)
    """
    d = np.zeros((len(label), len(predict)), dtype = 'int')
    sub_in_del = np.zeros(3, dtype = 'int')
    n_char = len(label)
    for i in range(1, d.shape[0]):
        for j in range(1, d.shape[1]):
            if label[i-1] == predict[j-1]:
                d[i,j] = d[i-1, j-1]
            else:
                sub_in_del[0] = d[i - 1, j - 1] + 1
                sub_in_del[1] = d[i, j - 1] + 1
                sub_in_del[2] = d[i - 1, j] + 1
                d[i,j] = np.min(sub_in_del)
    n_err = d[-1,-1]

    #('this n_err : ', n_err)
    #print('this n_char : ',n_char)

    return n_err / n_char, n_err, n_char

def logsoftmax(x):
    e_x = np.exp(x - np.max(x, axis = 1)[:,np.newaxis])
    out = e_x / np.sum(e_x, axis=1)[:,np.newaxis]
    return np.log(out)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
    out = e_x / np.sum(e_x, axis=1)[:, np.newaxis]
    return out