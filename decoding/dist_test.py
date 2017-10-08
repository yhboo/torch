import pickle
import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x, axis = 1)[:,np.newaxis])
    out = e_x / np.sum(e_x, axis=1)[:,np.newaxis]
    return out

def main():
    am_out_file = "test_uni_probs.pickle"
    label_file = "test_eval92.trans"

    with open(am_out_file, 'rb') as f:
        am_out = pickle.load(f)

    with open(label_file, 'r') as f:
        label_list = f.readlines()


    n_sentence = len(label_list)

    max_sparse_idx = 0
    min_sparse_idx = 0
    max_n_blank_ratio = 0
    min_n_blank_ratio = 1

    total_frame = 0
    total_blank = 0
    total_u = 0
    for i in range(n_sentence):
        this_am_out = am_out[i]
        this_probs = softmax(this_am_out)
        this_n_frame = this_am_out.shape[0]
        this_n_blank = np.sum(np.argmax(this_am_out, axis = 1) == 0)
        this_u = np.sum(this_probs[:,0] > 0.95)
        this_n_blank_ratio = this_n_blank / this_n_frame

        total_frame+=this_n_frame
        total_blank+=this_n_blank
        total_u +=this_u

        if this_n_blank_ratio > max_n_blank_ratio:
            max_sparse_idx = i
            max_n_blank_ratio = this_n_blank_ratio

        if this_n_blank_ratio < min_n_blank_ratio:
            min_sparse_idx = i
            min_n_blank_ratio = this_n_blank_ratio

        print(i,'-th sentence')
        print('n_frame : ', this_n_frame, 'n_blank : ', this_n_blank)

    print('-------------------------results-----------------')
    print('max sparse sentence idx : ', max_sparse_idx)
    print('min sparse sentence idx : ', min_sparse_idx)
    print('total frame : ', total_frame)
    print('total blank : ', total_blank)
    print('total u : ', total_u)
    max_sparse_seq = np.argmax(am_out[max_sparse_idx], axis=1)
    min_sparse_seq = np.argmax(am_out[min_sparse_idx], axis=1)

    plt.plot(max_sparse_seq)
    #plt.show()
    plt.plot(min_sparse_seq)
    #plt.show()


if __name__ == '__main__':
    main()