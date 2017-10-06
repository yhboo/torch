import pickle
from decoding import *

def main():
    am_out_file = "test_uni_probs.pickle"
    label_file = "test_eval92.trans"

    with open(am_out_file, 'rb') as f:
        am_out = pickle.load(f)

    with open(label_file, 'r') as f:
        label_list = f.readlines()


    n_sentence = len(label_list)
    print('1st am out shape : ',am_out[0].shape)
    print('n_sentence : ',n_sentence)

    n_total_word = 0
    n_total_char = 0
    n_total_word_err = 0
    n_total_char_err = 0

    for i in range(n_sentence):
    #for i in range(1):
        print(i+1,'-th sentence')
        label = label_list[i][:-1] + ' ^'
        predict_greedy = ctc_greedy(am_out[i], blank_pos=0)
        predict_beams = ctc_beamsearch(am_out[i], 128, 0)
        predict_beam_best = predict_beams.popitem(last = False)[0]
        print('greedy predict')
        print(predict_greedy)
        print('best beam predict')
        print(predict_beam_best)
        print('label')
        print(label)

        #this_wer, this_err, this_n = WER(predict_greedy, label_list[i])
        this_wer, this_err, this_n = WER(predict_beam_best, label)
        print('this WER : ', this_wer)
        n_total_word +=this_n
        n_total_word_err +=this_err
        #this_char, this_err, this_n = CER(predict_greedy, label_list[i])
        this_char, this_err, this_n = CER(predict_beam_best, label)
        n_total_char+=this_n
        n_total_char_err+=this_err

    print('total WER : ', n_total_word_err / n_total_word)
    print('total CER : ', n_total_char_err / n_total_char)
if __name__ == '__main__':
    main()
