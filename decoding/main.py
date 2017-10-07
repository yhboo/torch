import pickle
from decoding import *
from arpa_ctrl import read_tri_gram_arpa
import time

def main():
    am_out_file = "test_uni_probs.pickle"
    label_file = "test_eval92.trans"
    arpa_file = "data/WSJ_impkn.arpa"
    lm_weight = 0.6
    word_bonus = 2.5
    beam_width = 128

    print("---------configuration--------")
    print('lm weight : ',lm_weight)
    print('word bonus: ', word_bonus)
    print('beam width: ', beam_width)


    lm = read_tri_gram_arpa(arpa_file)
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
    n_total_word_beam = 0
    n_total_char_beam = 0
    n_total_word_err_beam = 0
    n_total_char_err_beam = 0
    total_beamsearch_time = 0
    total_greedy_time = 0

    #for i in range(1):
    for i in range(n_sentence):
        label = label_list[i][:-1] + ' </s>'
        begin = time.time()
        predict_greedy = ctc_greedy(am_out[i], blank_pos=0)
        end = time.time()
        greedy_time = end-begin
        begin = time.time()
        #predict_beams = ctc_beamsearch(am_out[i], 128, 0)
        predict_beams = ctc_beamsearch_tri_gram(am_out[i], lm, lm_weight, word_bonus, beam_width, 0)
        end = time.time()
        beam_time = end-begin
        # c = 1
        # for k in predict_beams.keys():
        #     print('---------', c, '-th beam-----------')
        #     print(k)
        #     print('probs : ',predict_beams[k])
        #     c+=1
        predict_beam_best = predict_beams.popitem(last = False)[0]
        total_greedy_time += greedy_time
        total_beamsearch_time += beam_time

        #calc word error rate
        this_wer, this_err, this_n = WER(predict_greedy, label)
        this_wer_beam, this_err_beam, this_n_beam = WER(predict_beam_best, label)
        n_total_word += this_n
        n_total_word_err += this_err
        n_total_word_beam += this_n_beam
        n_total_word_err_beam += this_err_beam

        #calc char error rate
        this_cer, this_err, this_n = CER(predict_greedy, label_list)
        this_cer_beam, this_err_beam, this_n_beam = CER(predict_beam_best, label)
        n_total_char+=this_n
        n_total_char_err+=this_err
        n_total_char_beam += this_n_beam
        n_total_char_err_beam += this_err_beam

        #print results
        print(i + 1, '-th sentence')
        print('greedy predict')
        print(predict_greedy)
        print('best beam predict')
        print(predict_beam_best)
        print('label')
        print(label)
        print('greedy wer : ', this_wer, ', cer : ', this_cer, 'time : ', greedy_time)
        print('beam   wer : ', this_wer_beam, ', cer : ', this_cer_beam, 'time : ', beam_time)

    print('-----------final results-----------')
    print('total greedy WER : ', n_total_word_err / n_total_word)
    print('total greedy CER : ', n_total_char_err / n_total_char)
    print('greedy decoding time : ', total_greedy_time)
    print('total beam WER : ', n_total_word_err_beam / n_total_word_beam)
    print('total beam CER : ', n_total_char_err_beam / n_total_char_beam)
    print('beam decoding time : ', total_beamsearch_time)
if __name__ == '__main__':
    main()
