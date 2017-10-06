import numpy as np

def read_tri_gram_arpa(file_name, unwritten_prob = 0):
    '''
    input : file name (string)
    return : arpa dict (dict)
    Discription
    read tri-gram arpa format and load data to dict
    words and probs should be devided with \t
    unwritten probs or backoff will be assumed as 'unwritten_prob'(default 0 for log prob)
    '''

    with open(file_name, 'r') as f:
        lines = f.readlines()


    mono_prob = dict()
    mono_backoff = dict()
    bi_prob = dict()
    bi_backoff = dict()
    tri_prob = dict()

    n_mono = int(lines[1].split('=')[1][:-1])
    n_bi = int(lines[2].split('=')[1][:-1])
    n_tri = int(lines[3].split('=')[1][:-1])

    print('n mono : ',n_mono)
    print('n bi : ', n_bi)
    print('n tri : ',n_tri)

    read_stat = 0 # 0 : skip, 1 : mono read, 2 : bi read, 3 : tri read
    for i in range(len(lines)):
        l = lines[i]
        if "-grams" in l:
            if "1" in l:
                read_stat = 1
            elif "2" in l:
                read_stat = 2
            elif "3" in l:
                read_stat = 3
            else:
                raise NotImplementedError("There are more than 3 grams in ARPA file!!")
            continue
        elif '\t' not in l:
            read_stat = 0
            continue

        if read_stat == 0:
            continue
        elif read_stat == 1:
            #read 1-grams
            l_arr = l.split('\t')
            p = float(l_arr[0])
            if len(l_arr) == 2:
                w = l_arr[1][:-1]
                b = unwritten_prob
            elif len(l_arr) == 3:
                w = l_arr[1]
                b = float(l_arr[2][:-1])

            if w in mono_prob.keys():
                raise NotImplementedError("ARPA file has same words sequence with different probs!!")

            mono_prob[w] = p
            mono_backoff[w] = b

        elif read_stat == 2:
            # read 1-grams
            l_arr = l.split('\t')
            p = float(l_arr[0])
            if len(l_arr) == 2:
                w = l_arr[1][:-1]
                b = unwritten_prob
            elif len(l_arr) == 3:
                w = l_arr[1]
                b = float(l_arr[2][:-1])

            if w in bi_prob.keys():
                raise NotImplementedError("ARPA file has same words sequence with different probs!!")

            bi_prob[w] = p
            bi_backoff[w] = b

        elif read_stat == 3:
            l_arr = l.split('\t')
            p = float(l_arr[0])
            w = l_arr[1][:-1]
            tri_prob[w] = p

    if (n_mono != len(mono_prob) or n_mono != len(mono_backoff)
        or n_bi != len(bi_prob) or n_bi != len(bi_backoff)
        or n_tri != len(tri_prob)
        ):
        raise NotImplementedError("Wrong ARPA file (number of components are not matched) !!")

    return [mono_prob, mono_backoff, bi_prob, bi_backoff, tri_prob]


def calc_otf_trigram(s, lm):
    '''
    :param s: string, last 3 words will be used 
    :param lm: list, elem : dict of probs and backoffs
    :lm[0] : mono_prob, lm[1] : mono_backoff, lm[2] : bi_prob, lm[3] : bi_backoff, lm[4] : tri_prob
    :return: float, posterior prob of last 3 word sequence  
    '''

    unk = '<unk>'

    '''
    while():
        if s.find('  ') is -1:
            break
        else:
            s = s.replace('  ', ' ')

    if s[-1] == ' ':
        s = s[:-1]
    '''

    words = s.split(' ')
    n_word = len(words)

    if n_word == 0:
        return -np.inf
    #mono
    elif n_word == 1:
        if words[0] in lm[0].keys():
            return lm[0][words[0]]
        else:
            return lm[0][unk]

    #bi
    elif n_word == 2:
        w = words[0] + ' ' + words[1]
        if w in lm[2].keys():
            return lm[2][w]
        else:
            if words[1] in lm[0].keys():
                new_word_prob = lm[0][words[1]]
            else:
                new_word_prob = lm[0][unk]

            if words[0] in lm[1].keys():
                old_word_back = lm[1][words[0]]
            else:
                old_word_back = lm[1][unk]
            return new_word_prob + old_word_back

    else:
        w = words[-3] + ' ' + words[-2] + ' ' + words[-1]
        if w in lm[4].keys():
            return lm[4][w]
        else:
            old_w = words[-3] + ' ' + words[-2]
            new_bi_w = words[-2] + ' ' + words[-1]

            if new_bi_w in lm[2].keys():
                new_bi_prob = lm[2][new_bi_w]
            else:
                if words[-1] in lm[0].keys():
                    new_word_prob = lm[0][words[-1]]
                else:
                    new_word_prob = lm[0][unk]
                if words[-2] in lm[1].keys():
                    old_word_back = lm[1][words[-2]]
                else:
                    old_word_back = lm[1][unk]
                new_bi_prob = new_word_prob + old_word_back

            if old_w in lm[3].keys():
                return lm[3][old_w] + new_bi_prob
            else:
                if words[-3] in lm[1].keys():
                    first_backoff = lm[1][words[-3]]
                else:
                    first_backoff = lm[1][unk]
                if words[-2] in lm[1].keys():
                    second_backoff = lm[1][words[-2]]
                else:
                    second_backoff = lm[1][unk]
                return new_bi_prob + first_backoff + second_backoff




if __name__ == '__main__':
    arpa_file = 'data/new_WSJ.arpa'
    #mono_p, mono_bf, bi_p, bi_bf, tri_p = read_tri_gram_arpa(arpa_file)
    lm = read_tri_gram_arpa(arpa_file)

    s = 'LAST INCREASED IN'
    p = calc_otf_trigram(s, lm)
    print(s)
    print(p)

    s = 'LAST DINCREASED IN'
    p = calc_otf_trigram(s, lm)
    print(s)
    print(p)

    s = 'DOLLARS FROM THIRTY FIVE DOLLARS <EOS>'
    p = calc_otf_trigram(s, lm)
    print(s)
    print(p)


