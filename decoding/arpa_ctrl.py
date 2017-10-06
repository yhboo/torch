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

    return mono_prob, mono_backoff, bi_prob, bi_backoff, tri_prob






if __name__ == '__main__':
    arpa_file = '../WSJ_impkn.arpa'
    mono_p, mono_bf, bi_p, bi_bf, tri_p = read_tri_gram_arpa(arpa_file)

    print(mono_p['<s>'])
    print(mono_bf['<s>'])
    print(bi_p['MANVILLE </s>'])
    print(bi_bf['MANVILLE </s>'])
    print(tri_p['WITH BANKRUPTCY </s>'])

