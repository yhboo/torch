import os
import collections
import numpy as np
from statistics import character_set


def _file_to_char_ids(file_path, char_set):
    with open(file_path, 'r', encoding='utf-8') as f:
        character_list = f.read()
    return [char_set[char] for char in character_list if char in char_set]


def parse_wsj(ptb_path='/home/khshim/data/wsj/'):
    text_path = ptb_path + 'wsj_text'
    with open(text_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    print('Number of sentences:', len(train_lines))

    valid_lines = []
    for idx, line in enumerate(train_lines):
        if (idx + 1) % 100 == 0:
            valid_lines.append(train_lines.pop(idx))

    print('Number of train sentences:', len(train_lines))
    print('Number of valid sentences:', len(valid_lines))
    train_char = ''.join(train_lines)
    valid_char = ''.join(valid_lines)
    train_data = [character_set[char] for char in train_char]
    valid_data = [character_set[char] for char in valid_char]
    print('Number of train characters:', len(train_data))
    print('Number of valid characters:', len(valid_data))

    return train_data, valid_data

if __name__ == '__main__':
    td, vd = parse_wsj()
