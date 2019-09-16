# -*- coding: utf-8 -*-

import pandas as pd
from config import config
from collections import Counter
import json
import os
import numpy as np


def make_mini_data(raw_data_path, mini_data_path):
    data = pd.read_csv(raw_data_path)
    # print(data.columns)
    # print(len(data.values.tolist()), data.values.tolist()[0])

    with open(mini_data_path, 'w', encoding='utf-8') as f:
        f.write('comment_text'+'\t'+"|".join(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])+'\n')
        for line in data.values.tolist():
            # print(line)
            f.write("{}\t{}\n".format(line[1].replace('\n', ''), '|'.join([str(_) for _ in line[2:]])))


def preprocess_data(raw_data_path):
    all_words = []
    X = []
    y = []
    with open(raw_data_path) as f:
        Head = f.readline()
        for line in f:
            lis = line.strip().split('\t')
            words = lis[0].lower().split()
            X.append(words)
            all_words.extend(words)
            if "1" not in lis[1]:
                y.append([int(_) for _ in lis[1].split('|')]+[1])
            else:
                y.append([int(_) for _ in lis[1].split('|')] + [0])
    return X, y, all_words


def generator_vocab(all_words, save_path):
    word_to_idx = {}
    idx_to_word = {}
    vocab = [item[0] for item in Counter(all_words).most_common(config["vocab_size"]-2)]
    for idx, word in enumerate(vocab):
        word_to_idx[word] = idx+2
        idx_to_word[idx+2] = word

    # add "UNK", "<PAD>"
    word_to_idx["UNK"] = 0
    idx_to_word[0] = "UNK"

    word_to_idx["<PAD>"] = 1
    idx_to_word[1] = "<PAD>"

    with open(os.path.join(save_path, 'word_to_idx.json'), 'w') as f:
        json.dump(word_to_idx, f)

    with open(os.path.join(save_path, 'idx_to_word.json'), 'w') as f:
        json.dump(idx_to_word, f)

    return word_to_idx, idx_to_word


def padding(X, word_to_idx):
    X_digit = []
    for line in X:
        X_digit.append([word_to_idx.get(word, 0) for word in line][:config["seq_lenght"]] +
                       [1]*(config["seq_lenght"]-len(line)))
    return X_digit


def generate_batchs(X_digit, y, shuffle=True):
    data = np.array(list(zip(X_digit, y)))
    data_size = len(data)
    num_batches_per_epoch = int(data_size/config["batch_size"])+1
    for epoch in config["num_epoch"]:
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num*config["batch_size"]
            end_index = min((batch_num+1)*config["batch_size"], data_size)
            yield shuffle_data[start_index: end_index]


if __name__ == '__main__':
    X, y, all_words = preprocess_data('./mini_data/train.txt')
    generator_vocab(all_words)


