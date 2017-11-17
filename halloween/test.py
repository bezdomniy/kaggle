import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
import string
from collections import deque
import tensorflow as tf
import csv
import pandas as pd
import spacy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 50

max_len = 0
filename = 'train.csv'


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_data(glove_dict):
    global max_len

    nlp = spacy.load('en')

    label_dict = {'EAP': [0, 0, 1], 'HPL': [0, 1, 0], 'MWS': [1, 0, 0]}
    row_count = sum(1 for line in open(filename, 'r',  encoding='utf-8'))

    with open(filename, 'r',  encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        feature_name = next(reader)

        labels = np.zeros(shape=[row_count, 3], dtype=np.int8)

        text = []
        #max_len = 0

        i = 0
        for row in reader:
            labels[i] = label_dict[row[-1]]
            words = nlp(row[-2])

            words = [word.lemma_ for word in words if not (word.is_stop or word.is_punct)]

            if i < 3:
                print(words)

            if max_len < len(words):
                max_len = len(words)
            text.append(words)

            i += 1



        data = np.zeros(shape=[row_count, max_len], dtype=np.int32)

        for i in range(len(text)):
            word_buffer = deque(text[i])

            for j in range(len(word_buffer)):
                while word_buffer:
                    next_word = word_buffer.popleft()
                    try:
                        data[i][j] = glove_dict[next_word]
                        break
                    except KeyError:
                        continue
                else:
                    data[i][j] = 0

    return unison_shuffled_copies(data, labels)


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """

    # if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    data = open("glove.6B.50d.txt", 'r', encoding="utf-8").read().split()

    vocab_size = 400001
    embeddings = np.zeros((vocab_size, 50), dtype=np.float32)

    word_index_dict = {'UNK': 0}

    word_counter = 1
    vector_place = 50

    for i in range(len(data)):
        if vector_place == 50:
            word_index_dict.update({data[i]: word_counter})
            word_counter += 1
            vector_place = 0
        else:
            embeddings[word_counter - 1][vector_place] = float(data[i])
            vector_place += 1
    return embeddings, word_index_dict


embed, word_dict = load_glove_embeddings()
data, labels = load_data(word_dict)

for i in range(5):
    print(data[i])
