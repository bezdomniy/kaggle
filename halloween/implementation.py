import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
import string
from collections import deque
import tensorflow as tf
import csv
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

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
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    label_dict = {'EAP':[0,0,1],'HPL':[0,1,0],'MWS':[1,0,0]}
    row_count = sum(1 for line in open(filename, 'r',  encoding='utf-8'))

    
    with open(filename, 'r',  encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        feature_name = next(reader)
           
        labels= np.zeros(shape=[row_count,3],dtype=np.int8)

        text=[]
        #max_len = 0

        i=0
        for row in reader:
            labels[i] = label_dict[row[-1]]
            words = tokenizer.tokenize(row[-2])

            words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
            if max_len < len(words):
                max_len = len(words)
            text.append(words)

            i+=1

        
        data = np.zeros(shape=[row_count,max_len],dtype=np.int32)

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

    return unison_shuffled_copies(data,labels)

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


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""

    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())
    tf_version = tf.__version__[:3]

    embedding_shape = 50
    bidirectional = False
    hidden_units = 128
    fully_connected_units = 128    # 0 for no fully connected layer
    num_layers = 2    # only works with non-bidirectional LSTM
    vocab_size = len(glove_embeddings_arr)

    # Length vector for 0 padded tensor
    def length(data):
        length = tf.reduce_sum(tf.sign(data), 1)
        length = tf.cast(length, tf.int32)
        return length

    # Return non-0 output tensor from RNN outputs
    def last(output, length):
        layer_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * 40 + (length - 1)
        flat = tf.reshape(output, [-1, layer_size])
        last = tf.gather(flat, index)
        return last


    input_data = tf.placeholder(tf.int32, [batch_size, max_len], name="input_data")
    labels = tf.placeholder(tf.int32, [batch_size, 3], name="labels")
    input_lengths = length(input_data)

    # Keep tensor for faster lookup - not used in final output
    #embeddings = tf.get_variable("embeddings", shape=[400001,embedding_shape], initializer=tf.constant_initializer(np.array(glove_embeddings_arr)), trainable=False)
    #inputs = tf.nn.embedding_lookup(embeddings, input_data)

    inputs = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)

    def lstm_cell_with_dropout():
        cell = tf.contrib.rnn.LSTMCell(hidden_units,forget_bias=0.0, state_is_tuple=True)
        return tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)

    def lstm_cell_with_dropout_reducing_by_half():
        cells = []
        for i in range(0,num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_units/int(pow(2,i)),forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
            cells.append(cell)
        return cells

    def lstm_cell_with_dropout_and_skip_connection():
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
        return tf.contrib.rnn.ResidualWrapper(cell)

    def lstm_cell_with_layernorm_and_dropout():
        return  tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_units,forget_bias=0.0,dropout_keep_prob=dropout_keep_prob)


    def gru_cell_with_dropout():
        cell = tf.contrib.rnn.GRUCell(hidden_units)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

    def bidirectional_lstm_cell_with_dropout():
        #fwcell = tf.contrib.rnn.LSTMCell(hidden_units,forget_bias=0.0, state_is_tuple=True)
        #bwcell = tf.contrib.rnn.LSTMCell(hidden_units,forget_bias=0.0, state_is_tuple=True)

        fwcell = lstm_cell_with_layernorm_and_dropout()
        bwcell = lstm_cell_with_layernorm_and_dropout()

        fwcell = tf.contrib.rnn.DropoutWrapper(fwcell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
        bwcell = tf.contrib.rnn.DropoutWrapper(bwcell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)

        return fwcell,bwcell

    def bidirectional_gru_cell_with_dropout():
        fwcell = tf.contrib.rnn.GRUCell(hidden_units)
        bwcell = tf.contrib.rnn.GRUCell(hidden_units)

        fwcell = tf.contrib.rnn.DropoutWrapper(fwcell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
        bwcell = tf.contrib.rnn.DropoutWrapper(bwcell, variational_recurrent=True, dtype=tf.float32 , output_keep_prob=dropout_keep_prob)

        return fwcell,bwcell


    if not bidirectional:
        if tf_version == '1.3' or tf_version == '1.2':
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell_with_dropout() for _ in range(num_layers)], state_is_tuple=True)
                #[lstm_cell_with_layernorm_and_dropout() for _ in range(num_layers)], state_is_tuple=True)
                #lstm_cell_with_dropout_reducing_by_half(), state_is_tuple=True)
                #[lstm_cell_with_dropout()]+[lstm_cell_with_dropout_and_skip_connection() for _ in range(num_layers-1)], state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.MultiRNNCell(
                #[lstm_cell_with_dropout() for _ in range(num_layers)], state_is_tuple=True)
                [lstm_cell_with_layernorm_and_dropout() for _ in range(num_layers)], state_is_tuple=True)
        
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs,sequence_length=input_lengths, initial_state=initial_state, dtype=tf.float32)
    else:
        # trying bidirectional lstm
        #fwcell, bwcell = bidirectional_lstm_cell_with_dropout()  
        fwcell, bwcell = bidirectional_gru_cell_with_dropout()      

        initial_state_fw = fwcell.zero_state(batch_size, tf.float32)
        initial_state_bw = bwcell.zero_state(batch_size, tf.float32)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            fwcell, bwcell, inputs, sequence_length=input_lengths, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype=tf.float32)

        outputs = tf.concat(outputs, 2)

    # Get the last non-0 output from RNN
    last_output = last(outputs, input_lengths)

    # Option of a fully connected layer
    if fully_connected_units > 0:
        fully_connected = tf.contrib.layers.fully_connected(
            last_output, fully_connected_units, activation_fn=tf.sigmoid)
        fully_connected = tf.contrib.layers.dropout(
            fully_connected, dropout_keep_prob)
    else:
        fully_connected = last_output

    logits = tf.contrib.layers.fully_connected(fully_connected, 3, activation_fn=None)
    preds = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")

    # Do gradient clipping over all variables
    _optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*_optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = _optimizer.apply_gradients(zip(gradients, variables))
    
    correct_preds = tf.equal(tf.round(tf.argmax(preds, 1)), tf.round(tf.argmax(labels, 1)))

    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32),name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
