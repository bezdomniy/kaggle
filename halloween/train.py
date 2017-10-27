"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint, shuffle
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 100000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 12499)
            labels.append([1, 0])
        else:
            num = randint(12500, 24999)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

def getValBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 12499)
            labels.append([1, 0])
        else:
            num = randint(12500, 24999)
            labels.append([0, 1])
        arr[i] = test_data[num]
    return arr, labels


# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
print("Loaded glove")

training_data = imp.load_data(glove_dict)
test_data = imp.load_data(glove_dict,test=True)
print("Loaded data")

input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(glove_array)

# tensorboard
train_acc_op = tf.summary.scalar("training_accuracy", accuracy)
test_acc_op = tf.summary.scalar("testing_accuracy", accuracy)
#tf.summary.scalar("loss", loss)
#summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

accuracies = []

for i in range(iterations+1):
    batch_data, batch_labels = getTrainBatch()

    val_data, val_labels = getValBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: 0.5})
    if (i % 50 == 0):
        accuracy_value, train_summ = sess.run(
            [accuracy, train_acc_op],
            {input_data: batch_data, labels: batch_labels})
        writer.add_summary(train_summ, i)

        accuracy_validation, valid_summ = sess.run([accuracy, test_acc_op],{input_data: val_data, labels: val_labels})
        #accuracies.append(accuracy_validation)
        writer.add_summary(valid_summ , i)

        print("Iteration: ", i)
        #print("loss", loss_value)
        print("acc", accuracy_value)
        print("test acc", accuracy_validation)

    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)

sess.close()


