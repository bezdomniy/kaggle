import numpy as np
import datetime
import os
import tensorflow as tf
from sklearn.utils import shuffle
from voice import load_data,max_len

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

checkpoints_dir = "./checkpoints"

batch_size = 200
iterations = 20001
train_size = 50121

def getTrainBatch(spectrograms,labels,lengths):
    #sample= np.random.randint(spectrograms.shape[0], size=batch_size)
    sample= np.random.randint(train_size, size=batch_size)

    arr = spectrograms[sample]
    lengths = lengths[sample]
    lab = labels[sample]
    return arr, lengths, lab

def getData(spectrograms,labels,lengths, test=False):
    if test:
        return spectrograms[train_size:], lengths[train_size:], labels[train_size:]
    else:
        return spectrograms[:train_size], lengths[:train_size], labels[:train_size]
            

def test(val_data, val_lengths, val_labels):
    saver = tf.train.import_meta_graph('./checkpoints/trained_model.ckpt-20000.meta')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './checkpoints/trained_model.ckpt-20000')

        graph = tf.get_default_graph()

        input_data = graph.get_tensor_by_name("inputs:0")
        labels = graph.get_tensor_by_name("labels:0")
        input_lengths = graph.get_tensor_by_name("input_lengths:0")

        accuracy = graph.get_tensor_by_name("accuracy:0")
        print("Model restored.")

        accuracies = []

        for i in range(0,val_data.shape[0],batch_size):  
            accuracy_value = sess.run([accuracy],{input_data: val_data[i:i+batch_size], labels: val_labels[i:i+batch_size]})
            print("Acc: ",accuracy_value)
            accuracies.append(accuracy_value)
        print("Test Accuracy = {:.3f}".format(np.mean(accuracies)))


def train(spectrograms,lengths, cats):
    #spectrograms,cats,lengths = load_data()

    input_data, labels, input_lengths, dropout_keep_prob, optimizer, accuracy, loss = \
    define_graph()

    # tensorboard
    train_acc_op = tf.summary.scalar("training_accuracy", accuracy)

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
        batch_data, batch_lengths, batch_labels = getTrainBatch(spectrograms,cats,lengths)

        sess.run(optimizer, {input_data: batch_data, input_lengths: batch_lengths, labels: batch_labels, dropout_keep_prob: 0.5})
        if (i % 50 == 0):
            accuracy_value, train_summ = sess.run(
                [accuracy, train_acc_op],
                {input_data: batch_data, input_lengths: batch_lengths, labels: batch_labels})
            writer.add_summary(train_summ, i)

            print("Iteration: ", i)
            #print("loss", loss_value)
            print("acc", accuracy_value)
            #print("test acc", accuracy_validation)

        if (i % 10000 == 0 and i != 0):
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
            save_path = all_saver.save(sess, checkpoints_dir +
                                    "/trained_model.ckpt",
                                    global_step=i)
            print("Saved model to %s" % save_path)

    sess.close()

def define_graph():
    fully_connected_units = 128    
 
    # Length vector for 0 padded tensor
    def length(data):
        length = tf.reduce_sum(tf.sign(data), 1)
        length = tf.cast(length, tf.int32)
        return length

    inputs = tf.placeholder(tf.float32, [batch_size, max_len, 129], name="inputs")
    labels = tf.placeholder(tf.int8, [batch_size, 30], name="labels")
    input_lengths = tf.placeholder(tf.int32, [batch_size], name="input_lengths")



    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    input_layer = tf.reshape(inputs, [batch_size, max_len, 129, 1])

    output = tf.layers.conv2d(input_layer,16,3,1,"same",activation=tf.nn.relu)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output,32,3,1,"same",activation=tf.nn.relu)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output,64,3,1,"same",activation=tf.nn.relu)
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.layers.conv2d(output,128,3,1,"same",activation=tf.nn.relu)
    output = tf.layers.max_pooling2d(output, 2, 2)
    
    output = tf.reshape(output, [batch_size, 7 * 8 * 128])

    fully_connected = tf.layers.dense(output, fully_connected_units, activation_fn=tf.sigmoid)

    dropout = tf.layers.dropout(inputs=fully_connected, rate=0.5)


    logits = tf.contrib.layers.fully_connected(dropout, 30, activation_fn=None)
    preds = tf.nn.softmax(logits)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")

    # Do gradient clipping over all variables
    _optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*_optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = _optimizer.apply_gradients(zip(gradients, variables))
    
    correct_preds = tf.equal(tf.round(tf.argmax(preds, 1)), tf.round(tf.argmax(labels, 1)))

    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32),name="accuracy")

    return inputs, labels,input_lengths, dropout_keep_prob, optimizer, accuracy, loss



def run():
    spectrograms,cats,lengths = load_data()
    spectrograms,cats,lengths = shuffle(spectrograms,cats,lengths, random_state=0)

    train_data, train_lengths, train_labels = getData(spectrograms,cats,lengths)
    train(train_data, train_lengths, train_labels)

    val_data, val_lengths, val_labels = getData(spectrograms,cats,lengths, test=True)
    test(val_data, val_lengths, val_labels)

run()