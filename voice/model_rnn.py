import numpy as np
import datetime
import os
import tensorflow as tf
from sklearn.utils import shuffle
from voice import load_data,max_len

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#checkpoints_dir = "./checkpoints"
checkpoints_dir = "./gc"

batch_size = 200
iterations = 50001
train_size = 64721

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
            

def validate(val_data, val_lengths, val_labels):
    saver = tf.train.import_meta_graph(checkpoints_dir+'/trained_model.ckpt-50000.meta')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoints_dir+'/trained_model.ckpt-50000')

        graph = tf.get_default_graph()

        input_data = graph.get_tensor_by_name("inputs:0")
        labels = graph.get_tensor_by_name("labels:0")
        input_lengths = graph.get_tensor_by_name("input_lengths:0")

        accuracy = graph.get_tensor_by_name("accuracy:0")
        print("Model restored.")

        accuracies = []

        for i in range(0,val_data.shape[0],batch_size):  
            accuracy_value = sess.run([accuracy],{input_data: val_data[i:i+batch_size], \
                labels: val_labels[i:i+batch_size]})
            print("Acc: ",accuracy_value)
            accuracies.append(accuracy_value)
        print("Test Accuracy = {:.3f}".format(np.mean(accuracies)))

def test(test_data, test_lengths, file_list):
    saver = tf.train.import_meta_graph(checkpoints_dir+'/trained_model.ckpt-30000.meta')
    #test_data = test_data[:203]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoints_dir+'/trained_model.ckpt-30000')

        graph = tf.get_default_graph()

        #print([n.name for n in tf.get_default_graph().as_graph_def().node])

        prediction_list = np.empty([test_data.shape[0]+1, 2],dtype="<U18")
        prediction_list[0,0] = "fname"
        prediction_list[0,1] = "label"
        limited_categories = np.array(["yes", "no", "up", "down", "left", "right", "on",
                        "off", "stop", "go", "silence", "unknown"])

        input_data = graph.get_tensor_by_name("inputs:0")
        input_lengths = graph.get_tensor_by_name("input_lengths:0")

        predictions = graph.get_tensor_by_name("predictions:0")
        print("Model restored.")


        for i in range(0,test_data.shape[0],batch_size):  
            if i+batch_size <= test_data.shape[0]:
                next_predictions = sess.run(predictions,{input_data: test_data[i:i+batch_size]})

                prediction_list[i+1:i+batch_size+1,0] = file_list[i:i+batch_size]
                prediction_list[i+1:i+batch_size+1,1] = limited_categories[np.argmax(next_predictions, axis = 1)]
            else:
                backtrack = 200 - (test_data.shape[0] % 200)

                next_predictions = sess.run(predictions, \
                    {input_data: test_data[i-backtrack:test_data.shape[0]]})

                prediction_list[i-backtrack+1:test_data.shape[0]+1,0] = \
                    file_list[i-backtrack:test_data.shape[0]]
                prediction_list[i-backtrack+1:test_data.shape[0]+1,1] = \
                    limited_categories[np.argmax(next_predictions, axis = 1)]
            print(i,end=" ")
            print(prediction_list[i+1])
        print("Test complete")
        np.save('prediction_list',prediction_list)
    return prediction_list


def train(spectrograms,lengths, cats, lim = False):
    #spectrograms,cats,lengths = load_data()

    input_data, labels, input_lengths, dropout_keep_prob, optimizer, accuracy, predictions, loss = \
        define_graph(limited = lim)

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

    weights = count_vec/np.max(count_vec)

    for i in range(iterations+1):
        batch_data, batch_lengths, batch_labels = getTrainBatch(spectrograms,cats,lengths)

        sess.run(optimizer, {input_data: batch_data, input_lengths: batch_lengths, \
            labels: batch_labels, dropout_keep_prob: 0.5})
        if (i % 50 == 0):
            accuracy_value, preds, train_summ = sess.run(
                [accuracy,predictions, train_acc_op],
                {input_data: batch_data, input_lengths: batch_lengths, labels: batch_labels})
            writer.add_summary(train_summ, i)

            print("Iteration: ", i)
            print(np.argmax(preds,axis=1))
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

def define_graph(limited= False):
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    if limited:
        output_units=12
    else:
        output_units=31
    hidden_units = 128
    fully_connected_units = 0    # 0 for no fully connected layer
    num_layers = 2    # only works with non-bidirectional LSTM

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

    inputs = tf.placeholder(tf.float32, [batch_size, max_len, 129], name="inputs")
    labels = tf.placeholder(tf.int8, [batch_size, output_units], name="labels")
    input_lengths = tf.placeholder(tf.int32, [batch_size], name="input_lengths")


    def lstm_cell_with_dropout():
        cell = tf.contrib.rnn.LSTMCell(hidden_units,forget_bias=0.0, state_is_tuple=True)
        return tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, \
            dtype=tf.float32, output_keep_prob=dropout_keep_prob)

    def lstm_cell_with_dropout_reducing_by_half():
        cells = []
        for i in range(0,num_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(hidden_units/int(pow(2,i)), \
                forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, \
                dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
            cells.append(cell)
        return cells

    def lstm_cell_with_dropout_and_skip_connection():
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
        cell = tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32, \
            output_keep_prob=dropout_keep_prob)
        return tf.contrib.rnn.ResidualWrapper(cell)

    def lstm_cell_with_layernorm_and_dropout():
        return  tf.contrib.rnn.LayerNormBasicLSTMCell(hidden_units,forget_bias=0.0, \
            dropout_keep_prob=dropout_keep_prob)


    def gru_cell_with_dropout():
        cell = tf.contrib.rnn.GRUCell(hidden_units)
        return tf.contrib.rnn.DropoutWrapper(cell, variational_recurrent=True, dtype=tf.float32, \
            output_keep_prob=dropout_keep_prob)

    def bidirectional_lstm_cell_with_dropout():
        #fwcell = tf.contrib.rnn.LSTMCell(hidden_units,forget_bias=0.0, state_is_tuple=True)
        #bwcell = tf.contrib.rnn.LSTMCell(hidden_units,forget_bias=0.0, state_is_tuple=True)

        fwcell = lstm_cell_with_layernorm_and_dropout()
        bwcell = lstm_cell_with_layernorm_and_dropout()

        fwcell = tf.contrib.rnn.DropoutWrapper(fwcell, variational_recurrent=True, \
            dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
        bwcell = tf.contrib.rnn.DropoutWrapper(bwcell, variational_recurrent=True, \
            dtype=tf.float32 , output_keep_prob=dropout_keep_prob)

        return fwcell,bwcell

    def bidirectional_gru_cell_with_dropout():
        fwcell = tf.contrib.rnn.GRUCell(hidden_units)
        bwcell = tf.contrib.rnn.GRUCell(hidden_units)

        fwcell = tf.contrib.rnn.DropoutWrapper(fwcell, variational_recurrent=True, \
            dtype=tf.float32 , output_keep_prob=dropout_keep_prob)
        bwcell = tf.contrib.rnn.DropoutWrapper(bwcell, variational_recurrent=True, \
            dtype=tf.float32 , output_keep_prob=dropout_keep_prob)

        return fwcell,bwcell



    cell = tf.nn.rnn_cell.MultiRNNCell(
        [gru_cell_with_dropout() for _ in range(num_layers)], state_is_tuple=True)
        #[lstm_cell_with_layernorm_and_dropout() for _ in range(num_layers)], state_is_tuple=True)
        #lstm_cell_with_dropout_reducing_by_half(), state_is_tuple=True)
        #[lstm_cell_with_dropout()]+[lstm_cell_with_dropout_and_skip_connection() for _ in range(num_layers-1)], state_is_tuple=True)

    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=input_lengths, \
        initial_state=initial_state, dtype=tf.float32)

    # Get the last non-0 output from RNN
    last_output = last(outputs, input_lengths)

    # Option of a fully connected layer
    if fully_connected_units > 0:
        fully_connected = tf.layers.dense(
            last_output, fully_connected_units, activation=tf.sigmoid)
        fully_connected = tf.contrib.layers.dropout(
            fully_connected, dropout_keep_prob)
    else:
        fully_connected = last_output

    logits = tf.layers.dense(fully_connected, output_units, activation=None)
    preds = tf.nn.softmax(logits, name="predictions")

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),name="loss")

    # Do gradient clipping over all variables
    _optimizer = tf.train.AdamOptimizer()
    gradients, variables = zip(*_optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = _optimizer.apply_gradients(zip(gradients, variables))
    
    correct_preds = tf.equal(tf.round(tf.argmax(preds, 1)), tf.round(tf.argmax(labels, 1)))

    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32),name="accuracy")

    return inputs, labels,input_lengths, dropout_keep_prob, optimizer, accuracy, preds, loss

