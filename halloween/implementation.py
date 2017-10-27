import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
import string
from collections import deque
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 50


def load_data(glove_dict,test=False):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    """Extract data from tarball and store as list of strings"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open('reviews.tar.gz', "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))

    print("READING DATA")
    data = np.zeros((25000, 40), dtype=np.int)

    if not test:
        dir = os.path.dirname(__file__)
    else:
        dir = os.path.join(os.path.dirname(__file__),'test/')
    file_list = glob.glob(os.path.join(dir,
                                       'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir,
                                            'data2/neg/*')))
    print("Parsing %s files" % len(file_list))

  # stopwords from: http://www.lextek.com/manuals/onix/stopwords1.html and https://www.link-assistant.com/seo-stop-words.html
    stopwords = set([  'about', 'mrs', 'few', 'since', 'big', 'part', 'i', 'under', 'area',
                        'above', 'much', 'find', 'small', 'both', 'parted', 'if', 'until', 'areas',
                        'across', 'must', 'finds', 'smaller', 'but', 'parting', 'up', 'around',
                        'after', 'my', 'first', 'smallest', 'by', 'parts', 'in', 'upon', 'as',
                        'again', 'myself', 'for', 'so', 'c', 'per', 'interest', 'us', 'ask',
                        'n', 'some', 'came', 'perhaps', 'interested', 'use', 'asked',
                        'all', 'necessary', 'from', 'somebody', 'can', 'place', 'used', 'asking',
                        'almost', 'need', 'full', 'someone', 'places', 'interests', 'uses', 'asks',
                        'alone', 'needed', 'fully', 'something', 'case', 'point', 'into', 'v', 'at',
                        'along', 'needing', 'further', 'somewhere', 'cases', 'pointed', 'is', 'very', 'away',
                        'already', 'needs', 'furthered', 'state', 'certain', 'pointing', 'it', 'w', 'b',
                        'also', 'furthering', 'states', 'certainly', 'points', 'its', 'want', 'back',
                        'although', 'new', 'furthers', 'still', 'clear', 'possible', 'itself', 'wanted', 'backed',
                        'always', 'new', 'g', 'still', 'clearly', 'present', 'j', 'wanting', 'backing',
                        'among', 'newer', 'gave', 'such', 'come', 'presented', 'just', 'wants', 'backs',
                        'an', 'newest', 'general', 'sure', 'could', 'presenting', 'k', 'was', 'be',
                        'and', 'next', 'generally', 't', 'd', 'presents', 'keep', 'way', 'became',
                        'another', 'get', 'take', 'did', 'keeps', 'ways', 'because',
                        'any', 'gets', 'taken', 'differ', 'problems', 'kind', 'we', 'become',
                        'anybody', 'give', 'than', 'different', 'put', 'knew', 'becomes',
                        'anyone', 'given', 'that', 'differently', 'puts', 'know', 'wells', 'been',
                        'anything', 'gives', 'the', 'do', 'q', 'known', 'went', 'before',
                        'anywhere', 'go', 'their', 'does', 'quite', 'knows', 'were', 'began',
                        'are', 'now', 'going', 'them', 'done', 'r', 'l', 'what', 'behind',
                        'then', 'down', 'rather', 'large', 'when', 'men', 'turned',
                        'number', 'goods', 'there', 'down', 'really', 'largely', 'where', 'might', 'turning',
                        'numbers', 'got', 'therefore', 'downed', 'right', 'last', 'whether', 'more', 'turns',
                        'o', 'these', 'downing', 'right', 'later', 'which', 'most', 
                        'of', 'greater', 'they', 'downs', 'room', 'latest', 'while', 'mostly', 'u',
                        'off', 'thing', 'during', 'rooms', 'least', 'who', 'mr', 'him',
                        'often', 'group', 'things', 'e', 's', 'less', 'whole', 'young', 'himself',
                        'old', 'grouped', 'think', 'each', 'said', 'let', 'whose', 'younger', 'his',
                        'older', 'grouping', 'thinks', 'early', 'same', 'lets', 'why', 'youngest', 'how',
                        'oldest', 'groups', 'this', 'either', 'saw', 'like', 'will', 'your', 'however',
                        'on', 'h', 'those', 'end', 'say', 'likely', 'with', 'yours', 'others',
                        'once', 'had', 'though', 'ended', 'says', 'long', 'within', 'z', 'our',
                        'has', 'thought', 'ending', 'second', 'longer', 'show', 'out',
                        'only', 'have', 'thoughts', 'ends', 'seconds', 'longest', 'work', 'showed', 'over',
                        'open', 'having', 'enough', 'see', 'm', 'worked', 'showing', 'p',
                        'opened', 'he', 'through', 'even', 'seem', 'made', 'working', 'shows', 'being',
                        'opening', 'her', 'thus', 'evenly', 'seemed', 'make', 'works', 'side', 'beings',
                        'opens', 'here', 'to', 'ever', 'seeming', 'making', 'would', 'sides', 'best',
                        'or', 'herself', 'today', 'every', 'seems', 'man', 'x', 'face', 'better',
                        'order', 'together', 'everybody', 'sees', 'many', 'y', 'faces', 'between',
                        'ordered', 'too', 'everyone', 'several', 'may', 'year', 'fact', 'br',
                        'ordering', 'took', 'everything', 'shall', 'me', 'years', 'facts',
                        'orders', 'higher', 'toward', 'everywhere', 'she', 'member', 'yet', 'far',
                        'other', 'turn', 'f', 'should', 'members', 'you', 'felt',

                        '\x97', 'a', 'hence', 'see', 'able', 'her', 'seeing',
                        'about', 'here', 'seem', 'above', 'hereafter', 'seemed',
                        'abroad', 'hereby', 'seeming', 'according', 'herein', 'seems',
                        'accordingly', 'heres', 'seen', 'across', 'hereupon', 'self',
                        'actually', 'hers', 'selves', 'adj', 'herself',
                        'after', 'hes', 'sent', 'afterwards', 'hi',
                        'again', 'him', 'against', 'himself', 
                        'ago', 'his', 'several', 'ahead', 'hither', 'shall',
                        'hopefully', 'all', 'how', 'she',
                        'allow', 'howbeit', 'shed', 'allows', 'however', 'shell',
                        'almost', 'hundred', 'shes', 'alone', 'i', 'should',
                        'along', 'id', 'alongside', 'ie', 'since',
                        'already', 'if', 'also', 'ignored', 'so',
                        'although', 'ill', 'some', 'always', 'im', 'somebody',
                        'am', 'immediate', 'someday', 'amid', 'in', 'somehow',
                        'amidst', 'inasmuch', 'someone', 'among', 'inc', 'something',
                        'amongst', 'inc.', 'sometime', 'an', 'indeed', 'sometimes',
                        'and', 'indicate', 'somewhat', 'another', 'indicated', 'somewhere',
                        'any', 'indicates', 'soon', 'anybody', 'inner', 'sorry',
                        'anyhow', 'inside', 'specified', 'anyone', 'insofar', 'specify',
                        'anything', 'instead', 'specifying', 'anyway', 'into', 'still',
                        'anyways', 'inward', 'sub', 'anywhere', 'is', 'such',
                        'apart', 'sup', 'appear', 'it', 'sure',
                        'itd', 't', 'appropriate', 'itll', 'take',
                        'are', 'its', 'taken', 'its', 'taking',
                        'around', 'itself', 'tell', 'as', 'ive', 'tends',
                        'as', 'j', 'th', 'aside', 'just', 'than',
                        'ask', 'k', 'asking', 'keep', 
                        'associated', 'keeps', 'thanx', 'at', 'kept', 'that',
                        'available', 'know', 'thatll', 'away', 'known', 'thats',
                        'knows', 'thats', 'b', 'l', 'thatve',
                        'back', 'last', 'the', 'backward', 'lately', 'their',
                        'backwards', 'later', 'theirs', 'be', 'latter', 'them',
                        'became', 'latterly', 'themselves', 'because', 'least', 'then',
                        'become', 'less', 'thence', 'becomes', 'lest', 'there',
                        'becoming', 'let', 'thereafter', 'been', 'lets', 'thereby',
                        'before', 'thered', 'beforehand', 'therefore',
                        'begin', 'likely', 'therein', 'behind', 'likewise', 'therell',
                        'being', 'little', 'therere', 'believe', 'look', 'theres',
                        'below', 'looking', 'theres', 'beside', 'looks', 'thereupon',
                        'besides', 'low', 'thereve', 'lower', 'these',
                        'ltd', 'they', 'between', 'm', 'theyd',
                        'beyond', 'made', 'theyll', 'both', 'mainly', 'theyre',
                        'brief', 'make', 'theyve', 'but', 'makes', 'thing',
                        'by', 'many', 'things', 'c', 'may', 'think',
                        'came', 'maybe', 'third', 'can', 'thirty',
                        'me', 'this', 'mean', 'thorough',
                        'meantime', 'thoroughly', 'caption', 'meanwhile', 'those',
                        'cause', 'merely', 'though', 'causes', 'might', 
                        'certain', 'through', 'certainly', 'mine', 'throughout',
                        'changes', 'minus', 'thru', 'clearly', 'miss', 'thus',
                        'cmon', 'more', 'till', 'co', 'moreover', 'to',
                        'co', 'most', 'together', 'com', 'mostly', 'too',
                        'come', 'mr', 'took', 'comes', 'mrs', 'toward',
                        'concerning', 'much', 'towards', 'consequently', 'must', 'tried',
                        'consider', 'tries', 'considering', 'my', 'truly',
                        'contain', 'myself', 'try', 'containing', 'n', 'trying',
                        'contains', 'name', 'ts', 'corresponding', 'namely', 'twice',
                        'could', 'nd', 'near', 'u',
                        'course', 'nearly', 'un', 'cs', 'necessary', 'under',
                        'currently', 'need', 'underneath', 'd', 'undoing',
                        'dare', 'needs', 'unfortunately', 'unless',
                        'definitely', 'unlike', 'described', 'unlikely',
                        'until', 'did', 'unto',
                        'new', 'up', 'different', 'next', 'upon',
                        'directly', 'upwards', 'do', 'ninety', 'us',
                        'does', 'use', 'used',
                        'doing', 'useful', 'done', 'uses',
                        'nonetheless', 'using', 'down', 'usually',
                        'downwards', 'v', 'during', 'value',
                        'e', 'normally', 'various', 'each', 'versus',
                        'edu', 'very', 'eg', 'via',
                        'novel', 'viz', 'eighty', 'now', 'vs',
                        'either', 'w', 'else', 'o', 'want',
                        'elsewhere', 'obviously', 'wants', 'end', 'of', 'was',
                        'ending', 'off', 'enough', 'often', 'way',
                        'entirely', 'oh', 'we', 'especially', 'wed',
                        'et', 'etc', 'old', 
                        'even', 'on', 'ever', 'once', 'went',
                        'evermore', 'were', 'every', 'ones', 'were',
                        'everybody', 'ones', 'everyone', 'only', 'weve',
                        'everything', 'onto', 'what', 'everywhere', 'whatever',
                        'ex', 'or', 'whatll', 'exactly', 'other', 'whats',
                        'example', 'others', 'whatve', 'otherwise', 'when',
                        'f', 'ought', 'whence', 'fairly', 'whenever',
                        'far', 'our', 'where', 'farther', 'ours', 'whereafter',
                        'few', 'ourselves', 'whereas', 'fewer', 'out', 'whereby',
                        'fifth', 'outside', 'wherein', 'first', 'over', 'wheres',
                        'overall', 'whereupon', 'followed', 'own', 'wherever',
                        'following', 'p', 'whether', 'follows', 'particular', 'which',
                        'for', 'particularly', 'whichever', 'forever', 'past', 'while',
                        'former', 'per', 'whilst', 'formerly', 'perhaps', 'whither',
                        'forth', 'placed', 'who', 'forward', 'please', 'whod',
                        'found', 'whoever', 'possible', 'whole',
                        'from', 'presumably', 'wholl', 'further', 'probably', 'whom',
                        'furthermore', 'provided', 'whomever', 'g', 'provides', 'whos',
                        'get', 'q', 'whose', 'gets', 'que', 'why',
                        'getting', 'quite', 'will', 'given', 'qv', 'willing',
                        'gives', 'r', 'wish', 'go', 'rather', 'with',
                        'goes', 'rd', 'within', 'going', 're', 
                        'gone', 'really', 'wonder', 'got', 'reasonably', 
                        'gotten', 'recent', 'would', 'greetings', 'recently', 
                        'h', 'regarding', 'x', 'had', 'regardless', 'y',
                        'regards', 'yes', 'half', 'relatively', 'yet',
                        'happens', 'respectively', 'you', 'right', 'youd',
                        'has', 'round', 'youll', 's', 'your',
                        'have', 'said', 'youre', 'same', 'yours',
                        'having', 'saw', 'yourself', 'he', 'say', 'yourselves',
                        'hed', 'saying', 'youve', 'hell', 'says', 'z',
                        'hello', 'second', 'help', 'secondly'])

    minimal_stopwords = set([   'a', 'down', 'their', 'onto',
                                'an', 'during', 'this', 'onto',
                                'another', 'except', 'and', 'opposite',
                                'any', 'following', 'but', 'out',
                                'certain', 'for', 'or', 'outside',
                                'each', 'from', 'yet', 'over',
                                'every', 'in', 'for', 'past',
                                'her', 'inside', 'nor', 'plus',
                                'his', 'into', 'so', 'round',
                                'its', 'like', ' as', 'since',
                                'its', 'minus', 'aboard', 'since',
                                'my', 'minus', 'about', 'than',
                                'no', 'near', 'above', 'through',
                                'our', 'next', 'across', 'to',
                                'some', 'of', 'after', 'toward',
                                'that', 'off', 'against', 'under',
                                'the', 'on', 'along', 'underneath',
                                'behind', 'upon', 'around', 'unlike',
                                'below', 'with', 'at', 'until',
                                'beneath', 'without', 'before', 'up',
                                'beside', 'but', 'between', 'by',
                                'beyond', 'br', '\x97'])

    for i in range(len(file_list)):
        with open(file_list[i], "r",  encoding="utf-8") as openf:
            s = openf.read()

            # Replace punctuation with space, except for ' character which could be inside a word
            punc = string.punctuation.replace("'", '')
            translator = s.maketrans(punc, ' ' * len(punc))
            no_punct = s.lower().translate(translator)

            # remove ' character
            no_punct = ''.join(c.lower() for c in no_punct if c != "'")
            no_punct = no_punct.split()

            # remove stopwords, words not in glove dictionary, and pad the end with 0s
            no_stop = [w for w in no_punct if w not in stopwords]
            #no_stop = [w for w in no_punct if w not in minimal_stopwords]

            word_buffer = deque(no_stop)
            for j in range(40):
                while word_buffer:
                    next_word = word_buffer.popleft()
                    try:
                        data[i][j] = glove_dict[next_word]
                        break
                    except KeyError:
                        continue
                else:
                    data[i][j] = 0
    return data


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
    bidirectional = True
    hidden_units = 32
    fully_connected_units = 32    # 0 for no fully connected layer
    num_layers = 1    # only works with non-bidirectional LSTM
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


    input_data = tf.placeholder(tf.int32, [batch_size, 40], name="input_data")
    labels = tf.placeholder(tf.int32, [batch_size, 2], name="labels")
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
                #[lstm_cell_with_dropout() for _ in range(num_layers)], state_is_tuple=True)
                [lstm_cell_with_layernorm_and_dropout() for _ in range(num_layers)], state_is_tuple=True)
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

    logits = tf.contrib.layers.fully_connected(fully_connected, 2, activation_fn=None)
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
