# BI-DIRECTIONAL RECURRENT NEURAL NETWORK FOR NAMED ENTITY RECOGNITION

# # https://danijar.com/variable-sequence-lengths-in-tensorflow/
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py


# Validation set score:
# precision =  [0.44907175112895131, 0.66107165913492572, 0.36440084092501751, 0.002442002442002442, 0.99655275779376495, 0.42173462335867312]
# recall =  [0.92746113989637302, 0.61134328358208956, 0.46846846846846846, 1.0, 0.9259156106391867, 0.65058635394456288]
# f1score =  [0.60513860716700474, 0.63523573200992556, 0.40993299172250686, 0.0048721071863580996, 0.95993647585360575, 0.51174004192872113]
# Test set score:
# precision =  [0.48692468619246859, 0.70717131474103589, 0.4333694474539545, 0.0015503875968992248, 0.99189912113106615, 0.47334574378067357]
# recall =  [0.92729083665338641, 0.57754880694143163, 0.57471264367816088, 0.5, 0.93249991019147183, 0.65935879302215938]
# f1score =  [0.63854595336076814, 0.63582089552238796, 0.49413218035824585, 0.0030911901081916542, 0.96128279667450522, 0.55107871145699927]
# Validation set score:
# precision =  [0.85398896136477676, 0.77081988379599742, 0.6313945339873861, 0.33455433455433453, 0.99655275779376495, 0.70335176226675877]
# recall =  [0.9413716814159292, 0.7476518472135254, 0.71621621621621623, 0.8404907975460123, 0.96758467639247647, 0.81599518941671678]
# f1score =  [0.89555380163114962, 0.75905912269548625, 0.67113594040968338, 0.47860262008733628, 0.98185509921550529, 0.75549781943026817]
# Test set score:
# precision =  [0.83158995815899583, 0.80677290836653381, 0.67984832069339107, 0.32558139534883723, 0.98903324417271687, 0.72262650194618383]
# recall =  [0.93916125221500291, 0.7142857142857143, 0.73736780258519385, 0.65015479876160986, 0.97085521380345086, 0.78796825982653629]
# f1score =  [0.88210818307905681, 0.7577174929840973, 0.70744081172491535, 0.43388429752066116, 0.97985992807117173, 0.75388418079096042]
# Validation set score:
# precision =  [0.9001505268439538, 0.8244028405422853, 0.6720392431674842, 0.60073260073260071, 0.99670263788968827, 0.78127159640635802]
# recall =  [0.94720168954593453, 0.83355091383812008, 0.79321753515301907, 0.88172043010752688, 0.97496609610380092, 0.87078759869054501]
# f1score =  [0.92307692307692313, 0.82895163907822145, 0.72761760242792117, 0.71459694989106748, 0.98571455040669986, 0.82360440761315001]
# Test set score:
# precision =  [0.87081589958159, 0.83864541832669326, 0.72589382448537376, 0.53488372093023251, 0.988536492166603, 0.78067354882382811]
# recall =  [0.94709897610921501, 0.77770935960591137, 0.76878944348823863, 0.69838056680161942, 0.97770219198790631, 0.82096458444563092]
# f1score =  [0.90735694822888291, 0.80702875399361029, 0.74672610755084978, 0.60579455662862158, 0.98308949268478063, 0.80031228313671077]
# Validation set score:
# precision =  [0.89964877069744109, 0.83537766300839256, 0.70918009810791871, 0.6495726495726496, 0.99666516786570747, 0.80010366275051836]
# recall =  [0.96190987124463523, 0.85638649900727992, 0.80766161213088583, 0.90016920473773265, 0.97585941226107054, 0.88733473845564281]
# f1score =  [0.9297381384495722, 0.8457516339869281, 0.75522388059701495, 0.75460992907801427, 0.98615256279543972, 0.84146452257654225]
# Test set score:
# precision =  [0.8655857740585774, 0.85192563081009298, 0.7491874322860238, 0.57519379844961238, 0.98975926633549871, 0.79404298527669659]
# recall =  [0.95775462962962965, 0.80691823899371073, 0.77827799662352282, 0.71346153846153848, 0.97876360338573154, 0.83561887800534285]
# f1score =  [0.90934065934065944, 0.82881136950904388, 0.76345569969638416, 0.63690987124463516, 0.98423072538663225, 0.81430059007289135]
# Validation set score:
# precision =  [0.90817862518815851, 0.87540348612007746, 0.7407147862648914, 0.69841269841269837, 0.99666516786570747, 0.82843814789219072]
# recall =  [0.96276595744680848, 0.87596899224806202, 0.84899598393574294, 0.89655172413793105, 0.97916436591201916, 0.90284315571455465]
# f1score =  [0.93467596178672863, 0.87568614788504995, 0.79116766467065869, 0.78517501715854487, 0.98783726069114086, 0.8640418055680692]
# Test set score:
# precision =  [0.88702928870292885, 0.86387782204515273, 0.75785482123510295, 0.62790697674418605, 0.9903324417271685, 0.81248942291419868]
# recall =  [0.95495495495495497, 0.81210986267166041, 0.81384525887143688, 0.70804195804195802, 0.98133282847406289, 0.84688657611571705]
# f1score =  [0.91973969631236452, 0.83719433719433722, 0.78485273492286112, 0.66557107641741986, 0.98581209585393692, 0.82933149075833468]

import numpy as np
import pickle
import tflearn
import gc


import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, BasicLSTMCell, GRUCell
from tensorflow.contrib import rnn


def load_data(features_file, target_file):
    embeddings = pickle.load(open(features_file, 'rb'))
    tag = pickle.load(open(target_file, 'rb'))
    print("Loading data from files: %s, %s" % (features_file, target_file))
    return np.asarray(embeddings), tag

class BiRNNLSTM:
    def __init__(self):

        SENTENCE_LENGTH = 30
        WORD_DIM = 311
        CLASS_SIZE = 5
        NUM_HIDDEN = 256
        NUM_LAYERS = 1
        LEARNING_RATE = 0.001

        self.input_data = tf.placeholder(tf.float32, [None, SENTENCE_LENGTH, WORD_DIM])
        self.output_data = tf.placeholder(tf.float32, [None, SENTENCE_LENGTH, CLASS_SIZE])

        fw_cell = rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
        fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)

        bw_cell = rnn.LSTMCell(NUM_HIDDEN, state_is_tuple=True)
        bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)

        fw_cell = rnn.MultiRNNCell([fw_cell] * NUM_LAYERS, state_is_tuple=True)
        bw_cell = rnn.MultiRNNCell([bw_cell] * NUM_LAYERS, state_is_tuple=True)

        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        output, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                               tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                               dtype=tf.float32, sequence_length=self.length)

        weight, bias = self.weight_and_bias(2 * NUM_HIDDEN, CLASS_SIZE)
        output = tf.reshape(tf.transpose(tf.stack(output), perm=[1, 0, 2]), [-1, 2 * NUM_HIDDEN])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)

        self.prediction = tf.reshape(prediction, [-1, SENTENCE_LENGTH, CLASS_SIZE])
        self.loss = self.cost()

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)


def f1_score(prediction, target, length):
    CLASS_SIZE = 5
    tp = np.array([0] * (CLASS_SIZE + 1))
    fp = np.array([0] * (CLASS_SIZE + 1))
    fn = np.array([0] * (CLASS_SIZE + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    unnamed_entity = CLASS_SIZE - 1
    for i in range(CLASS_SIZE ):
        if i != unnamed_entity:
            tp[CLASS_SIZE ] += tp[i]
            fp[CLASS_SIZE ] += fp[i]
            fn[CLASS_SIZE ] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(CLASS_SIZE + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))

    print("precision = ", precision)
    print("recall = ", recall)
    print("f1score = ", fscore)
    return fscore[CLASS_SIZE]



X_train, y_train = load_data("train_X.pkl",  "train_y.pkl")
X_val, y_val = load_data("val_X.pkl",  "val_y.pkl")
X_test, y_test = load_data("test_X.pkl",  "test_y.pkl")
model = BiRNNLSTM()

epochs = 5
batch_size = 128
best_score = 0

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for e in range(epochs):
        for p in range(0, len(X_train), batch_size):
            sess.run(model.train_op, {model.input_data: X_train[p:p + batch_size],
                                model.output_data: y_train[p:p + batch_size]})

        pred, length = sess.run([model.prediction, model.length], {model.input_data: X_val,
                                                                   model.output_data: y_val})
        print('Validation set score for epoch: %d' % e)
        score = f1_score(pred, y_val, length)
        #
        if score > best_score:
            best_score = score
            pred, length = sess.run([model.prediction, model.length], {model.input_data: X_test,
                                                                   model.output_data: y_test})
            print("Best Test set score yet:")
            f1_score(pred, y_test, length)



