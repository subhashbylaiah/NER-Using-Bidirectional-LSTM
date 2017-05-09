import numpy as np
import pickle as pkl
import sys
from gensim.models.keyedvectors import KeyedVectors
from random import random
import gc

class RandomWordEmbedding:
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.words = {}
        self.wordembeddingvec = []

    def __getitem__(self, word):
        ind = self.words.get(word, -1)
        if ind == -1:
            new_vec = np.array([random() for i in range(self.dimensions)])
            self.words[word] = len(self.words)
            self.wordembeddingvec.append(new_vec)
            return new_vec
        else:
            return self.wordembeddingvec[ind]

def max_sentence_length(file_name):
    curr_sent_len = 0
    max_length = 0
    lengths = []
    for line in open(file_name):
        if line in ['\n', '\r\n']:
            if curr_sent_len > max_length:
                max_length = curr_sent_len
            lengths.append(curr_sent_len)
            curr_sent_len = 0
        else:
            curr_sent_len += 1
    return max_length


def get_POS_OHE(tag):
    one_hot = np.zeros(5)
    if tag == 'NN' or tag == 'NNS':
        one_hot[0] = 1
    elif tag == 'FW':
        one_hot[1] = 1
    elif tag == 'NNP' or tag == 'NNPS':
        one_hot[2] = 1
    elif 'VB' in tag:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def get_chunk_OHE(tag):
    one_hot = np.zeros(5)
    if 'NP' in tag:
        one_hot[0] = 1
    elif 'VP' in tag:
        one_hot[1] = 1
    elif 'PP' in tag:
        one_hot[2] = 1
    elif tag == 'O':
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def is_capital(word):
    if ord('A') <= ord(word[0]) <= ord('Z'):
        return np.array([1])
    else:
        return np.array([0])


def create_features(model, word_dim, input_file, output_embed, output_tag, sentence_length=-1, max_sentences=-1):
    print('processing %s' % input_file)
    random_embedding = RandomWordEmbedding(300)
    word = []
    tag = []
    sentence = []
    sentence_tag = []
    if sentence_length == -1:
        max_sent_length = max_sentence_length(input_file)
    else:
        max_sent_length = sentence_length

    sentence_length = 0
    print("max sentence length is %d" % max_sent_length)

    num_sents = 0

    for line in open(input_file):
        if line in ['\n', '\r\n']:
            for _ in range(max_sent_length - sentence_length):
                tag.append(np.array([0] * 5))
                temp = np.array([0 for _ in range(word_dim + 11)])
                word.append(temp)
            sentence.append(word)
            sentence_tag.append(np.array(tag))
            sentence_length = 0
            word = []
            tag = []

            num_sents += 1
            if (num_sents > max_sentences) and (max_sentences > 0):
                break

        else:
            assert (len(line.split()) == 4)
            sentence_length += 1



            try:
                # temp = model['word_embeddings'][trained_model['wordIndex'][line.split()[0]]]
                temp = model[line.split()[0]]
            except KeyError:
                temp = random_embedding[line.split()[0]]

            # assert len(temp) == word_dim
            temp = np.append(temp, get_POS_OHE(line.split()[1]))
            temp = np.append(temp, get_chunk_OHE(line.split()[2]))
            temp = np.append(temp, is_capital(line.split()[0]))
            word.append(temp)
            t = line.split()[3]
            if t.endswith('PER'):
                tag.append(np.array([1, 0, 0, 0, 0]))
            elif t.endswith('LOC'):
                tag.append(np.array([0, 1, 0, 0, 0]))
            elif t.endswith('ORG'):
                tag.append(np.array([0, 0, 1, 0, 0]))
            elif t.endswith('MISC'):
                tag.append(np.array([0, 0, 0, 1, 0]))
            elif t.endswith('O'):
                tag.append(np.array([0, 0, 0, 0, 1]))
            else:
                print("ERROR")
                sys.exit(0)
    assert (len(sentence) == len(sentence_tag))
    pkl.dump(sentence, open(output_embed, 'wb'))
    pkl.dump(sentence_tag, open(output_tag, 'wb'))


def load_word_embeddings(embeddings_file_path):
    model = KeyedVectors.load_word2vec_format(embeddings_file_path, binary=True)
    return model


if __name__ == '__main__':
    SENTENCE_LENGTH = 30
    WORD_DIMENSIONS = 300

    trained_model = load_word_embeddings('/home/subhash/Courses/AML IN CL/NER/embeddings/GoogleNews-vectors-negative300.bin')
    train_file = 'CoNLL-2003/eng.train.trimmed'
    validation_file = 'CoNLL-2003/eng.testa.trimmed'
    test_file = 'CoNLL-2003/eng.testb.trimmed'

    create_features(trained_model, WORD_DIMENSIONS, train_file, 'train_X.pkl', 'train_y.pkl', sentence_length = SENTENCE_LENGTH)
    gc.collect()

    create_features(trained_model, WORD_DIMENSIONS, validation_file, 'val_X.pkl', 'val_y.pkl', sentence_length = SENTENCE_LENGTH)
    gc.collect()

    create_features(trained_model, WORD_DIMENSIONS, test_file, 'test_X.pkl', 'test_y.pkl', sentence_length = SENTENCE_LENGTH)
    gc.collect()
