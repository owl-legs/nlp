import pickle
import config
import random
import numpy as np
from gensim.models import word2vec, KeyedVectors

class DataReader:

    def __init__(self):

        self.currentPostFix = -1
        self.counter = 0
        self.maxPostFix = config.MAX_POSTFIX
        self.word_vector_size = config.EMBEDDING_SIZE
        self.embedding_map = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
        self.vocab_size = len(self.embedding_map)

        '''if isTraining:
            self.load_next_training_batch()
        else:
            print("\nloading test data")
            #self.test_x, self.test_y, self.data_size = pickle.load(open('model/embedded_test_vectors.txt', "rb"))'''
    def refresh(self):
        self.currentPostFix = -1
        self.counter = 0
    def __generate_batch_ordering__(self):
        self.index = random.sample(range(self.data_size), self.data_size)
    def __load_next_training_batch__(self):

        self.currentPostFix = (self.currentPostFix + 1) % self.maxPostFix
        print(f'''\n loading training data, postfix {self.currentPostFix}''')
        self.train_x, self.train_y, self.data_size = pickle.load( \
            open(f'''data/embedded_train_data_{self.currentPostFix}''', "rb"))
    def get_next_batch(self):
        self.__load_next_training_batch__()
        self.__load_next_training_batch__()
        batchX = [self.train_x[i] for i in self.index]
        batchY = [self.train_y[i] for i in self.index]
        return np.array(batchX), np.array(batchY)












