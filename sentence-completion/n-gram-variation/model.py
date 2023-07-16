import n_gram_config
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy, cosine_similarity
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

class Model:
    def __init__(self, type):
        function_map = {'bidrectional':self.__bidirectionalrnn__(),\
                        'trigram':self.__trigram__()}
        self.model = function_map[type]
    def __bidirectionalrnn__(self):
        model = Sequential()
        return model
    def __trigram__(self):
        model = Sequential()
        return model

