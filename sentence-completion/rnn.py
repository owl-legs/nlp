import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

import pickle

x, y, datalen = pickle.load(open('data/embedded_train_full', 'rb'))
vocab_size = len(pickle.load(open('model/one_hot_dic.txt', 'rb')))

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation='relu'))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()
