import pickle
import config
import numpy as np
from functools import reduce

def flatten_reduce_lambda(matrix):
    return list(reduce(lambda x, y: x + y, matrix, []))

tokens = pickle.load(open(config.TOKEN_DIC_OUTPUT, 'rb'))
allText = pickle.load(open(config.EMBEDDED_SENTENCES, 'rb'))
allText = flatten_reduce_lambda([sentence[1:-1] for sentence in allText])

sequences = []
for i in range(1, len(allText)-1):
    words = allText[i-1:i+1]
    sequences.append([words])

X = list(map(lambda x: x[0][0], sequences))
y = list(map(lambda x: x[0][1], sequences))

X = np.array(X)
y = np.array(y)


import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

vocab_size = len(tokens) + 1
y = to_categorical(y, num_classes=vocab_size)

print("\n creating model")

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation='relu'))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

checkpoint = ModelCheckpoint("metamorphasis.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

print("\n compiling model")

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))

print("\n training model")

model.fit(X, y, epochs=150, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])

