import pickle
import config
import numpy as np

tokens = pickle.load(open(config.TOKEN_DIC_OUTPUT, 'rb'))
allText = np.array(list(map(lambda x: x[1:-1], pickle.load(open(config.TOKENIZED_SENTENCE_OUTPUT, 'rb'))))).flatten()

sequences = []
for i in range(1, len(allText)-1):
    sequences.append(allText[i-1:i+1])

X = list(map(lambda x: x[0], sequences))
y = list(map(lambda x: x[1], sequences))

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

vocab_size = len(tokens) + 1
y = to_categorical(y, num_classes=vocab_size)

print("\n creating model")

model = Sequential()
model.add(Embedding(vocab_size=vocab_size, embedding_size=10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(activation='relu'))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()

print("\n training model")
