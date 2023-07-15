import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

import pickle

print("\n loading embedded training data")

x, y, datalen = pickle.load(open('data/embedded_train_full', 'rb'))

print(x.shape)
print(y.shape)

x, y = x[:1000], y[:1000]

print(x.shape)
print(y.shape)

vocab_size = max(y) + 1 #len(pickle.load(open('model/one_hot_dic.txt', 'rb'))) + 1
y = to_categorical(y, num_classes=vocab_size)

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

checkpoint = ModelCheckpoint("sherlock.h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto')

reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)

logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

print("\n compiling model")

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))

print("\n training model")

model.fit(x, y, epochs=150, batch_size=10, callbacks=[checkpoint, reduce, tensorboard_Visualization])
