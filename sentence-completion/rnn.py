import config
import data_reader
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


class Model:
    def __init__(self):
        self.vocab_size = 101485
        self.model = self.__build_model__()
        self.optimzer = self.__build_optimizer__()
    def __build_model__(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 10, input_length=1))
        model.add(LSTM(1000, return_sequences=True))
        model.add(LSTM(1000))
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(self.vocab_size, activation="softmax"))
        return model

    def __build_optimizer__(self):
        return Adam(learning_rate=0.001)
    def __step__(self, X, y):

        with tf.GradientTape() as tape:
            pred = self.model(X)
            loss = categorical_crossentropy(pred, y)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimzer.apply_gradients(zip(grads, self.model.trainable_variables))

        print(grads)
    def train(self):

        for epoch in range(config.EPOCHS):

            self.dr = data_reader.DataReader()

            for batch in range(config.MAX_POSTFIX):

                batchX, batchY = self.dr.get_next_batch()
                batchY = to_categorical(batchY, num_classes=self.vocab_size)

                self.__step__(batchX, batchY)


mod = Model()
mod.train()