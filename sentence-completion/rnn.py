import config
import pickle
import data_reader
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy, cosine_similarity
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


class Model:
    def __init__(self, embeddingType='word2vec'):
        self.vocab_size = max(pickle.load(open(config.ONE_HOT_MAP_PATH, "rb")).values())
        self.model = self.__build_model__()
        self.optimizer = self.__build_optimizer__()
        self.dr = data_reader.DataReader()

        self.n_inputs = config.EMBEDDING_SIZE

    def __build_model__(self):
        model = Sequential()
        #model.add(Embedding(self.vocab_size, 10, input_length=1))
        model.add(LSTM(1000, return_sequences=True))
        model.add(LSTM(1000))
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(config.EMBEDDING_SIZE))
        return model

    def __build_optimizer__(self):
        return Adam(learning_rate=0.001)

    def __step__(self, X, y):

        with tf.GradientTape() as tape:
            pred = self.model(X)
            loss = cosine_similarity(y.astype('float32'), pred, axis=1)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        print(grads)

    def save(self):
        self.model.save('sherlock.h5')

    def train(self):

        for epoch in range(config.EPOCHS):

            self.dr.refresh()

            for batch in range(config.MAX_POSTFIX):

                batchX, batchY = self.dr.get_next_batch()

                #shape needs to be (n_samples, timestamps, input_size)

                batchX = batchX.reshape((batchX.shape[0], batchX.shape[1], self.n_inputs))
                batchY = batchY.reshape((batchY.shape[0], self.n_inputs))

                self.__step__(batchX, batchY)

        self.model.compile(optimizer=self.optimizer, loss=cosine_similarity, metrics=['accuracy'])


mod = Model()
mod.train()
mod.save()