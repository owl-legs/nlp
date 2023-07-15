import pickle
import config
from gensim.models import KeyedVectors
import tensorflow as tf


model = tf.keras.models.load_model('sherlock.h5')
embeddingMap = KeyedVectors.load_word2vec_format(config.WORD2VEC_BIN_PATH, binary=True)
test_data = pickle.load(open(config.PROCESSED_TEST_DATA_PATH, 'rb'))

