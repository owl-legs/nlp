import pickle
import config
from gensim.models import KeyedVectors
import tensorflow as tf


model = tf.keras.models.load_model('sherlock.h5')