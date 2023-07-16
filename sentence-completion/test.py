import pickle
import config
from gensim.models import KeyedVectors
from scipy import spatial
import tensorflow as tf


model = tf.keras.models.load_model('sherlock.h5')
embeddingMap = KeyedVectors.load_word2vec_format(config.WORD2VEC_BIN_PATH, binary=True)
test_x, test_y, test_set_size = pickle.load(open(config.EMBEDDED_TEST_PATH, 'rb'))

def predict_test_answer(vector_list, test_y):
    answer_list = ['a', 'b', 'c', 'd', 'e']
    max_similarity = float('-inf')
    answer_index = 0
    for i in range(len(answer_list)):
        similarity = 1-spatial.distance.cosine(vector_list[i][0], test_y)
        if similarity > max_similarity:
            max_similarity = similarity
            answer_index = i
    return answer_list[answer_index]

def predict_test():

    answers = []
    ydx = 0

    for i in range(0, test_set_size, 5):
        preds = []
        for j in range(5):
            print(f'''predicting question: {i + 1} option: {j + 1}''')
            # shape needs to be (n_samples, timestamps, input_size)
            x = test_x[i+j].reshape((1, 2, config.EMBEDDING_SIZE))
            y = test_y[ydx].reshape((1, config.EMBEDDING_SIZE))
            preds.append(model.predict(x))
        answers += [predict_test_answer(preds, y[0])]
        ydx += 1

    return answers

print(predict_test())




