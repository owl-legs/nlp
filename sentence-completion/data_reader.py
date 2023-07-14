import pickle
class DataReader:

    def __init__(self, isTraining):

        self.currentPostFix = -1
        self.maxPostFix = 5
        self.word_vector_size = 300

        if isTraining:
            self.load_next_training_batch()
        else:
            print("\nloading test data")
            self.test_x, self.test_y, self.data_size = pickle.load(open('model/embedded_test_vectors.txt', "rb"))

    def load_next_training_batch(self):
        self.currentPostFix = (self.currentPostFix + 1) % self.maxPostFix
        print(f'''\n loading training data, postfix {self.currentPostFix}''')
        self.train_x, self.train_y, self.data_size = pickle.load(
            open(f'''data/embedded_train_data_{self.currentPostFix}''', "rb"))








