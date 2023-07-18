import pickle
class Embedding:
    def __init__(self):
        self.padded_sentences = pickle.load(open('padded_training_sentences', 'rb'))
        self.all_words = self.__process_and_load_corpus__()
    def __process_and_load_corpus__(self):
        pass