import pickle
import config
import collections

class Generator:
    def __init__(self):
        self.sequences = pickle.load(open(config.TRAINING_DATA_PATH, 'rb'))
        self.__generate_tokens__()

    def __generate_tokens__(self):
        word_string = ' '.join(self.sequences)
        tokens = word_string.split(' ')
        self.tokens = collections.Counter(tokens)

    def __tokenize_sequences__(self):
        for sequence in self.sequences:
            yield list(lambda x: self.tokens[x], sequence.split(' '))
    def generate_training_data(self):
        pass


