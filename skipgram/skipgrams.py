import pickle
import config
import collections
import os
class Generator:
    def __init__(self):
        print(os.getcwd())
        self.sequences = pickle.load(open('data/trainingData.txt', 'rb'))
        print(self.sequences)
        self.tokens = self.__generate_tokens__()
        print(self.tokens)

    def __filter__(self):
        pass
    def __generate_tokens__(self):
        with open('model/all_words.txt', 'rb') as file:
            all_words = file.readline()
            all_words = str(all_words)
            all_words = all_words.split(' ')
            all_words = list(set(all_words))
            return {w:i for i,w in enumerate(all_words)}

    def __tokenize_sequences__(self):

        tokenized_sequences = []
        for sequence in self.sequences:
            yield list(map(lambda x: self.tokens[x], sequence))

    def generate_training_data(self):

        vocab_size = len(self.tokens)
        targets, contexts, labels = [], [], []

        i = 1
        n = len(self.sequences)

        for tokenized_sequence in self.__tokenize_sequences__():

            print(f'''{(i/n)*100.00}% complete''')
            i += 1

        return targets, contexts, labels


gen = Generator()
gen.generate_training_data()