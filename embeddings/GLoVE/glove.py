# stochastic gradient descent
from typing import Any
import numpy as np

from embeddings.utils.vocab import CorpusVocab


class GLoVE:
    probability_matrix: list[list[int]]

    def __init__(self, vocab: CorpusVocab, tokenized_document_list: list[list[Any]], context_length):
        n_words = vocab.vocab_size
        self.probability_matrix = np.zeroes(shape=(n_words, n_words))

        for document in tokenized_document_list:
            for document_index in range(len(document) - 1):
                focal_word = document[document_index]
                focal_word_index = vocab.get_token_index(token=focal_word)

                for context_word in document[document_index + 1: document_index + 1 + context_length]:
                    context_word_index = vocab.get_token_index(token=context_word)

                    self.probability_matrix[focal_word_index][context_word_index] += 1
                    self.probability_matrix[context_word_index][focal_word_index] += 1

        self.frequency = self.probability_matrix.sum(axis=1)

        for i in range(n_words):
            for j in range(n_words):
                self.probability_matrix[i][j] /= self.frequency[i]

        self.probability_matrix = self.probability_matrix.tolist()

    def create_emebddings(self):
        pass






