# stochastic gradient descent
from dataclasses import dataclass
import collections
from typing import Any, Optional
import numpy as np

from collections import defaultdict
from embeddings.utils.vocab import CorpusVocab

@dataclass
class GLoVE:
    vocab: Optional[CorpusVocab] = None
    co_occurrence_dictionary: Optional[dict] = None

    @classmethod
    def create(
        cls,
        vocab: CorpusVocab,
        tokenized_document_list: list[list[Any]],
        context_window_length: int
    ):
        co_occurrence_dictionary = collections.defaultdict(float)

        for document in tokenized_document_list:
            number_of_document_tokens = len(document)

            for i in range(number_of_document_tokens):
                focal_word = document[i]

                context_window_start_index = max(0, i - context_window_length)
                context_window_end_index = min(number_of_document_tokens, i + context_window_length + 1)

                for j in range(context_window_start_index, context_window_end_index):
                    if i != j:
                        surrounding_word = document[j]
                        co_occurrence_dictionary[(focal_word, surrounding_word)] += 1 / abs(i - j)

        return GLoVE(
            co_occurrence_dictionary=co_occurrence_dictionary,
            vocab=vocab
        )

    def create_embeddings(
            self,
            embedding_dimension: int,
            learning_rate: float,
            number_of_epochs: int,
    ):
        embeddings = {
            word: np.random.rand(embedding_dimension)
            for word in self.vocab.get_vocabulary_list
        }

        for epoch in range(number_of_epochs):
            total_loss = 0
            for (focal_word, context_word), co_occurrence_count in self.co_occurrence_dictionary.items():
                embedding_dot_product = np.dot(embeddings[focal_word], embeddings[context_word])

                difference = embedding_dot_product - np.log(co_occurrence_count)
                total_loss += 0.5 * difference**2

                gradient = difference * embeddings[context_word]
                embeddings[focal_word] -= learning_rate * gradient

            print(f"mean loss for epoch #{1 + epoch} is {np.round(total_loss / len(self.co_occurrence_dictionary), 5)}")

        return embeddings







