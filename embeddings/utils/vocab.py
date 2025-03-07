from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np

from embeddings.utils.constants import UNKNOWN_WORD
from embeddings.utils.document_config import DocumentConfig
from embeddings.utils.vocab_config import CorpusVocabConfig
from embeddings.utils.preprocessing.string_preprocessing import lower_text, remove_punctuation, tokenize_document
from embeddings.utils.preprocessing.token_preprocessing import remove_stopwords

@dataclass
class CorpusVocab:
    unknown_word_identifier: str
    vocab: dict = field(default_factory=dict)

    @classmethod
    def create(cls,
               documents: list[str],
               document_config: DocumentConfig,
               corpus_vocab_config: CorpusVocabConfig
               ):

        _tokens = {}

        for document in documents:

            if document_config.lower_text:
                document = lower_text(document=document)
            if document_config.remove_punctuation:
                document = remove_punctuation(document=document)

            document_tokens = tokenize_document(document=document)

            if document_config.remove_stopwords:
                document_tokens = remove_stopwords(document_tokens=document_tokens,
                                                   stopwords=document_config.stopwords)

            document_token_counts = Counter(document_tokens)

            for token in document_tokens:
                if token in _tokens:
                    _tokens[token]['frequency'] += document_token_counts[token]
                else:
                    _tokens[token] = {
                        'frequency': document_token_counts[token]
                    }

        _tokens = cls.__n_most_frequent_tokens__(_tokens=_tokens, max_tokens=corpus_vocab_config.max_tokens)
        _tokens[UNKNOWN_WORD] = {
            'frequency': 1
        }
        _tokens = cls.__assign_token_indexes__(_tokens=_tokens, randomize_token_index=corpus_vocab_config.randomize_token_index)

        return CorpusVocab(
            unknown_word_identifier=UNKNOWN_WORD,
            vocab=_tokens
        )

    @staticmethod
    def __assign_token_indexes__(_tokens, randomize_token_index):
        tokens = np.array(_tokens.keys())
        if randomize_token_index:
            tokens = np.random.choice(tokens, size=len(tokens), replace=False)
        for i, token in enumerate(tokens):
            _tokens[token].update({'index': i})
        return _tokens


    @staticmethod
    def __n_most_frequent_tokens__(_tokens, max_tokens) -> dict:
        _token_tuples = [(_token, _data) for _token, _data in _tokens.items()]
        _token_tuples.sort(key=lambda x: x[1])
        _token_tuples = _token_tuples[:max_tokens]

        return {_token: _data for _token, _data in _token_tuples}

    @property
    def get_vocabulary_list(self) -> list[str]:
        return list(self.vocab.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_token_index(self, token: str) -> int:
        token_info = self.vocab.get(token) or {}
        return token_info.get('index') or -1

    def get_token_corpus_count(self, token: str) -> int:
        token_info = self.vocab.get(token) or {}
        return token_info.get('frequency') or 0

    def as_dict(self) -> dict:
        return asdict(self)






