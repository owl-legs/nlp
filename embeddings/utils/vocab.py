import collections
from dataclasses import dataclass, field

from embeddings.utils.config import DocumentConfig
from embeddings.utils.tokenizer import lower_text, remove_punctuation, tokenize_document


@dataclass
class CorpusVocab:
    tokens: dict = field(default_factory=dict)

    @classmethod
    def create(cls, documents: list[str], document_config: DocumentConfig):

        _tokens = {}
        token_count = 0

        for document in documents:

            if document_config.lower_text:
                document = lower_text(document=document)
            if document_config.remove_punctuation:
                document = remove_punctuation(document=document)

            document_tokens = tokenize_document(document=document)
            document_tokens = set(document_tokens)

            for token in document_tokens:
                if token in _tokens:
                    _tokens[token]['frequency'] += 1
                else:
                    _tokens[token] = {
                        'frequency': 1,
                        'first_observed_index': token_count
                    }
                    token_count += 1

        return CorpusVocab(
            tokens=_tokens
        )

    @property
    def get_vocabulary_list(self) -> list[str]:
        return list(self.tokens.keys())

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def get_token_index(self, token: str) -> int:
        token_info = self.tokens.get(token) or {}
        return token_info.get('first_observed_index') or -1

    def get_token_corpus_count(self, token: str) -> int:
        token_info = self.tokens.get(token) or {}
        return token_info.get('frequency') or 0






