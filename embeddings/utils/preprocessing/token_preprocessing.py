from embeddings.utils.document_config import DocumentConfig
from embeddings.utils.vocab import CorpusVocab


def remove_stopwords(document_tokens: list[str], document_config: DocumentConfig) -> list[str]:
    return [
        token for token in document_tokens if token not in document_config.stopwords
    ]


def remove_unknown_words(document_tokens: list[str], corpus_vocab: CorpusVocab) -> list[str]:
    return [corpus_vocab.vocab.get(token) or corpus_vocab.unknown_word_identifier for token in document_tokens]



