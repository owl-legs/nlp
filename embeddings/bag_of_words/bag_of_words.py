from collections import Counter
from typing import Optional

import numpy as np

from embeddings.utils.vocab_config import CorpusVocabConfig
from embeddings.utils.vocab import CorpusVocab
from embeddings.utils.preprocessing.string_preprocessing import lower_text, remove_punctuation, tokenize_document
from embeddings.utils.preprocessing.token_preprocessing import remove_stopwords
from embeddings.utils.document_config import DocumentConfig


def bag_of_words_embedding(
        tokens: list[str],
        corpus_vocab: CorpusVocab
) -> list[int]:
    token_counts = Counter(tokens)
    embedding = np.zeros(shape=(corpus_vocab.vocab_size,))

    for token, token_frequency in token_counts.items():
        token_index = corpus_vocab.get_token_index(token=token)
        embedding[token_index] += token_frequency

    return embedding.tolist()


def embed_documents(
    document_list: Optional[list[str]],
    document_config: DocumentConfig,
    vocab_config: CorpusVocabConfig
) -> list[list[int]]:
    corpus_vocab = CorpusVocab().create(documents=document_list, document_config=document_config, max_tokens=400)

    embeddings = []

    for document in document_list:

        if document_config.lower_text:
            document = lower_text(document=document)
        if document_config.remove_punctuation:
            document = remove_punctuation(document=document)

        document_tokens = tokenize_document(document=document)
        if document_config.remove_punctuation:
            document_tokens = remove_stopwords(document_tokens=document_tokens, document_config=document_config)

        embeddings.append(
            bag_of_words_embedding(
                tokens=document_tokens,
                corpus_vocab=corpus_vocab
            )
        )

    return embeddings










