import string
from typing import Optional

from embeddings.utils.config import DocumentConfig

STANDARD_PUNCTUATION = string.punctuation


def lower_text(document: str) -> str:
    return document.lower()


def remove_punctuation(document: str) -> str:
    return document.translate(str.maketrans('', '', STANDARD_PUNCTUATION))


def tokenize_document(
        document: str,
) -> list[str]:
    tokens = document.split()

    valid_tokens = [
        token
        for token in tokens
        if token and not token.isspace()
    ]
    return valid_tokens


def remove_stopwords(document_tokens: list[str], document_config: DocumentConfig) -> list[str]:
    return [
        token for token in document_tokens if token not in document_config.stopwords
    ]




