import string

STANDARD_PUNCTUATION = string.punctuation


def lower_text(document: str) -> str:
    return document.lower()


def remove_punctuation(document: str) -> str:
    return document.translate(str.maketrans('', '', STANDARD_PUNCTUATION))


def tokenize_document(document: str) -> list[str]:
    tokens = document.split()

    valid_tokens = [
        token
        for token in tokens
        if token and not token.isspace()
    ]
    return valid_tokens



