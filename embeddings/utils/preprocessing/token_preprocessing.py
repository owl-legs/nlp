def remove_stopwords(document_tokens: list[str], stopwords: list[str]) -> list[str]:
    return [
        token for token in document_tokens if token not in stopwords
    ]


def remove_unknown_words(document_tokens: list[str],
                         vocab: dict,
                         unknown_word_identifier: str) -> list[str]:
    return [token if vocab.get(token) else unknown_word_identifier for token in document_tokens]



