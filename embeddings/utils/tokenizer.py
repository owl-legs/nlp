import nltk
import re
import string

STANDARD_PUNCTUATION = string.punctuation

def tokenize(document: str) -> list[str]:

    document = document.lower()
    document = document.translate(str.maketrans('', '', STANDARD_PUNCTUATION))

    return nltk.word_tokenize(document)
