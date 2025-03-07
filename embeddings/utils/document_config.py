from dataclasses import dataclass, field

import string
from typing import Optional

from stopwords import get_stopwords

STANDARD_PUNCTUATION = string.punctuation
STANDARD_STOPWORDS = get_stopwords()


@dataclass
class DocumentConfig:
    lower_text: bool
    remove_punctuation: bool
    remove_stopwords: bool

    excluded_punctuation: field(default_factory=list)
    stopwords: field(default_factory=list)

    @classmethod
    def create(cls,
               lower_text: bool,
               exclude_punctuation: bool,
               remove_stopwords: bool,
               excluded_punctuation: Optional[list[str]],
               stopwords: Optional[list[str]]
               ):
        return DocumentConfig(
            lower_text=lower_text,
            remove_punctuation=exclude_punctuation,
            remove_stopwords=remove_stopwords,
            excluded_punctuation=excluded_punctuation or STANDARD_PUNCTUATION,
            stopwords=stopwords or STANDARD_STOPWORDS
        )

