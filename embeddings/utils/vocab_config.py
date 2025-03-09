from dataclasses import dataclass

from embeddings.utils.constants import UNKNOWN_WORD
@dataclass
class CorpusVocabConfig:
    max_tokens: int
    randomize_token_index: bool
    unknown_word_identifier: str = UNKNOWN_WORD

