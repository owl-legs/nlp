from dataclasses import dataclass


@dataclass
class CorpusVocabConfig:
    max_tokens: int
    randomize_token_index: bool

