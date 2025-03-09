from dataclasses import dataclass
import torch.nn as nn

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1

@dataclass
class ContinuousBagOfWords(nn.module):
    def __init__(self, vocab_size: int):

        super(ContinuousBagOfWords, self).__init__()

        # embedding layer
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        # linear layer
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x



