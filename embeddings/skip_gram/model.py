import torch.nn as nn

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size: int):

        super(SkipGramModel, self).__init__()

        """
            embedding layer
            -> input: BATCH_SIZE x VOCAB_SIZE x 1
            -> output: BATCH_SIZE x EMBED_DIMMENSION
        """
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )

        """
            linear layer
            -> input: BATCH_SIZE x EMBED_DIMMENSION
            -> output: BATCH_SIZE x VOCAB_SIZE
        """
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, x):
        """
            forward pass
        """
        x = self.embeddings(x)
        x = self.linear(x)
        return x



