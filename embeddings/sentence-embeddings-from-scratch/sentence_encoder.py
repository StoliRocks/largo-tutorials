"""
Sentence Embeddings from Scratch with PyTorch

Build a complete sentence encoder from the ground up. This implementation
demonstrates tokenization, embedding layers, LSTM encoding, and pooling
strategies for creating sentence-level representations.

Tutorial: https://largo.dev/tutorials/embeddings/sentence-embeddings-from-scratch/
Author: Steven W. White
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenizer:
    """
    Character-level tokenizer for demonstration purposes.

    In production, use subword tokenizers like BPE (GPT) or WordPiece (BERT)
    which handle unknown words by breaking them into known subword units.
    """

    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz ")
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = len(self.chars) + 1
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        """Encode a single text string to token IDs."""
        text = text.lower()
        tokens = [self.char_to_idx.get(c, self.char_to_idx['<UNK>'])
                  for c in text[:max_len]]
        tokens += [0] * (max_len - len(tokens))  # Pad to max_len
        return tokens

    def batch_encode(self, texts: list[str], max_len: int = 64) -> list[list[int]]:
        """Encode multiple texts to token IDs."""
        return [self.encode(t, max_len) for t in texts]


class SentenceEncoder(nn.Module):
    """
    Bidirectional LSTM-based sentence encoder with mean pooling.

    Architecture:
        Token IDs → Embedding → BiLSTM → Mean Pooling → L2 Normalize

    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of token embeddings (default: 128)
        hidden_dim: Hidden dimension of LSTM (default: 256)
        pooling: Pooling strategy - 'mean', 'max', or 'cls' (default: 'mean')
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, pooling: str = 'mean'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.1
        )
        self.output_dim = hidden_dim * 2  # Bidirectional doubles the size
        self.pooling = pooling

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode token IDs to sentence embeddings.

        Args:
            token_ids: (batch_size, seq_len) tensor of token IDs
            attention_mask: Optional (batch_size, seq_len) mask, 1 for real tokens

        Returns:
            (batch_size, output_dim) tensor of L2-normalized sentence embeddings
        """
        # Create attention mask from token_ids if not provided
        if attention_mask is None:
            attention_mask = (token_ids != 0).long()

        # Embed tokens: (batch, seq, embed_dim)
        embedded = self.embedding(token_ids)

        # Encode with BiLSTM: (batch, seq, hidden*2)
        lstm_out, _ = self.lstm(embedded)

        # Pool to sentence embedding
        if self.pooling == 'mean':
            sentence_emb = self._mean_pool(lstm_out, attention_mask)
        elif self.pooling == 'max':
            sentence_emb = self._max_pool(lstm_out, attention_mask)
        else:  # cls - use first token
            sentence_emb = lstm_out[:, 0, :]

        # L2 normalize for cosine similarity
        return F.normalize(sentence_emb, p=2, dim=1)

    def _mean_pool(self, token_embeddings: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Average all token embeddings, ignoring padding."""
        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pool(self, token_embeddings: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
        """Take max value for each dimension across the sequence."""
        mask = attention_mask.unsqueeze(-1).float()
        token_embeddings = token_embeddings.masked_fill(mask == 0, -1e9)
        return token_embeddings.max(dim=1)[0]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between two embedding tensors."""
    return torch.mm(a, b.T)


def main():
    """Demonstrate the sentence encoder with example sentences."""
    print("=" * 60)
    print("Sentence Embeddings from Scratch")
    print("=" * 60)

    # Initialize tokenizer and encoder
    tokenizer = SimpleTokenizer()
    encoder = SentenceEncoder(tokenizer.vocab_size)
    encoder.eval()

    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Embedding output dimension: {encoder.output_dim}")

    # Test sentences
    sentences = [
        "the cat sat on the mat",
        "a cat was sitting on a mat",
        "the dog ran in the park",
        "machine learning is fascinating"
    ]

    print("\nTest sentences:")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")

    # Encode sentences
    with torch.no_grad():
        token_ids = torch.tensor(tokenizer.batch_encode(sentences))
        embeddings = encoder(token_ids)

    # Compute similarity matrix
    similarity = cosine_similarity(embeddings, embeddings)

    print("\nCosine similarity matrix:")
    print(similarity.numpy().round(3))

    print("\nNote: This is an UNTRAINED model with random weights.")
    print("After training with contrastive learning, similar sentences")
    print("will have similarity scores of 0.8+ while unrelated ones")
    print("will be near 0.")

    # Show embedding shape
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Each sentence is represented as a {embeddings.shape[1]}-dimensional vector.")


if __name__ == "__main__":
    main()
