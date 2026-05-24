"""
Token + positional embeddings.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale by sqrt(d_model) — stabilizes gradients early in training
        return self.embed(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    The model has no notion of order by default — this injects position info.

    For position pos and dimension i:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the encoding table once — shape (max_len, embed_dim)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()          # (max_len, 1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * -(math.log(10000.0) / embed_dim)
        )                                                              # (embed_dim/2,)

        pe[:, 0::2] = torch.sin(pos * div)   # even dims → sin
        pe[:, 1::2] = torch.cos(pos * div)   # odd  dims → cos

        # Register as buffer: saved with model, not a trainable parameter
        self.register_buffer("pe", pe.unsqueeze(0))                   # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int, dropout: float):
        super().__init__()
        self.token    = TokenEmbedding(vocab_size, embed_dim)
        self.position = PositionalEncoding(embed_dim, max_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.position(self.token(x))