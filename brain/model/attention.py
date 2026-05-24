"""
Multi-head self-attention — the core of every transformer.

The idea: instead of each token only knowing about itself,
every token gets to "look at" all other tokens and decide
what's relevant. That's attention.

Scaled dot-product attention:
    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

Q = "what am I looking for?"
K = "what do I contain?"
V = "what do I actually give you?"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention in one clean module.
    Instead of separate SelfAttentionHead objects, we batch all heads
    together using reshape tricks — this is how real implementations do it.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads  # each head works on this slice

        # Single linear projects input into Q, K, V all at once (3x efficient)
        self.qkv_proj  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape   # batch, sequence length, embed_dim

        # 1. Project to Q, K, V — all in one matmul, then split
        qkv = self.qkv_proj(x)                          # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)     # each (B, T, C)

        # 2. Reshape for multi-head: split C into (num_heads, head_dim)
        #    then move heads dimension next to batch so we can batch matmul
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            # result: (B, num_heads, T, head_dim)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # 3. Scaled dot-product attention
        #    scores = Q @ K^T / sqrt(head_dim)  → (B, heads, T, T)
        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 4. Causal mask: token at position i cannot attend to j > i
        #    This makes it autoregressive (predicts next token, not future tokens)
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))

        # 5. Optional padding mask (passed in from outside)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # 6. Softmax → attention weights, then dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 7. Weighted sum of values
        out = torch.matmul(weights, v)           # (B, heads, T, head_dim)

        # 8. Merge heads back: (B, heads, T, head_dim) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 9. Final output projection
        return self.out_proj(out)