"""
Full decoder-only Transformer (GPT-style).

Data flow:
    token ids
        → Embeddings (token + positional)
        → N × TransformerBlock
            → LayerNorm → MultiHeadAttention → residual
            → LayerNorm → FeedForward        → residual
        → final LayerNorm
        → Linear projection → logits over vocab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embeddings import Embeddings
from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Two-layer MLP inside each transformer block.
    Expands to ff_dim (usually 4x embed_dim) then back.
    GELU activation — smoother than ReLU, works better for transformers.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block. Uses Pre-LN (normalize before attention),
    which trains more stably than the original Post-LN paper.

    x → norm → attention → x + residual
      → norm → feedforward → x + residual
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff    = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Residual connection: output = input + transform(input)
        # This lets gradients flow directly back through the network (no vanishing)
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class KokomiTransformer(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int,
        num_heads:   int,
        num_layers:  int,
        ff_dim:      int,
        max_seq_len: int,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_dim, max_seq_len, dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm     = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between input embedding and output projection
        # The model learns the same representation for input and output tokens
        self.out_proj.weight = self.embeddings.token.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights — small values prevent exploding activations at start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq_len) — token ids
        x = self.embeddings(x)          # → (batch, seq_len, embed_dim)
        for block in self.blocks:
            x = block(x, mask)          # each block refines the representation
        x = self.norm(x)                # final layer norm
        return self.out_proj(x)         # → (batch, seq_len, vocab_size) logits

    @torch.no_grad()
    def generate(
        self,
        prompt:         torch.Tensor,
        max_new_tokens: int   = 200,
        temperature:    float = 0.8,
        top_k:          int   = 40,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        At each step: run forward pass → sample next token → append → repeat.

        temperature: higher = more creative/random, lower = more focused
        top_k: only sample from the top-k most likely tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len if needed
            ctx    = prompt[:, -512:]
            logits = self(ctx)                        # (1, T, vocab_size)
            logits = logits[:, -1, :] / temperature  # take last token's logits

            # Top-k filtering: zero out everything outside top k
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample
            prompt     = torch.cat([prompt, next_token], dim=1)

        return prompt

    @classmethod
    def from_config(cls, config) -> "KokomiTransformer":
        return cls(
            vocab_size  = config.vocab_size,
            embed_dim   = config.embed_dim,
            num_heads   = config.num_heads,
            num_layers  = config.num_layers,
            ff_dim      = config.ff_dim,
            max_seq_len = config.max_seq_len,
            dropout     = config.dropout,
        )