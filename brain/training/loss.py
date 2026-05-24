"""
Loss functions.

For language modeling we use cross-entropy:
    predict next token → compare with actual next token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """
    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len)
        pad_id:  token id to ignore in loss (padding tokens don't count)
    Returns:
        scalar loss
    """
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        ignore_index=pad_id,
    )
