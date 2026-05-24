"""
Dataset classes for training.

TextDataset:   raw .txt files → (input, target) tensor pairs
Stage 1 target: understand how sliding windows work.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from brain.tokenizer.tokenizer import CharTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class TextDataset(Dataset):
    """
    Sliding window dataset for language modeling.

    Given text: "hello world"
    Window size 4, stride 1:
        input  → [h, e, l, l]  target → [e, l, l, o]
        input  → [e, l, l, o]  target → [l, l, o, ' ']
        ...

    This is how the model learns to predict the next token.
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int = 128):
        self.tokenizer = tokenizer
        self.seq_len   = seq_len
        self.data      = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        logger.info(f"Dataset: {len(self.data)} tokens, {len(self)} samples")

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk  = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]   # input:  all tokens except last
        y = chunk[1:]    # target: all tokens except first (shifted by 1)
        return x, y

    @classmethod
    def from_file(cls, path: str, tokenizer: CharTokenizer, seq_len: int = 128) -> "TextDataset":
        text = Path(path).read_text(encoding="utf-8")
        logger.info(f"Loaded {path}: {len(text)} characters")
        return cls(text, tokenizer, seq_len)
