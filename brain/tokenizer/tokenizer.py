"""
Tokenizers for Kokomi.

CharTokenizer  — character-level (Stage 1, kept for reference)
BPETokenizer   — byte-pair encoding (Stage 3, production)

BPE works by:
1. Start with all characters as tokens
2. Find the most frequent pair of tokens
3. Merge them into a new token
4. Repeat N times (vocab_size controls this)

Result: common words become single tokens, rare words split into pieces.
"running" → ["run", "ning"]
"pytorch" → ["py", "torch"]
"wahts"   → ["wa", "hts"]  ← handles typos gracefully
"""

import json
import re
from collections import Counter
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class CharTokenizer:
    """Character-level tokenizer. Kept for reference."""

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SPECIAL = [PAD, UNK, BOS, EOS]

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}
        self._built = False

    def build(self, text: str) -> None:
        chars = sorted(set(text))
        vocab = self.SPECIAL + chars
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self._built = True
        logger.info(f"CharTokenizer vocab: {len(vocab)} tokens")

    def encode(self, text: str) -> list[int]:
        unk = self.char_to_id.get(self.UNK, 1)
        return [self.char_to_id.get(c, unk) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, self.UNK) for i in ids)

    @property
    def vocab_size(self) -> int: return len(self.char_to_id)
    @property
    def pad_id(self)  -> int: return self.char_to_id.get(self.PAD, 0)
    @property
    def bos_id(self)  -> int: return self.char_to_id.get(self.BOS, 2)
    @property
    def eos_id(self)  -> int: return self.char_to_id.get(self.EOS, 3)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "vocab": self.char_to_id}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.char_to_id = data["vocab"]
        tok.id_to_char = {v: k for k, v in tok.char_to_id.items()}
        tok._built = True
        return tok


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer.

    Better than char-level because:
    - Common words become single tokens (faster, more context per sequence)
    - Handles unknown words by splitting into known subwords
    - Typos get split into known pieces instead of unknown chars

    Usage:
        tok = BPETokenizer(vocab_size=2000)
        tok.build(text)
        ids = tok.encode("hello world")
        txt = tok.decode(ids)
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SPECIAL = [PAD, UNK, BOS, EOS]

    def __init__(self, vocab_size: int = 2000):
        self.target_vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []  # learned merge rules
        self._built = False

    def build(self, text: str) -> None:
        """Learn BPE merges from text and build vocabulary."""
        logger.info(f"Building BPE tokenizer (target vocab: {self.target_vocab_size})...")

        # Step 1: start with character-level words
        # Split text into words, represent each word as tuple of chars + end marker
        words = self._get_word_freqs(text)

        # Step 2: initialize vocab with all unique characters
        vocab = set()
        for word in words:
            for char in word:
                vocab.add(char)

        # Step 3: iteratively merge most frequent pairs
        num_merges = self.target_vocab_size - len(self.SPECIAL) - len(vocab)
        num_merges = max(0, num_merges)

        for i in range(num_merges):
            pairs = self._get_pair_freqs(words)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # Merge that pair everywhere in the corpus
            words = self._merge_pair(words, best_pair)
            merged = best_pair[0] + best_pair[1]
            vocab.add(merged)

            if (i + 1) % 200 == 0:
                logger.info(f"  Merges: {i+1}/{num_merges} | Vocab: {len(vocab) + len(self.SPECIAL)}")

        # Step 4: build final vocab with special tokens first
        all_tokens = self.SPECIAL + sorted(vocab)
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self._built = True
        logger.info(f"BPE tokenizer built: {len(all_tokens)} tokens, {len(self.merges)} merges")

    def _get_word_freqs(self, text: str) -> dict[tuple, int]:
        """Split text into words and count frequencies."""
        # Simple whitespace split, add end-of-word marker
        word_freqs: dict[tuple, int] = Counter()
        for word in re.findall(r'\S+', text.lower()):
            # Represent word as tuple of characters with end marker on last char
            chars = tuple(list(word[:-1]) + [word[-1] + "</w>"])
            word_freqs[chars] += 1
        return dict(word_freqs)

    def _get_pair_freqs(self, words: dict[tuple, int]) -> dict[tuple, int]:
        """Count frequency of all adjacent token pairs."""
        pairs: dict[tuple, int] = Counter()
        for word, freq in words.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return dict(pairs)

    def _merge_pair(self, words: dict[tuple, int], pair: tuple[str, str]) -> dict[tuple, int]:
        """Merge all occurrences of pair in every word."""
        new_words = {}
        merged = pair[0] + pair[1]
        for word, freq in words.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words[tuple(new_word)] = freq
        return new_words

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids using learned BPE merges."""
        if not self._built:
            raise RuntimeError("Tokenizer not built. Call build() first.")

        ids = []
        unk_id = self.token_to_id.get(self.UNK, 1)

        for word in re.findall(r'\S+|\s+', text.lower()):
            if word.isspace():
                # Encode spaces as individual characters
                for ch in word:
                    ids.append(self.token_to_id.get(ch, unk_id))
                continue

            # Apply BPE: start with characters, apply merges in order
            chars = list(word[:-1]) + [word[-1] + "</w>"]
            tokens = chars

            for pair in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for token in tokens:
                ids.append(self.token_to_id.get(token, unk_id))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        tokens = [self.id_to_token.get(i, self.UNK) for i in ids]
        text = "".join(tokens)
        # Remove end-of-word markers
        text = text.replace("</w>", " ")
        return text.strip()

    @property
    def vocab_size(self) -> int: return len(self.token_to_id)
    @property
    def pad_id(self)  -> int: return self.token_to_id.get(self.PAD, 0)
    @property
    def bos_id(self)  -> int: return self.token_to_id.get(self.BOS, 2)
    @property
    def eos_id(self)  -> int: return self.token_to_id.get(self.EOS, 3)

    def save(self, path: str) -> None:
        data = {
            "type": "bpe",
            "vocab_size": self.target_vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"BPE tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        tok._built = True
        logger.info(f"BPE tokenizer loaded: {tok.vocab_size} tokens")
        return tok
