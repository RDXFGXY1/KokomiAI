# Tokenizer

Kokomi ships with two tokenizers. Understanding the difference matters
when you're debugging or extending the model.

---

## CharTokenizer (Stage 1)

Treats every single character as a token.

```
"hello" → [7, 4, 11, 11, 14]
"h" → 1 token
"e" → 1 token
```

**Vocab size:** ~80–120 tokens (just the unique characters in your text)

**Pros:** Simple. Easy to understand. Good for learning.

**Cons:**
- Sequences are very long (every character = one position)
- Typos produce completely unknown patterns
- No word-level understanding

---

## BPETokenizer (Stage 2+, default)

Byte-Pair Encoding. Starts with characters, then learns to merge
frequent pairs into subword tokens.

```
"running" → ["run", "ning</w>"]   (2 tokens)
"pytorch" → ["py", "torch</w>"]   (2 tokens)
"hello"   → ["hello</w>"]         (1 token — common enough to be one)
"wahts"   → ["wa", "hts</w>"]     (splits gracefully — handles typos)
```

**Vocab size:** configurable, default 4000

**Pros:**
- Common words become single tokens (more context per sequence)
- Unknown words split into known pieces instead of failing
- Handles typos gracefully
- Much better language understanding

**Cons:** Takes 2–5 minutes to build from corpus. Saved to `tokenizer.json`.

---

## How BPE is Built

```
1. Start: every character is a token
   "hello" → ["h", "e", "l", "l", "o</w>"]

2. Count all adjacent pairs across the whole corpus
   ("h", "e") appears 12,400 times
   ("e", "l") appears 8,200 times
   ...

3. Merge the most frequent pair into a new token
   ("h", "e") → "he"
   "hello" → ["he", "l", "l", "o</w>"]

4. Repeat until vocab_size is reached
```

Each merge rule is saved in `tokenizer.json` so the same encoding
can be reproduced at inference time.

---

## The tokenizer.json File

Located at `data/checkpoints/tokenizer.json`. Contains:

```json
{
  "type": "bpe",
  "vocab_size": 4000,
  "token_to_id": {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "h": 4,
    "e": 5,
    ...
    "the</w>": 847,
    "and</w>": 312,
    ...
  },
  "merges": [
    ["t", "h"],
    ["th", "e</w>"],
    ...
  ]
}
```

**This file must exist alongside the `.pt` checkpoint.**
If you move or delete `tokenizer.json`, the model cannot generate text.

---

## Special Tokens

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding — ignored in loss calculation |
| `<UNK>` | 1 | Unknown token fallback |
| `<BOS>` | 2 | Beginning of sequence |
| `<EOS>` | 3 | End of sequence |

---

## Switching Tokenizers

To go back to CharTokenizer for experiments, edit `brain/training/trainer.py`:

```python
# Replace this line:
self.tokenizer = BPETokenizer(vocab_size=cfg.model.vocab_size)

# With this:
self.tokenizer = CharTokenizer()
```

And update `config/settings.py`:
```python
vocab_size: int = 100   # small enough for char-level
```
