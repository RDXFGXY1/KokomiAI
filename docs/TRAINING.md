# Training Guide

This document explains how to prepare data, configure the model, and run training.

---

## Overview

Kokomi learns from plain `.txt` files you place in `data/raw/`.
The trainer automatically merges all files it finds there into one corpus.
No dataset downloads, no HuggingFace, no external dependencies.

---

## Step 1 — Prepare Training Data

Drop any `.txt` files into `data/raw/`. The more text, the better.

**Recommended sources:**

| Source | Where to get it | Size |
|---|---|---|
| Project Gutenberg books | gutenberg.org | 300KB–2MB each |
| Your own notes and docs | copy/paste into a .txt | any size |
| Conversation examples | write them manually | see format below |
| Code files | concatenate your .py files | any size |

**Minimum viable:** 500KB total. **Recommended:** 3–10MB.

---

## Step 2 — Add Conversation Examples

The model learns to chat from examples written in this exact format:

```
Kyros: hi
Kokomi: Hey. What are we working on?

Kyros: what is python
Kokomi: A general-purpose programming language. Known for readability and a huge ecosystem.

Kyros: i have a bug
Kokomi: Paste the error. Let's look at it.
```

Save these in `data/raw/train.txt`. The more varied the examples, the better
the model generalizes. Aim for at least 200 pairs before training.

**Rules for good conversation examples:**
- One blank line between each pair
- Keep Kokomi's responses short and direct
- Vary the phrasing of similar questions
- Include greetings, technical questions, and short replies

---

## Step 3 — Configure the Model

Open `config/settings.py` and adjust to your hardware and time budget:

```python
@dataclass
class ModelConfig:
    vocab_size:  int   = 4000   # BPE vocab size — don't go below 1000
    embed_dim:   int   = 256    # model width — increase for more capacity
    num_heads:   int   = 8      # attention heads — must divide embed_dim
    num_layers:  int   = 6      # transformer depth — more = smarter but slower
    ff_dim:      int   = 1024   # feed-forward inner dim — usually 4x embed_dim
    max_seq_len: int   = 256    # context window — longer = more memory usage
    dropout:     float = 0.15   # regularization — increase if overfitting

@dataclass
class TrainingConfig:
    batch_size:    int   = 128    # lower if you get CUDA out-of-memory errors
    learning_rate: float = 3e-4   # standard AdamW starting LR
    epochs:        int   = 30     # more epochs = more training time
    grad_clip:     float = 1.0    # prevents exploding gradients
    device:        str   = "cuda" # change to "cpu" if no GPU
```

**Hardware reference:**

| VRAM | Recommended batch_size | Recommended model size |
|---|---|---|
| 4GB  | 64  | embed=128, layers=4 |
| 6GB  | 128 | embed=256, layers=6 |
| 8GB  | 256 | embed=256, layers=8 |
| 12GB | 512 | embed=512, layers=8 |

---

## Step 4 — Run Training

```bash
python main.py --train
```

The trainer will:
1. Load and merge all `.txt` files from `data/raw/`
2. Build the BPE tokenizer from the corpus
3. Save `tokenizer.json` to `data/checkpoints/`
4. Train the model for the configured number of epochs
5. Save the final checkpoint to `data/checkpoints/`

---

## Understanding the Loss Curve

```
Epoch 1  → 5.5–6.0   model knows nothing, random predictions
Epoch 5  → 4.0–4.5   learning basic token patterns
Epoch 10 → 3.0–3.5   recognizing words and phrases
Epoch 20 → 2.5–3.0   generating coherent text
Epoch 30 → 2.0–2.5   solid generalization on this data size
```

**val loss should roughly track train loss.**
If train loss drops sharply but val loss rises, the model is overfitting.
Fix: add more data, increase dropout, reduce epochs.

---

## Checkpoints

Every training run saves:
- `data/checkpoints/kokomi_efinal_s0_YYYYMMDD_HHMMSS.pt` — model weights
- `data/checkpoints/tokenizer.json` — BPE vocabulary and merge rules

The `.pt` file is **not included in the repository**.
You must train your own. This is intentional.

To load a specific checkpoint instead of the latest:
```bash
# Edit ui/cli/terminal.py — pass checkpoint_path to Terminal()
terminal = Terminal(checkpoint_path="data/checkpoints/your_file.pt")
```

---

## Retraining

To improve the model over time:
1. Add more `.txt` files to `data/raw/`
2. Add more conversation pairs to `train.txt`
3. Run `python main.py --train` again

Each retrain starts from scratch with a fresh model.
Future versions will support resuming from a checkpoint.
