# Configuration

All settings live in `config/settings.py`. Edit this file before training.
Never hardcode values in other modules — always import from Settings.

---

## ModelConfig

Controls the transformer architecture.

```python
@dataclass
class ModelConfig:
    vocab_size:  int   = 4000
    embed_dim:   int   = 256
    num_heads:   int   = 8
    num_layers:  int   = 6
    ff_dim:      int   = 1024
    max_seq_len: int   = 256
    dropout:     float = 0.15
```

| Field | What it does | Rule |
|---|---|---|
| `vocab_size` | BPE vocabulary size | min 1000, max ~8000 for small models |
| `embed_dim` | model width | must be divisible by `num_heads` |
| `num_heads` | parallel attention heads | must divide `embed_dim` evenly |
| `num_layers` | transformer depth | more = smarter + slower |
| `ff_dim` | feed-forward size | usually 4x `embed_dim` |
| `max_seq_len` | context window length | longer = more VRAM |
| `dropout` | regularization | 0.1–0.2 for most cases |

---

## TrainingConfig

Controls the training loop.

```python
@dataclass
class TrainingConfig:
    batch_size:       int   = 128
    learning_rate:    float = 3e-4
    epochs:           int   = 30
    grad_clip:        float = 1.0
    checkpoint_every: int   = 500
    device:           str   = "cuda"
```

| Field | What it does | Tip |
|---|---|---|
| `batch_size` | samples per gradient step | lower if CUDA OOM |
| `learning_rate` | step size for AdamW | 1e-4 to 5e-4 is safe |
| `epochs` | full passes over dataset | 10–30 for most use cases |
| `grad_clip` | gradient norm cap | keep at 1.0 |
| `checkpoint_every` | save every N steps | 500 is fine |
| `device` | `"cuda"` or `"cpu"` | always use cuda if available |

---

## MemoryConfig

```python
@dataclass
class MemoryConfig:
    db_path:              str   = "data/kokomi.db"
    max_memory_entries:   int   = 1000
    similarity_threshold: float = 0.75
```

---

## PersonalityConfig

Controls Kokomi's tone and behavior.

```python
@dataclass
class PersonalityConfig:
    name:          str  = "Kokomi"
    tone:          str  = "calm"    # calm | playful | sharp
    supportive:    bool = True
    humor:         bool = True
    sarcasm_level: int  = 2         # 0–5
```

Change `tone` to `"playful"` for more energetic greetings,
or `"sharp"` for minimal one-word responses.

---

## Presets

**Fast experiment (5 min training):**
```python
embed_dim=128, num_heads=4, num_layers=4, ff_dim=512
batch_size=256, epochs=10, max_seq_len=128
```

**Balanced (20–30 min):**
```python
embed_dim=256, num_heads=8, num_layers=6, ff_dim=1024
batch_size=128, epochs=30, max_seq_len=256
```

**Max quality (1–2 hours):**
```python
embed_dim=512, num_heads=8, num_layers=8, ff_dim=2048
batch_size=64, epochs=50, max_seq_len=512
```
