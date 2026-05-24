# Architecture

Kokomi is a modular Python application. Each directory has one responsibility.
Nothing imports in a circle. The brain does not know about the UI.

---

## Data Flow

```
User Input
    │
    ▼
ui/cli/terminal.py          ← reads input, routes commands
    │
    ├── /command → _handle_command()
    │
    └── natural text → brain/inference.py
                            │
                            ▼
                    prompt wrapped in template
                    "RDXFGXY1: {input}\nKokomi:"
                            │
                            ▼
                    brain/model/transformer.py  ← forward pass
                            │
                            ▼
                    generated token ids
                            │
                            ▼
                    brain/tokenizer/tokenizer.py ← decode to text
                            │
                            ▼
                    personality/response_shaper.py ← tone layer
                            │
                            ▼
                    ui/cli/renderer.py ← print to terminal
```

---

## Directory Map

```
kokomi/
│
├── main.py                     Entry point. Parses --train / --web flags.
│
├── config/
│   └── settings.py             Single source of truth for all configuration.
│                               ModelConfig, TrainingConfig, MemoryConfig, PersonalityConfig.
│
├── brain/                      Everything PyTorch. No UI imports allowed here.
│   ├── model/
│   │   ├── transformer.py      KokomiTransformer — decoder-only GPT-style model.
│   │   ├── attention.py        MultiHeadAttention — scaled dot-product, causal mask.
│   │   └── embeddings.py       TokenEmbedding + sinusoidal PositionalEncoding.
│   ├── tokenizer/
│   │   └── tokenizer.py        CharTokenizer (Stage 1) and BPETokenizer (Stage 2+).
│   ├── training/
│   │   ├── trainer.py          Full train/eval/checkpoint loop.
│   │   ├── dataset.py          TextDataset — sliding window over token ids.
│   │   └── loss.py             cross_entropy_loss with padding mask.
│   └── inference.py            KokomiInference — loads checkpoint, runs generate().
│
├── memory/                     Persistence layer. No PyTorch imports allowed here.
│   ├── store/
│   │   └── sqlite_store.py     Raw SQLite access — memories, projects, tasks, chat log.
│   ├── embeddings/
│   │   └── embedder.py         sentence-transformers wrapper for semantic search.
│   └── memory_manager.py       High-level interface used by all other modules.
│
├── personality/
│   ├── kokomi.py               Personality definition — name, tone, greetings.
│   └── response_shaper.py      Wraps raw model output in Kokomi's voice.
│
├── planner/
│   ├── task_manager.py         Task CRUD over MemoryManager.
│   └── project_tracker.py      Project listing and creation.
│
├── tools/
│   ├── tool_registry.py        Plugin registry — register and call tools by name.
│   ├── file_scanner.py         Directory tree walker and TODO finder.
│   └── code_helper.py          Traceback parser and error explainer.
│
├── ui/
│   ├── cli/
│   │   ├── terminal.py         Main CLI loop — input, routing, command dispatch.
│   │   └── renderer.py         Rich output helpers — panels, tables, message formatting.
│   └── web/
│       ├── server.py           Flask server stub (Stage 4).
│       └── routes.py           API route stubs (Stage 4).
│
├── utils/
│   ├── logger.py               Centralized logging — all modules use get_logger(__name__).
│   └── helpers.py              get_device(), count_parameters(), format_number(), etc.
│
└── data/
    ├── raw/                    Drop your .txt training files here.
    ├── processed/              Reserved for future preprocessed datasets.
    └── checkpoints/            Saved .pt model files and tokenizer.json live here.
```

---

## Module Dependency Rules

```
brain      → config, utils only
memory     → config, utils only
personality → config only
planner    → memory only
tools      → utils only
ui         → all of the above (top layer)
```

If you ever find `brain` importing from `ui`, something is wrong.

---

## Model Architecture

```
Input token ids  (batch, seq_len)
        │
        ▼
TokenEmbedding   nn.Embedding(vocab_size, embed_dim) × sqrt(embed_dim)
        +
PositionalEncoding  sinusoidal, fixed, registered as buffer
        │
        ▼
TransformerBlock × num_layers
    ├── LayerNorm
    ├── MultiHeadAttention   (causal mask, num_heads parallel heads)
    ├── residual connection
    ├── LayerNorm
    ├── FeedForward          Linear → GELU → Dropout → Linear
    └── residual connection
        │
        ▼
LayerNorm (final)
        │
        ▼
Linear(embed_dim → vocab_size)   weight-tied with input embedding
        │
        ▼
Logits  (batch, seq_len, vocab_size)
```
