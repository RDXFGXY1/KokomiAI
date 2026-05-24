# Memory System

Kokomi remembers things across sessions using SQLite.
All memory operations go through `MemoryManager` — never touch the store directly.

---

## What Gets Stored

| Table | Content | Persists |
|---|---|---|
| `memories` | facts you ask Kokomi to remember | yes |
| `projects` | project names and descriptions | yes |
| `tasks` | tasks per project with priority | yes |
| `chat_history` | full conversation log | yes |

Everything lives in `data/kokomi.db` — a single SQLite file.

---

## Using Memory from the CLI

```
/remember working on BPE tokenizer upgrade
/remember RDXFGXY1 prefers dark red interfaces
/remember TravelersBot uses discord.py

/recall tokenizer          ← finds relevant memories
/recall project structure  ← semantic search if embedder loaded
```

---

## MemoryManager API

Other modules use this interface:

```python
from memory.memory_manager import MemoryManager

mem = MemoryManager()

# Store a memory
mem.remember("RDXFGXY1 likes PyTorch", tags=["preference", "tech"])

# Retrieve relevant memories
results = mem.recall("pytorch", top_k=5)

# Projects
mem.add_project("kokomi", "local AI assistant")
mem.get_projects()

# Tasks
mem.add_task("kokomi", "implement BPE tokenizer", priority=3)
mem.get_tasks("kokomi")

# Chat log
mem.log("user", "hi")
mem.log("assistant", "Hey. What are we working on?")
mem.get_context(n=10)   # last 10 messages
```

---

## Semantic Search

If `sentence-transformers` is installed, recall uses embedding similarity
to find memories that are *semantically* related to your query — not just
exact keyword matches.

```
/recall machine learning
→ finds: "Kokomi uses PyTorch"
→ finds: "transformer architecture has attention heads"
→ finds: "training loss should decrease each epoch"
```

Without sentence-transformers it falls back to returning the most recent memories.

Install it:
```bash
pip install sentence-transformers
```

---

## Database Location

Default: `data/kokomi.db`

Change in `config/settings.py`:
```python
@dataclass
class MemoryConfig:
    db_path: str = "data/kokomi.db"
```

To reset all memory: delete `kokomi.db` and restart. A fresh database
will be created automatically.
