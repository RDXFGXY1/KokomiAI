# CLI Reference

Start the terminal UI:
```bash
python main.py
```

---

## Commands

### General
| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/status` | Show model load status |
| `/clear` | Clear the terminal |
| `/quit` or `/exit` | Exit Kokomi |

### Projects
| Command | Description |
|---|---|
| `/projects` | List all active projects |
| `/project <name>` | Create a new project |

### Tasks
| Command | Description |
|---|---|
| `/tasks <project>` | List open tasks for a project |
| `/task <project> <title>` | Add a task to a project |
| `/done <id>` | Mark a task as complete |

### Memory
| Command | Description |
|---|---|
| `/remember <text>` | Store a memory |
| `/recall <query>` | Search stored memories |

### Tools
| Command | Description |
|---|---|
| `/scan` | Show directory tree of current folder |
| `/todos` | Find TODO/FIXME/HACK comments in source files |

---

## Natural Language

Any input that doesn't start with `/` is sent to the model:

```
> what is pytorch
KOKOMI ▸ An open-source machine learning framework. Core unit is the tensor...

> help me understand attention
KOKOMI ▸ Attention lets every token look at every other token...
```

If the model is not loaded, you'll see:
```
KOKOMI ▸ model not loaded — run python main.py --train first
```

---

## Training Mode

```bash
python main.py --train
```

Runs the full training pipeline. See [TRAINING.md](TRAINING.md) for details.

---

## Debug Mode

```bash
python main.py --debug
```

Enables verbose logging to both terminal and `data/kokomi.log`.
