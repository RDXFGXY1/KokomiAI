<div align="center">

```
██╗  ██╗ ██████╗ ██╗  ██╗ ██████╗ ███╗   ███╗██╗
██║ ██╔╝██╔═══██╗██║ ██╔╝██╔═══██╗████╗ ████║██║
█████╔╝ ██║   ██║█████╔╝ ██║   ██║██╔████╔██║██║
██╔═██╗ ██║   ██║██╔═██╗ ██║   ██║██║╚██╔╝██║██║
██║  ██╗╚██████╔╝██║  ██╗╚██████╔╝██║ ╚═╝ ██║██║
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝
```

**Local AI Assistant — Built from Scratch with PyTorch**

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![NullStudio](https://img.shields.io/badge/NullStudio-Inc.-darkred?style=flat-square)

</div>

---

## What is Kokomi?

Kokomi is a **fully local, privacy-first AI assistant** built from scratch using PyTorch.
No cloud. No API keys. No data leaving your machine. Ever.

It runs entirely on your hardware — trained on your data, shaped by your preferences,
and owned completely by you. It is not trying to be ChatGPT. It is trying to be
a reliable local partner for developers who value ownership and privacy.

Built on a decoder-only Transformer architecture with BPE tokenization, SQLite memory,
a Rich terminal UI, and a modular Python codebase designed for easy extension.

**What it can do:**
- Chat with a trained local language model
- Remember projects, tasks, and notes via SQLite
- Scan your directory tree and find TODO comments
- Manage tasks and projects from the terminal
- Be retrained on any text data you provide
- Be extended with new tools via the tool registry

**What it is not:**
- A wrapper around OpenAI or any external API
- A pretrained model download
- Connected to the internet in any way

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/RDXFGXY1/kokomi.git
cd kokomi

# 2. Create a Python 3.12 virtual environment
py -3.12 -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/macOS

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Add your training data
# Drop .txt files into data/raw/
# See docs/TRAINING.md for guidance on what to put there

# 5. Train the model
python main.py --train

# 6. Start chatting
python main.py
```

> **Note:** A pre-trained `.pt` checkpoint is not included in this repository.
> You must train your own model. This is intentional — Kokomi is meant to learn
> from *your* data. See [docs/TRAINING.md](docs/TRAINING.md) for the full guide.

---

## Documentation

| File | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full system architecture, data flow, module map |
| [docs/TRAINING.md](docs/TRAINING.md) | How to prepare data, configure, and run training |
| [docs/TOKENIZER.md](docs/TOKENIZER.md) | CharTokenizer vs BPETokenizer, how encoding works |
| [docs/MEMORY.md](docs/MEMORY.md) | SQLite memory system, embeddings, recall |
| [docs/TOOLS.md](docs/TOOLS.md) | Tool registry, adding custom tools |
| [docs/CONFIG.md](docs/CONFIG.md) | All settings explained with recommended values |
| [docs/CLI.md](docs/CLI.md) | Terminal UI commands and usage |
| [docs/EXTENDING.md](docs/EXTENDING.md) | How to add new features, commands, and modules |

---

## Credits

Built by **RDXFGXY1** — Founder & CEO of [NullStudio Inc.](https://github.com/RDXFGXY1)

Kokomi is a solo project. Every module — the transformer, the tokenizer,
the memory system, the training pipeline — was written from scratch as a learning
exercise and a practical tool.

**Stack:** Python 3.12 · PyTorch 2.2 · SQLite · Rich · sentence-transformers

---

## Support

If Kokomi helped you learn something or saved you time, consider supporting the project:

- ⭐ Star the repository
- 🍕 [Buy me a coffee](https://buymeacoffee.com) *(link coming soon)*
- 🔁 Share it with someone who wants to build their own AI

Issues and pull requests are welcome.
