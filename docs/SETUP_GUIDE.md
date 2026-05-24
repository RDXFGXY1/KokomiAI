# Kokomi AI Chat Setup - Complete Guide

## What's New

I've created a complete CLI chat interface for you to talk with your AI! Here's what was added:

### New Files Created

1. **`brain/inference.py`** - AI model inference engine
   - Loads trained checkpoints automatically
   - Generates text responses using your trained transformer
   - Handles tokenization and decoding

2. **`demo_chat.py`** - Simple demo script
   - Quick way to launch the chat interface
   - Use: `python demo_chat.py`

3. **`CHAT_GUIDE.md`** - Detailed command reference
   - Full list of commands
   - Usage examples
   - Tips and tricks

### Modified Files

1. **`ui/cli/terminal.py`** - Enhanced terminal interface
   - Integrated AI inference
   - Added `/status` command to check model
   - Added `/clear` command
   - Now generates real AI responses (not just echoes)

2. **`utils/logger.py`** - Fixed Unicode encoding
   - Fixes the `→` character encoding error you saw
   - Now properly handles UTF-8 on Windows

## Quick Start

### Option 1: Run Main Command (Recommended)
```bash
python main.py
```
This automatically starts the CLI chat interface.

### Option 2: Run Demo Script
```bash
python demo_chat.py
```
Alternative way to launch the chat.

## First Chat Session

When you start, you'll see:
```
  K O K O M I
  local ai assistant
  NullStudio Inc.

KOKOMI: Back again. What's on your mind?
ℹ  Type /help for commands.

> 
```

Now you can:
- **Chat naturally**: Just type and Kokomi will respond
- **Use commands**: Type `/help` to see all commands
- **Check status**: Type `/status` to see if model is loaded

## Chat Examples

### Natural Chat
```
> Tell me about yourself
KOKOMI: [AI-generated response based on training]

> How do I use Python for web development?
KOKOMI: [AI-generated advice]
```

### Project Management
```
> /project MyApp
✓ Project 'MyApp' added.

> /task MyApp "Add login feature"
✓ Task added to MyApp.

> /tasks MyApp
[Shows tasks in a nice table]
```

### Memory System
```
> /remember I like Python and clean code
✓ Memory stored.

> /recall What do I like?
ℹ  I like Python and clean code
```

## How It Works

1. **Startup**: Terminal loads your latest trained checkpoint automatically
2. **Model Loading**: If a checkpoint exists, it's loaded into memory
3. **Your Input**: You type something
4. **Processing**: 
   - If it starts with `/`, it's a command (project management, memory, etc.)
   - Otherwise, it's sent to the AI model
5. **Response**: AI generates a response using the trained transformer
6. **Logging**: Both input and response are saved to conversation history

## Model Integration

The chat uses your trained transformer model:
- **Model Location**: `data/checkpoints/` (automatically finds latest)
- **Tokenizer**: Character-level tokenizer (matches your training)
- **Generation**: Uses top-k sampling with temperature for natural responses
- **Memory**: Stores conversations in SQLite database

## Troubleshooting

### "Model not loaded" message
- Make sure you've run training: `python main.py --train`
- Check that `data/checkpoints/` contains `.pt` files
- Use `/status` command to see details

### Unicode encoding errors (Fixed!)
The `→` character error is now fixed in the logger. If you still see encoding issues:
- The logger now forces UTF-8 encoding on Windows
- Errors are replaced with safe characters instead of crashing

### Chat not responding
- Check `/status` to see if model loaded
- Check the latest checkpoint file exists
- Try clearing and restarting

## Advanced Usage

### Programmatic API
```python
from brain.inference import KokomiInference
from config.settings import Settings

# Initialize
inference = KokomiInference(settings=Settings.load())
latest_ckpt = inference.find_latest_checkpoint()
inference.load(latest_ckpt)

# Generate
response = inference.generate("Hello, tell me about yourself", max_tokens=100)
print(response)
```

### Custom Settings
Edit `config/settings.py` to change:
- Model size (embedding dimension, layers, attention heads)
- Generation parameters (temperature, top-k)
- Memory database location
- Personality tone

## Feature Walkthrough

### Commands You Can Use

| Command | What It Does |
|---------|-------------|
| `/help` | Show all commands |
| `/status` | Check if AI model is loaded |
| `/clear` | Clear the screen |
| `/projects` | List all your projects |
| `/project <name>` | Create a new project |
| `/tasks <project>` | View tasks in a project |
| `/task <project> <title>` | Add a task |
| `/done <id>` | Mark task complete |
| `/remember <text>` | Save something to memory |
| `/recall <query>` | Search your memories |
| `/scan` | Browse directory structure |
| `/todos` | Find TODO comments in code |
| `/quit` | Exit the chat |

### Just Type Naturally
Anything that doesn't start with `/` goes to the AI:
```
> What's the best way to structure a Python project?
> How do I learn machine learning?
> Write me a joke
```

## Next Steps

1. **Start chatting**: `python main.py`
2. **Build context**: Use `/remember` to teach Kokomi about yourself
3. **Manage work**: Use project/task commands to organize
4. **Iterate**: Chat naturally while working on projects

## Additional Notes

- **Conversation History**: Saved automatically in memory
- **Checkpoints**: Latest checkpoint auto-loaded on startup
- **Configuration**: All settings in `config/settings.py`
- **Logging**: All logs saved to `data/kokomi.log`
- **Database**: Memories stored in `data/kokomi.db`

---

**You're all set! Start chatting with:** `python main.py` 🚀

Questions? Check CHAT_GUIDE.md for detailed examples.
