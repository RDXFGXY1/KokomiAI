# Chat Interface Guide

Your Kokomi AI assistant now has a full CLI chat interface! Here's how to use it:

## Quick Start

```bash
# Start the chat
python main.py

# Or explicitly start CLI mode
python main.py
```

## Chat Interface Features

### Natural Language Chat
Just type normally and Kokomi will respond:
```
> Tell me about yourself
KOKOMI: [AI response from trained model]

> What's the weather like?
KOKOMI: [AI response]
```

### Commands

| Command | Usage | Description |
|---------|-------|-------------|
| `/help` | `/help` | Show all commands |
| `/status` | `/status` | Check if model is loaded |
| `/clear` | `/clear` | Clear the screen |
| **Projects** | | |
| `/projects` | `/projects` | List all projects |
| `/project` | `/project <name>` | Create new project |
| **Tasks** | | |
| `/tasks` | `/tasks <project>` | List project tasks |
| `/task` | `/task <project> <title>` | Add task to project |
| `/done` | `/done <task_id>` | Mark task complete |
| **Memory** | | |
| `/remember` | `/remember <text>` | Store in memory |
| `/recall` | `/recall <query>` | Search memories |
| **Utilities** | | |
| `/scan` | `/scan` | Browse directory structure |
| `/todos` | `/todos` | Find TODO comments in code |
| `/quit` | `/quit` or `/exit` | Exit chat |

## Examples

**Chat with Kokomi:**
```
> How can I improve my code?
KOKOMI: Consider adding type hints and breaking large functions into smaller ones...

> Remember: I prefer Python over Java
KOKOMI: Memory stored.

> What did I tell you about my preferences?
KOKOMI: You prefer Python over Java...
```

**Manage Projects:**
```
> /project MyWebApp
✓ Project 'MyWebApp' added.

> /task MyWebApp "Add authentication"
✓ Task added to MyWebApp.

> /tasks MyWebApp
┌─ Tasks ────────────────┐
│ ID │ Priority │ Task   │
├────┼──────────┼────────┤
│ 1  │ NORMAL   │ Add... │
└────┴──────────┴────────┘

> /done 1
✓ Task 1 done.
```

**Utilities:**
```
> /scan
Directory structure...

> /todos
TODO comments found in your code...

> /status
Model: LOADED
```

## Model Status

- **Model Loaded**: Kokomi will use the trained transformer model to generate responses
- **Model Not Loaded**: Kokomi will echo your input (demo mode)

The chat automatically looks for the latest checkpoint in `data/checkpoints/`. 
If no checkpoint is found, run training first:
```bash
python main.py --train
```

## Features Coming Soon

- Context awareness from project files
- Integration with code analysis tools
- Custom personality modes
- Multi-turn conversation memory
- Export chat history

---

**Tips:**
- Keep prompts clear and concise for better responses
- Use `/remember` to teach Kokomi about your preferences
- Use `/recall` to ask Kokomi what it remembers about you
- Check `/status` if responses seem off
