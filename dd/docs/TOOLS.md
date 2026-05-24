# Tools System

Kokomi has a plugin-style tool registry. Any callable can be registered
as a tool and invoked by name.

---

## Built-in Tools

### FileScanner
Scans your project directory.

```python
from tools.file_scanner import FileScanner

scanner = FileScanner(".")
print(scanner.tree(max_depth=3))       # ASCII directory tree
print(scanner.find_todos_summary())    # all TODO/FIXME comments
todos = scanner.find_todos()           # list of dicts: file, line, text
```

### CodeHelper
Parses Python tracebacks.

```python
from tools.code_helper import CodeHelper

helper = CodeHelper()
parsed = helper.parse_traceback(traceback_string)
# returns: {"error": "...", "locations": [...], "raw": "..."}
```

### ToolRegistry
Central registry for all tools.

```python
from tools.tool_registry import ToolRegistry

registry = ToolRegistry()
registry.register("my_tool", my_function, description="does something")
result = registry.run("my_tool", arg1, arg2)
print(registry.help())   # list all registered tools
```

---

## Adding a Custom Tool

1. Create a function that takes arguments and returns a string:

```python
def word_count(path: str) -> str:
    from pathlib import Path
    text = Path(path).read_text(errors="ignore")
    words = len(text.split())
    lines = len(text.splitlines())
    return f"{path}: {words} words, {lines} lines"
```

2. Register it in `tools/tool_registry.py` or in `ui/cli/terminal.py`:

```python
self.tool_registry = ToolRegistry()
self.tool_registry.register("word_count", word_count, "count words in a file")
```

3. Add a CLI command in `terminal.py` to expose it:

```python
case "/wc":
    if len(parts) < 2:
        return "usage: /wc <file>"
    result = self.tool_registry.run("word_count", parts[1])
    console.print(_msg_system(result))
```

---

## Planned Tools (Stage 3+)

- `web_search` — search the web via a local proxy
- `run_python` — execute a Python snippet in a sandbox
- `git_status` — show current git state of a repo
- `read_file` — read and summarize a file
- `write_file` — create or append to a file
