# Extending Kokomi

This guide explains how to add new features without breaking existing ones.
Follow the module dependency rules in ARCHITECTURE.md.

---

## Adding a New CLI Command

Open `ui/cli/terminal.py`. Find the `_handle_command` method.
Add a new case to the match block:

```python
case "/mycommand":
    if len(parts) < 2:
        return "usage: /mycommand <arg>"
    result = do_something(parts[1])
    console.print(_msg_success(result))
```

Add it to `HELP_TEXT` at the top of the file:
```python
("/mycommand <arg>", "description of what it does"),
```

---

## Adding a New Memory Field

Open `memory/store/sqlite_store.py`. Add a new table to the `SCHEMA` string:

```python
SCHEMA = """
...existing tables...

CREATE TABLE IF NOT EXISTS notes (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    title   TEXT NOT NULL,
    content TEXT,
    created TEXT NOT NULL
);
"""
```

Add methods to `SQLiteStore`:
```python
def add_note(self, title: str, content: str) -> int:
    cur = self.conn.execute(
        "INSERT INTO notes (title, content, created) VALUES (?, ?, ?)",
        (title, content, datetime.now().isoformat())
    )
    self.conn.commit()
    return cur.lastrowid
```

Expose it through `MemoryManager`:
```python
def add_note(self, title: str, content: str) -> int:
    return self.store.add_note(title, content)
```

---

## Adding a New Personality Tone

Open `personality/kokomi.py`. Add to the `GREETINGS` dict:

```python
GREETINGS = {
    "calm":     ["Hey. What are we working on?", ...],
    "playful":  ["Yoo, I'm here! Let's go.", ...],
    "sharp":    ["Ready.", "Talk."],
    "mentor":   ["Welcome back. What are you learning today?", ...],  # new
}
```

Set it in `config/settings.py`:
```python
tone: str = "mentor"
```

---

## Swapping the Model Brain

The inference engine is isolated in `brain/inference.py`.
To swap in a different model (e.g., Ollama), replace `KokomiInference.generate()`:

```python
def generate(self, user_input: str, **kwargs) -> str:
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": f"RDXFGXY1: {user_input}\nKokomi:"}
    )
    return response.json().get("response", "...").strip()
```

Everything else — memory, tasks, CLI, tools — stays exactly the same.

---

## Adding the Web Dashboard (Stage 4)

Open `ui/web/server.py` and implement the Flask server:

```python
from flask import Flask, jsonify, request
from memory.memory_manager import MemoryManager
from brain.inference import KokomiInference

app = Flask(__name__)
memory = MemoryManager()
inference = KokomiInference()
inference.load(inference.find_latest_checkpoint())

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    response = inference.generate(data["message"])
    memory.log("user", data["message"])
    memory.log("assistant", response)
    return jsonify({"response": response})

class WebServer:
    def run(self):
        app.run(host="127.0.0.1", port=7860)
```
