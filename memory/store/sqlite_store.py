"""
SQLite persistence layer.
All memory, projects, tasks, and chat history live here.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    content   TEXT    NOT NULL,
    tags      TEXT    DEFAULT '[]',
    embedding BLOB,
    created   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT,
    status      TEXT    DEFAULT 'active',
    created     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER REFERENCES projects(id),
    title      TEXT    NOT NULL,
    notes      TEXT,
    priority   INTEGER DEFAULT 2,
    done       INTEGER DEFAULT 0,
    created    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_history (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    ts      TEXT NOT NULL
);
"""


class SQLiteStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info(f"SQLite store ready: {db_path}")

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ── Memories ────────────────────────────────────────────
    def add_memory(self, content: str, tags: list[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO memories (content, tags, created) VALUES (?, ?, ?)",
            (content, json.dumps(tags or []), datetime.now().isoformat())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_memories(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM memories ORDER BY created DESC").fetchall()
        return [dict(r) for r in rows]

    # ── Projects ─────────────────────────────────────────────
    def add_project(self, name: str, description: str = "") -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO projects (name, description, created) VALUES (?, ?, ?)",
            (name, description, datetime.now().isoformat())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_projects(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM projects WHERE status='active'").fetchall()
        return [dict(r) for r in rows]

    # ── Tasks ────────────────────────────────────────────────
    def add_task(self, project_id: int, title: str, priority: int = 2) -> int:
        cur = self.conn.execute(
            "INSERT INTO tasks (project_id, title, priority, created) VALUES (?, ?, ?, ?)",
            (project_id, title, priority, datetime.now().isoformat())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_tasks(self, project_id: int, done: bool = False) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE project_id=? AND done=? ORDER BY priority DESC",
            (project_id, int(done))
        ).fetchall()
        return [dict(r) for r in rows]

    def complete_task(self, task_id: int):
        self.conn.execute("UPDATE tasks SET done=1 WHERE id=?", (task_id,))
        self.conn.commit()

    # ── Chat History ─────────────────────────────────────────
    def log_message(self, role: str, content: str):
        self.conn.execute(
            "INSERT INTO chat_history (role, content, ts) VALUES (?, ?, ?)",
            (role, content, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_recent_history(self, n: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chat_history ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
        return list(reversed([dict(r) for r in rows]))

    def close(self):
        self.conn.close()
