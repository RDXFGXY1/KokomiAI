"""
High-level memory interface.
All other modules use this, never the store/embedder directly.
"""

from memory.store.sqlite_store import SQLiteStore
from memory.embeddings.embedder import Embedder
from config.settings import Settings
from utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Unified interface for:
        - storing / retrieving memories
        - managing projects and tasks
        - chat history
        - semantic search
    """

    def __init__(self, settings: Settings = None):
        s = settings or Settings.load()
        self.store   = SQLiteStore(s.memory.db_path)
        self.embedder = Embedder()
        self.embedder.load()

    # ── Memory ───────────────────────────────────────────────
    def remember(self, content: str, tags: list[str] = None) -> int:
        """Store a new memory."""
        mid = self.store.add_memory(content, tags)
        logger.debug(f"Memory stored: {content[:60]}…")
        return mid

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve memories relevant to query (semantic if available)."""
        memories = self.store.get_all_memories()
        return self.embedder.find_similar(query, memories, top_k)

    # ── Projects ─────────────────────────────────────────────
    def add_project(self, name: str, description: str = "") -> int:
        return self.store.add_project(name, description)

    def get_projects(self) -> list[dict]:
        return self.store.get_projects()

    # ── Tasks ────────────────────────────────────────────────
    def add_task(self, project_name: str, title: str, priority: int = 2) -> int:
        projects = {p["name"]: p["id"] for p in self.get_projects()}
        if project_name not in projects:
            pid = self.add_project(project_name)
        else:
            pid = projects[project_name]
        return self.store.add_task(pid, title, priority)

    def get_tasks(self, project_name: str) -> list[dict]:
        projects = {p["name"]: p["id"] for p in self.get_projects()}
        if project_name not in projects:
            return []
        return self.store.get_tasks(projects[project_name])

    # ── Chat history ─────────────────────────────────────────
    def log(self, role: str, content: str):
        self.store.log_message(role, content)

    def get_context(self, n: int = 10) -> list[dict]:
        return self.store.get_recent_history(n)
