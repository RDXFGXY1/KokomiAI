"""
Task management interface.
Thin layer over MemoryManager for task-specific operations.
"""

from memory.memory_manager import MemoryManager
from utils.logger import get_logger

logger = get_logger(__name__)

PRIORITY_LABELS = {1: "LOW", 2: "NORMAL", 3: "HIGH", 4: "CRITICAL"}


class TaskManager:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def add(self, project: str, title: str, priority: int = 2) -> int:
        tid = self.memory.add_task(project, title, priority)
        logger.info(f"Task added [{PRIORITY_LABELS[priority]}] '{title}' → {project}")
        return tid

    def list_tasks(self, project: str) -> list[dict]:
        return self.memory.get_tasks(project)

    def complete(self, task_id: int):
        self.memory.store.complete_task(task_id)
        logger.info(f"Task {task_id} completed")

    def summary(self, project: str) -> str:
        tasks = self.list_tasks(project)
        if not tasks:
            return f"No open tasks for '{project}'."
        lines = [f"[{PRIORITY_LABELS.get(t['priority'], '?')}] {t['title']}" for t in tasks]
        return f"{project} — {len(tasks)} task(s):\n" + "\n".join(f"  {l}" for l in lines)
