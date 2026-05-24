"""
Project overview and progress tracking.
"""

from memory.memory_manager import MemoryManager


class ProjectTracker:
    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def new_project(self, name: str, description: str = "") -> int:
        return self.memory.add_project(name, description)

    def list_projects(self) -> list[dict]:
        return self.memory.get_projects()

    def overview(self) -> str:
        projects = self.list_projects()
        if not projects:
            return "No active projects."
        lines = [f"  • {p['name']}" + (f": {p['description']}" if p.get("description") else "") for p in projects]
        return "Active projects:\n" + "\n".join(lines)
