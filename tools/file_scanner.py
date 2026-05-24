"""
File scanner — lets Kokomi read your project structure.
"""

import os
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

IGNORE = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}


class FileScanner:
    def __init__(self, root: str = "."):
        self.root = Path(root)

    def tree(self, max_depth: int = 3) -> str:
        """Return ASCII tree of the project directory."""
        lines = []
        self._walk(self.root, lines, depth=0, max_depth=max_depth)
        return "\n".join(lines)

    def _walk(self, path: Path, lines: list, depth: int, max_depth: int):
        if depth > max_depth:
            return
        indent = "  " * depth
        lines.append(f"{indent}{path.name}/")
        try:
            for item in sorted(path.iterdir()):
                if item.name in IGNORE:
                    continue
                if item.is_dir():
                    self._walk(item, lines, depth + 1, max_depth)
                else:
                    lines.append(f"{indent}  {item.name}")
        except PermissionError:
            pass

    def find_todos(self, extensions: list[str] = None) -> list[dict]:
        """Find TODO/FIXME/HACK comments across all source files."""
        exts = set(extensions or [".py", ".js", ".ts", ".cpp", ".h"])
        results = []
        for f in self.root.rglob("*"):
            if f.suffix not in exts or any(p in IGNORE for p in f.parts):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    upper = line.upper()
                    if any(k in upper for k in ("TODO", "FIXME", "HACK", "XXX")):
                        results.append({"file": str(f), "line": i, "text": line.strip()})
            except Exception:
                pass
        return results

    def find_todos_summary(self) -> str:
        todos = self.find_todos()
        if not todos:
            return "No TODOs found."
        return "\n".join(f"  {t['file']}:{t['line']} — {t['text']}" for t in todos[:20])
