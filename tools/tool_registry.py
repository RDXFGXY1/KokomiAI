"""
Tool registry — central catalog of all callable tools.
Each tool is a callable that takes a string argument and returns a string.

Usage:
    registry = ToolRegistry()
    result = registry.run("scan_files", "/home/RDXFGXY1/projects")
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, callable] = {}

    def register(self, name: str, fn: callable, description: str = ""):
        self._tools[name] = {"fn": fn, "desc": description}
        logger.debug(f"Tool registered: {name}")

    def run(self, name: str, *args, **kwargs) -> str:
        if name not in self._tools:
            return f"Unknown tool: '{name}'. Available: {self.list()}"
        try:
            return self._tools[name]["fn"](*args, **kwargs)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return f"Tool error: {e}"

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def help(self) -> str:
        if not self._tools:
            return "No tools registered."
        return "\n".join(f"  {n}: {v['desc']}" for n, v in self._tools.items())
