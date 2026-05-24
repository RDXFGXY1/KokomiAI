"""
Code assistant — helps analyze tracebacks, explain errors, suggest fixes.
Stage 1: stub.
Stage 3: powered by Kokomi's model.
"""

import re
from utils.logger import get_logger

logger = get_logger(__name__)


class CodeHelper:
    def parse_traceback(self, tb: str) -> dict:
        """Extract key info from a Python traceback string."""
        lines  = tb.strip().splitlines()
        error  = lines[-1] if lines else "Unknown error"
        files  = re.findall(r'File "(.+?)", line (\d+)', tb)
        return {
            "error":     error,
            "locations": [{"file": f, "line": int(l)} for f, l in files],
            "raw":       tb,
        }

    def explain_error(self, tb: str) -> str:
        """Return a human-friendly explanation of an error. Stage 3: use model."""
        parsed = self.parse_traceback(tb)
        # TODO Stage 3: pass to Kokomi model for explanation
        return f"Error: {parsed['error']}\nLocations: {parsed['locations']}"

    def suggest_fix(self, tb: str, code_context: str = "") -> str:
        """Suggest a fix for the error. Stage 3: use model."""
        # TODO Stage 3: implement
        return "Stage 3: model-based fix suggestions not yet implemented."
