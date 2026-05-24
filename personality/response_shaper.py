"""
ResponseShaper wraps raw model output in Kokomi's voice.
Stage 1: rule-based shaping.
Stage 3: model-aware shaping.
"""

from .kokomi import KokomiPersonality


class ResponseShaper:
    def __init__(self, personality: KokomiPersonality = None):
        self.p = personality or KokomiPersonality()

    def shape(self, raw: str, context: str = "") -> str:
        """Apply personality layer to a raw response string."""
        # TODO Stage 3: smarter shaping with context awareness
        return raw.strip()

    def wrap_error(self, msg: str) -> str:
        if self.p.tone == "playful":
            return f"Bruh 💀 — {msg}"
        return f"Error: {msg}"

    def wrap_thinking(self) -> str:
        import random
        return random.choice(self.p.THINKING)
