"""
Kokomi's personality definition.
Shapes how responses are framed, toned, and structured.
"""

from dataclasses import dataclass
from config.settings import PersonalityConfig


@dataclass
class KokomiPersonality:
    name:          str   = "Kokomi"
    tone:          str   = "calm"       # calm | playful | sharp
    supportive:    bool  = True
    humor:         bool  = True
    sarcasm_level: int   = 2

    # Greeting variants by tone
    GREETINGS = {
        "calm":    ["Hey. What are we working on?", "Back again. What's on your mind?"],
        "playful": ["Yoo, I'm here! Let's go 🔥", "Kokomi online. What do you need?"],
        "sharp":   ["Ready.", "Talk."],
    }

    THINKING = ["Hmm...", "Let me check...", "One sec..."]

    def greeting(self) -> str:
        import random
        return random.choice(self.GREETINGS.get(self.tone, self.GREETINGS["calm"]))

    @classmethod
    def from_config(cls, cfg: PersonalityConfig) -> "KokomiPersonality":
        return cls(
            name          = cfg.name,
            tone          = cfg.tone,
            supportive    = cfg.supportive,
            humor         = cfg.humor,
            sarcasm_level = cfg.sarcasm_level,
        )
