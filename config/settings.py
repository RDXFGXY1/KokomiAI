"""
Central config. All settings live here.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    vocab_size:  int   = 4000   # BPE vocab — more than char, less than full BPE
    embed_dim:   int   = 256    # bigger than before
    num_heads:   int   = 8
    num_layers:  int   = 6
    ff_dim:      int   = 1024
    max_seq_len: int   = 256    # longer context
    dropout:     float = 0.15   # slightly more dropout for bigger data


@dataclass
class TrainingConfig:
    batch_size:       int   = 128      # good balance for 5-10MB data
    learning_rate:    float = 3e-4
    epochs:           int   = 10       # 30 epochs, ~20-25 min on RTX 4050
    grad_clip:        float = 1.0
    checkpoint_every: int   = 500
    device:           str   = "cuda"


@dataclass
class MemoryConfig:
    db_path:             str   = str(ROOT / "data" / "kokomi.db")
    max_memory_entries:  int   = 1000
    similarity_threshold: float = 0.75


@dataclass
class PersonalityConfig:
    name:          str  = "Kokomi"
    tone:          str  = "calm"
    supportive:    bool = True
    humor:         bool = True
    sarcasm_level: int  = 2


@dataclass
class Settings:
    model:          ModelConfig       = field(default_factory=ModelConfig)
    training:       TrainingConfig    = field(default_factory=TrainingConfig)
    memory:         MemoryConfig      = field(default_factory=MemoryConfig)
    personality:    PersonalityConfig = field(default_factory=PersonalityConfig)
    debug:          bool              = False
    data_dir:       str               = str(ROOT / "data")
    checkpoint_dir: str               = str(ROOT / "data" / "checkpoints")

    @classmethod
    def load(cls, path: str = None) -> "Settings":
        config_path = Path(path or ROOT / "config" / "kokomi.json")
        if config_path.exists():
            with open(config_path) as f:
                raw = json.load(f)
        return cls()

    def save(self, path: str = None):
        pass
