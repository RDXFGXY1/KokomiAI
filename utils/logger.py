"""
Centralized logger. All modules use get_logger(__name__).
"""

import logging
import sys
import os
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE = "%H:%M:%S"

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # console handler with UTF-8 encoding
    ch = logging.StreamHandler(sys.stdout)
    # Force UTF-8 by reconfiguring the stream if possible
    try:
        if hasattr(ch.stream, "reconfigure"):
            ch.stream.reconfigure(encoding="utf-8", errors="replace")
    except:
        pass  # Fallback if reconfigure not available
    
    ch.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(ch)

    # file handler with explicit UTF-8 encoding
    fh = logging.FileHandler(LOG_DIR / "kokomi.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, _DATE))
    logger.addHandler(fh)

    return logger
