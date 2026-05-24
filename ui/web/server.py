"""
Web dashboard server (Flask).
Stage 4 implementation target — stub for now.
"""

from utils.logger import get_logger

logger = get_logger(__name__)


class WebServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 7860):
        self.host = host
        self.port = port

    def run(self):
        # TODO Stage 4: build Flask/FastAPI web dashboard
        logger.info(f"Web UI: http://{self.host}:{self.port}")
        raise NotImplementedError("Stage 4: Web dashboard not yet implemented. Use CLI for now.")
