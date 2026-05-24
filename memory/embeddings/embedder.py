"""
Semantic embeddings for memory search.
Uses sentence-transformers (PyTorch-backed) for similarity search.

Stage 1: stub (returns None).
Stage 2: implement real embeddings.
"""

import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    """
    Wraps sentence-transformers to embed text into vectors.
    Used for semantic memory retrieval: find memories similar to a query.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good quality

    def __init__(self):
        self._model = None
        self._ready = False

    def load(self):
        """Lazy load — only imports when actually needed."""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
            self._ready = True
            logger.info(f"Embedder loaded: {self.MODEL_NAME}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic search disabled.")

    def embed(self, text: str) -> np.ndarray | None:
        """Embed a single string → float vector."""
        if not self._ready:
            return None
        return self._model.encode(text, convert_to_numpy=True)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        if a is None or b is None:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def find_similar(self, query: str, candidates: list[dict], top_k: int = 5, threshold: float = 0.6) -> list[dict]:
        """
        Find top-k memories semantically similar to the query.
        candidates: list of dicts with 'content' key.
        """
        # TODO Stage 2: use real embeddings + cosine similarity
        return candidates[:top_k]
