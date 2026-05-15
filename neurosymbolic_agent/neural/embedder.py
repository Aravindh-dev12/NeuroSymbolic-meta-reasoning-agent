"""
neural/embedder.py — Sentence embeddings using sentence-transformers.
Falls back to a simple bag-of-words TF-IDF embedding if the model isn't available.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
from loguru import logger


class Embedder:
    """
    Wraps sentence-transformers for dense embeddings.
    Falls back gracefully if the library is unavailable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._fallback = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"[Embedder] Loaded model: {self.model_name}")
        except Exception as e:
            logger.warning(f"[Embedder] Could not load sentence-transformers: {e}. Using fallback.")
            self._fallback = True

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a dense vector."""
        if self._fallback:
            return self._fallback_embed(text)
        return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        if self._fallback:
            return np.array([self._fallback_embed(t) for t in texts])
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _fallback_embed(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Deterministic pseudo-embedding based on character n-gram hashing.
        Not useful for real similarity but maintains interface compatibility.
        """
        vec = np.zeros(dim)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    @property
    def embedding_dim(self) -> int:
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return 384
