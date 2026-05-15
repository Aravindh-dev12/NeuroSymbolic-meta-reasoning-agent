"""
memory/episodic_memory.py — Long-term episodic memory store.
Persists past reasoning episodes and supports similarity-based retrieval via FAISS.
Falls back to linear scan if FAISS is unavailable.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class Episode:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task: str = ""
    path_used: str = ""
    answer: str = ""
    confidence: float = 0.0
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    self_improvement_rounds: int = 0
    reasoning_steps: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove embedding — stored separately
        d.pop("embedding", None)
        return d


class EpisodicMemory:
    """
    Long-term episodic memory with:
    - JSON persistence
    - FAISS vector index for fast similarity search (with linear fallback)
    - Tagging and filtering
    """

    def __init__(
        self,
        persist_path: str = "logs/episodic_store.json",
        embedding_dim: int = 384,
    ):
        self.persist_path = persist_path
        self.embedding_dim = embedding_dim
        self._episodes: list[Episode] = []
        self._embeddings: list[np.ndarray] = []
        self._faiss_index = None

        os.makedirs(Path(persist_path).parent, exist_ok=True)
        self._load()
        self._init_faiss()
        logger.info(f"[EpisodicMemory] Loaded {len(self._episodes)} episodes")

    def _init_faiss(self) -> None:
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            if self._embeddings:
                arr = np.array(self._embeddings, dtype=np.float32)
                faiss.normalize_L2(arr)
                self._faiss_index.add(arr)
            logger.info("[EpisodicMemory] FAISS index ready")
        except ImportError:
            logger.warning("[EpisodicMemory] FAISS not available. Using linear scan.")

    def store_episode(
        self,
        episode: Episode,
        embedding: np.ndarray | None = None,
    ) -> Episode:
        self._episodes.append(episode)

        # Handle embedding
        if embedding is not None:
            self._embeddings.append(embedding.astype(np.float32))
        else:
            self._embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        # Update FAISS
        if self._faiss_index is not None and embedding is not None:
            import faiss
            arr = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(arr)
            self._faiss_index.add(arr)

        self._save()
        return episode

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> list[tuple[float, Episode]]:
        """Retrieve top-k similar past episodes."""
        if not self._episodes:
            return []

        if self._faiss_index is not None and len(self._embeddings) > 0:
            return self._faiss_retrieve(query_embedding, top_k, min_confidence)
        return self._linear_retrieve(query_embedding, top_k, min_confidence)

    def _faiss_retrieve(
        self,
        query: np.ndarray,
        top_k: int,
        min_confidence: float,
    ) -> list[tuple[float, Episode]]:
        import faiss
        arr = query.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(arr)
        k = min(top_k, len(self._episodes))
        scores, indices = self._faiss_index.search(arr, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and self._episodes[idx].confidence >= min_confidence:
                results.append((float(score), self._episodes[idx]))
        return results

    def _linear_retrieve(
        self,
        query: np.ndarray,
        top_k: int,
        min_confidence: float,
    ) -> list[tuple[float, Episode]]:
        scored = []
        for ep, emb in zip(self._episodes, self._embeddings):
            if ep.confidence < min_confidence:
                continue
            sim = self._cosine_sim(query, emb)
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def retrieve_by_tag(self, tag: str) -> list[Episode]:
        return [ep for ep in self._episodes if tag in ep.tags]

    def get_recent(self, n: int = 10) -> list[Episode]:
        return sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _save(self) -> None:
        data = [ep.to_dict() for ep in self._episodes]
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not Path(self.persist_path).exists():
            return
        try:
            with open(self.persist_path) as f:
                data = json.load(f)
            for d in data:
                self._episodes.append(Episode(**d))
                self._embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        except Exception as e:
            logger.warning(f"[EpisodicMemory] Load error: {e}")

    def __len__(self) -> int:
        return len(self._episodes)
