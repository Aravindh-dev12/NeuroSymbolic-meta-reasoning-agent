"""
memory/working_memory.py — Short-term working memory with capacity limits.
Implements a capacity-bounded buffer with LRU eviction and relevance scoring.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger


@dataclass
class WorkingMemoryItem:
    key: str
    content: str
    embedding: np.ndarray | None
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """
    Short-term working memory with:
    - Fixed capacity with LRU eviction
    - Vector-similarity-based retrieval
    - Recency-weighted access scoring
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self._store: OrderedDict[str, WorkingMemoryItem] = OrderedDict()
        logger.info(f"[WorkingMemory] Initialised with capacity={capacity}")

    def store(
        self,
        key: str,
        content: str,
        embedding: np.ndarray | None = None,
        **metadata,
    ) -> WorkingMemoryItem:
        """Store an item, evicting LRU if at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            item.content = content
            item.embedding = embedding
            item.timestamp = time.time()
            item.metadata.update(metadata)
            return item

        if len(self._store) >= self.capacity:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug(f"[WorkingMemory] Evicted: {evicted_key}")

        item = WorkingMemoryItem(
            key=key,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )
        self._store[key] = item
        self._store.move_to_end(key)
        return item

    def retrieve(self, key: str) -> WorkingMemoryItem | None:
        if key in self._store:
            self._store.move_to_end(key)
            item = self._store[key]
            item.access_count += 1
            return item
        return None

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[WorkingMemoryItem]:
        """Retrieve items by cosine similarity to query embedding."""
        if query_embedding is None:
            return list(self._store.values())[-top_k:]

        scored: list[tuple[float, WorkingMemoryItem]] = []
        for item in self._store.values():
            if item.embedding is not None:
                sim = self._cosine_similarity(query_embedding, item.embedding)
                if sim >= threshold:
                    scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:top_k]]

        for item in results:
            item.access_count += 1

        return results

    def get_recent(self, n: int = 5) -> list[WorkingMemoryItem]:
        """Return n most recently stored items."""
        items = list(self._store.values())
        return items[-n:]

    def clear(self) -> None:
        self._store.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._store.values()]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"WorkingMemory(capacity={self.capacity}, used={len(self._store)})"
