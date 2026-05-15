"""
memory/memory_manager.py — Production-grade unified memory interface.
Coordinates working memory, episodic memory, and vector storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from loguru import logger

from .episodic_memory import Episode, EpisodicMemory
from .vector_memory import MemoryEpisode, VectorMemoryStore
from .working_memory import WorkingMemory, WorkingMemoryItem


@dataclass
class MemoryContext:
    """Compiled memory context for a reasoning step."""
    working_memory_items: list[WorkingMemoryItem]
    similar_episodes: list[tuple[float, Episode]]
    context_summary: str


class MemoryManager:
    """
    Production-grade memory manager coordinating working memory, episodic memory, and vector storage.
    Provides a unified interface for storing and retrieving information with persistence.
    """

    def __init__(
        self,
        working_memory_capacity: int = 20,
        episodic_memory_path: str = "logs/episodic_store.json",
        retrieval_top_k: int = 5,
        similarity_threshold: float = 0.6,
        embedding_dim: int = 384,
        vector_db_type: str = "chroma",
        persist_directory: str = "data/vector_db",
        enable_compression: bool = True,
    ):
        self.wm = WorkingMemory(capacity=working_memory_capacity)
        self.em = EpisodicMemory(
            persist_path=episodic_memory_path,
            embedding_dim=embedding_dim,
        )
        self.vector_store = VectorMemoryStore(
            persist_directory=persist_directory,
            backend=vector_db_type,
            embedding_dim=embedding_dim,
        )
        self.top_k = retrieval_top_k
        self.sim_threshold = similarity_threshold
        self.enable_compression = enable_compression
        logger.info(f"[MemoryManager] Initialized with vector backend: {vector_db_type}")

    def store_reasoning_step(
        self,
        key: str,
        content: str,
        embedding: np.ndarray | None = None,
        **metadata,
    ) -> WorkingMemoryItem:
        """Store a reasoning step in working memory."""
        return self.wm.store(key, content, embedding=embedding, **metadata)

    def store_episode(
        self,
        task: str,
        answer: str,
        path_used: str,
        confidence: float,
        success: bool = True,
        reasoning_steps: list[str] | None = None,
        embedding: np.ndarray | None = None,
        tags: list[str] | None = None,
        **metadata,
    ) -> Episode:
        """Store a completed reasoning episode in both episodic and vector memory."""
        # Store in episodic memory (JSON-based)
        episode = Episode(
            task=task,
            answer=answer,
            path_used=path_used,
            confidence=confidence,
            success=success,
            reasoning_steps=reasoning_steps or [],
            tags=tags or [],
            metadata=metadata,
        )
        self.em.store_episode(episode, embedding=embedding)
        
        # Store in vector memory for similarity search
        if embedding is not None:
            vector_episode = MemoryEpisode(
                task=task,
                answer=answer,
                path_used=path_used,
                confidence=confidence,
                success=success,
                reasoning_steps=reasoning_steps or [],
                embedding=embedding,
                tags=tags or [],
                metadata=metadata,
            )
            self.vector_store.add_episode(vector_episode)
        
        return episode

    def get_context(
        self,
        query_embedding: np.ndarray | None,
        recent_n: int = 3,
    ) -> MemoryContext:
        """
        Assemble relevant memory context for a new task.
        Returns recent working memory items + similar past episodes from vector store.
        """
        # Recent working memory
        recent_wm = self.wm.get_recent(n=recent_n)

        # Similar episodes from vector store
        similar_eps = []
        if query_embedding is not None:
            vector_episodes = self.vector_store.search(
                query_embedding,
                top_k=self.top_k,
            )
            # Convert to tuple format for compatibility
            similar_eps = [
                (0.8, Episode(  # Placeholder score, will be improved
                    task=ep.task,
                    answer=ep.answer,
                    path_used=ep.path_used,
                    confidence=ep.confidence,
                    success=ep.success,
                    reasoning_steps=ep.reasoning_steps,
                    tags=ep.tags,
                    metadata=ep.metadata,
                ))
                for ep in vector_episodes
            ]

        # Build context summary
        summary_parts = []
        if recent_wm:
            summary_parts.append(
                "Recent working memory: " +
                "; ".join(item.content[:100] for item in recent_wm)
            )
        if similar_eps:
            ep_summaries = [
                f"[{score:.2f}] {ep.task[:60]}... → {ep.answer[:60]}"
                for score, ep in similar_eps[:3]
            ]
            summary_parts.append("Similar past episodes:\n" + "\n".join(ep_summaries))

        return MemoryContext(
            working_memory_items=recent_wm,
            similar_episodes=similar_eps,
            context_summary="\n\n".join(summary_parts) if summary_parts else "No relevant memory.",
        )

    def update_from_critique(
        self,
        key: str,
        original: str,
        corrected: str,
        critique: str,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Update working memory with corrected reasoning trace."""
        self.wm.store(
            key=f"{key}_corrected",
            content=corrected,
            embedding=embedding,
            original=original,
            critique=critique,
            is_correction=True,
        )
        logger.debug(f"[MemoryManager] Stored correction for: {key}")

    def clear_working_memory(self) -> None:
        self.wm.clear()

    def summary(self) -> dict[str, Any]:
        return {
            "working_memory": self.wm.snapshot(),
            "episodic_count": len(self.em),
            "vector_backend": self.vector_store.backend,
        }

    def cleanup(self):
        """Cleanup all memory resources."""
        self.vector_store.cleanup()
        logger.info("[MemoryManager] Cleanup complete")
