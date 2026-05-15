"""
memory/vector_memory.py — Production-grade vector memory with ChromaDB/FAISS support.
Provides persistent, scalable vector storage with similarity search.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger


@dataclass
class MemoryEpisode:
    """A single memory episode."""
    task: str
    answer: str
    path_used: str
    confidence: float
    success: bool
    reasoning_steps: list[str]
    embedding: np.ndarray
    tags: list[str]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task": self.task,
            "answer": self.answer,
            "path_used": self.path_used,
            "confidence": self.confidence,
            "success": self.success,
            "reasoning_steps": self.reasoning_steps,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEpisode":
        """Create from dictionary."""
        if isinstance(data.get("embedding"), list):
            data["embedding"] = np.array(data["embedding"])
        return cls(**data)


class VectorMemoryStore:
    """
    Production-grade vector memory store supporting multiple backends.
    """

    def __init__(
        self,
        persist_directory: str = "data/vector_db",
        backend: str = "chroma",
        collection_name: str = "episodes",
        embedding_dim: int = 384,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        logger.info(f"[VectorMemory] Initializing with backend: {backend}")

        if backend == "chroma":
            self._init_chroma()
        elif backend == "faiss":
            self._init_faiss()
        elif backend == "sqlite":
            self._init_sqlite()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _init_chroma(self):
        """Initialize ChromaDB backend."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            logger.error("[VectorMemory] chromadb not installed. Run: pip install chromadb")
            raise

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[VectorMemory] ChromaDB collection ready: {self.collection_name}")

    def _init_faiss(self):
        """Initialize FAISS backend."""
        try:
            import faiss
        except ImportError:
            logger.error("[VectorMemory] faiss-cpu not installed. Run: pip install faiss-cpu")
            raise

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.episodes: list[MemoryEpisode] = []
        self.faiss_path = self.persist_directory / "faiss_index.bin"
        self.episodes_path = self.persist_directory / "episodes.pkl"

        # Load existing index if available
        if self.faiss_path.exists():
            self.index = faiss.read_index(str(self.faiss_path))
            with open(self.episodes_path, "rb") as f:
                self.episodes = pickle.load(f)
            logger.info(f"[VectorMemory] Loaded FAISS index with {len(self.episodes)} episodes")

        logger.info("[VectorMemory] FAISS backend ready")

    def _init_sqlite(self):
        """Initialize SQLite backend."""
        import sqlite3

        self.db_path = self.persist_directory / "memory.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_sqlite_tables()
        logger.info("[VectorMemory] SQLite backend ready")

    def _create_sqlite_tables(self):
        """Create SQLite tables."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                answer TEXT NOT NULL,
                path_used TEXT NOT NULL,
                confidence REAL NOT NULL,
                success BOOLEAN NOT NULL,
                reasoning_steps TEXT NOT NULL,
                embedding BLOB NOT NULL,
                tags TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON episodes(timestamp DESC)
        """)
        self.conn.commit()

    def add_episode(self, episode: MemoryEpisode) -> str:
        """Add a memory episode."""
        if self.backend == "chroma":
            return self._add_chroma(episode)
        elif self.backend == "faiss":
            return self._add_faiss(episode)
        elif self.backend == "sqlite":
            return self._add_sqlite(episode)

    def _add_chroma(self, episode: MemoryEpisode) -> str:
        """Add episode to ChromaDB."""
        episode_id = f"{episode.timestamp}_{hash(episode.task)}"
        
        self.collection.add(
            ids=[episode_id],
            embeddings=[episode.embedding.tolist()],
            documents=[episode.task],
            metadatas=[{
                "answer": episode.answer,
                "path_used": episode.path_used,
                "confidence": episode.confidence,
                "success": episode.success,
                "tags": json.dumps(episode.tags),
                "timestamp": episode.timestamp,
            }],
        )
        return episode_id

    def _add_faiss(self, episode: MemoryEpisode) -> str:
        """Add episode to FAISS."""
        episode_id = f"{episode.timestamp}_{hash(episode.task)}"
        
        # Normalize embedding for cosine similarity
        embedding = episode.embedding.copy()
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        self.index.add(embedding.reshape(1, -1).astype("float32"))
        self.episodes.append(episode)
        
        # Persist
        faiss.write_index(self.index, str(self.faiss_path))
        with open(self.episodes_path, "wb") as f:
            pickle.dump(self.episodes, f)
        
        return episode_id

    def _add_sqlite(self, episode: MemoryEpisode) -> str:
        """Add episode to SQLite."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO episodes (
                task, answer, path_used, confidence, success,
                reasoning_steps, embedding, tags, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.task,
            episode.answer,
            episode.path_used,
            episode.confidence,
            episode.success,
            json.dumps(episode.reasoning_steps),
            pickle.dumps(episode.embedding),
            json.dumps(episode.tags),
            episode.timestamp,
            json.dumps(episode.metadata),
        ))
        self.conn.commit()
        return str(cursor.lastrowid)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[MemoryEpisode]:
        """Search for similar episodes."""
        if self.backend == "chroma":
            return self._search_chroma(query_embedding, top_k, filters)
        elif self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k, filters)
        elif self.backend == "sqlite":
            return self._search_sqlite(query_embedding, top_k, filters)

    def _search_chroma(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list[MemoryEpisode]:
        """Search ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters,
        )
        
        episodes = []
        for i, doc_id in enumerate(results["ids"][0]):
            episodes.append(MemoryEpisode(
                task=results["documents"][0][i],
                answer=results["metadatas"][0][i]["answer"],
                path_used=results["metadatas"][0][i]["path_used"],
                confidence=results["metadatas"][0][i]["confidence"],
                success=results["metadatas"][0][i]["success"],
                reasoning_steps=[],
                embedding=query_embedding,  # Placeholder
                tags=json.loads(results["metadatas"][0][i]["tags"]),
                timestamp=results["metadatas"][0][i]["timestamp"],
            ))
        
        return episodes

    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list[MemoryEpisode]:
        """Search FAISS."""
        # Normalize query
        query = query_embedding.copy()
        if np.linalg.norm(query) > 0:
            query = query / np.linalg.norm(query)
        
        similarities, indices = self.index.search(
            query.reshape(1, -1).astype("float32"),
            min(top_k, len(self.episodes)),
        )
        
        episodes = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < len(self.episodes):
                episode = self.episodes[idx]
                # Apply filters if provided
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key == "path_used" and episode.path_used != value:
                            match = False
                        elif key == "success" and episode.success != value:
                            match = False
                    if not match:
                        continue
                episodes.append(episode)
        
        return episodes[:top_k]

    def _search_sqlite(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list[MemoryEpisode]:
        """Search SQLite with cosine similarity."""
        cursor = self.conn.cursor()
        
        # Get all episodes (for simplicity - in production, use vector extension)
        query_str = "SELECT * FROM episodes"
        params = []
        
        if filters:
            conditions = []
            if "path_used" in filters:
                conditions.append("path_used = ?")
                params.append(filters["path_used"])
            if "success" in filters:
                conditions.append("success = ?")
                params.append(filters["success"])
            if conditions:
                query_str += " WHERE " + " AND ".join(conditions)
        
        query_str += " ORDER BY timestamp DESC LIMIT 100"
        cursor.execute(query_str, params)
        
        rows = cursor.fetchall()
        episodes = []
        
        for row in rows:
            embedding = pickle.loads(row[7])
            # Compute cosine similarity
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-8
            )
            
            episodes.append((sim, MemoryEpisode(
                task=row[1],
                answer=row[2],
                path_used=row[3],
                confidence=row[4],
                success=bool(row[5]),
                reasoning_steps=json.loads(row[6]),
                embedding=embedding,
                tags=json.loads(row[8]),
                timestamp=row[9],
                metadata=json.loads(row[10]),
            )))
        
        # Sort by similarity and return top_k
        episodes.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in episodes[:top_k]]

    def get_recent(self, n: int = 10) -> list[MemoryEpisode]:
        """Get recent episodes."""
        if self.backend == "chroma":
            results = self.collection.get(
                limit=n,
                order_by="timestamp",
            )
            # Convert to MemoryEpisode objects
            episodes = []
            for i, doc_id in enumerate(results["ids"]):
                episodes.append(MemoryEpisode(
                    task=results["documents"][i],
                    answer=results["metadatas"][i]["answer"],
                    path_used=results["metadatas"][i]["path_used"],
                    confidence=results["metadatas"][i]["confidence"],
                    success=results["metadatas"][i]["success"],
                    reasoning_steps=[],
                    embedding=np.zeros(self.embedding_dim),
                    tags=json.loads(results["metadatas"][i]["tags"]),
                    timestamp=results["metadatas"][i]["timestamp"],
                ))
            return episodes
        elif self.backend == "faiss":
            return self.episodes[-n:]
        elif self.backend == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (n,),
            )
            episodes = []
            for row in cursor.fetchall():
                episodes.append(MemoryEpisode(
                    task=row[1],
                    answer=row[2],
                    path_used=row[3],
                    confidence=row[4],
                    success=bool(row[5]),
                    reasoning_steps=json.loads(row[6]),
                    embedding=pickle.loads(row[7]),
                    tags=json.loads(row[8]),
                    timestamp=row[9],
                    metadata=json.loads(row[10]),
                ))
            return episodes

    def cleanup(self):
        """Cleanup resources."""
        if self.backend == "sqlite":
            self.conn.close()
        logger.info("[VectorMemory] Cleanup complete")
