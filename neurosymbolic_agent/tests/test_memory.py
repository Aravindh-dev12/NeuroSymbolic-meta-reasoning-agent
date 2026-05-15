"""
tests/test_memory.py — Unit tests for working memory, episodic memory, and memory manager.
"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from memory.working_memory import WorkingMemory, WorkingMemoryItem
from memory.episodic_memory import EpisodicMemory, Episode
from memory.memory_manager import MemoryManager, MemoryContext


class TestWorkingMemory:
    def setup_method(self):
        self.wm = WorkingMemory(capacity=5)

    def test_store_and_retrieve(self):
        self.wm.store("key1", "Hello world")
        item = self.wm.retrieve("key1")
        assert item is not None
        assert item.content == "Hello world"

    def test_retrieve_nonexistent(self):
        assert self.wm.retrieve("nonexistent") is None

    def test_capacity_eviction(self):
        for i in range(6):
            self.wm.store(f"key{i}", f"content {i}")
        assert len(self.wm) == 5  # Capacity enforced
        # First item should be evicted
        assert self.wm.retrieve("key0") is None

    def test_update_existing(self):
        self.wm.store("key1", "original")
        self.wm.store("key1", "updated")
        item = self.wm.retrieve("key1")
        assert item.content == "updated"
        assert len(self.wm) == 1  # No duplicate

    def test_access_count_increments(self):
        self.wm.store("key1", "content")
        self.wm.retrieve("key1")
        self.wm.retrieve("key1")
        item = self.wm.retrieve("key1")
        assert item.access_count >= 2

    def test_get_recent(self):
        for i in range(5):
            self.wm.store(f"key{i}", f"content {i}")
        recent = self.wm.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].key == "key4"

    def test_clear(self):
        self.wm.store("key1", "content")
        self.wm.clear()
        assert len(self.wm) == 0

    def test_snapshot(self):
        self.wm.store("key1", "content1")
        self.wm.store("key2", "content2")
        snap = self.wm.snapshot()
        assert len(snap) == 2
        assert all("key" in s for s in snap)

    def test_retrieve_similar_with_embeddings(self):
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        emb3 = np.array([0.9, 0.1, 0.0])

        self.wm.store("k1", "first", embedding=emb1)
        self.wm.store("k2", "second", embedding=emb2)
        self.wm.store("k3", "third", embedding=emb3)

        query = np.array([1.0, 0.0, 0.0])
        results = self.wm.retrieve_similar(query, top_k=2, threshold=0.5)
        assert len(results) >= 1
        keys = [r.key for r in results]
        assert "k1" in keys  # Most similar to query

    def test_cosine_similarity_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = WorkingMemory._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        sim = WorkingMemory._cosine_similarity(v1, v2)
        assert abs(sim) < 1e-6

    def test_repr(self):
        self.wm.store("k", "v")
        r = repr(self.wm)
        assert "WorkingMemory" in r
        assert "capacity=5" in r


class TestEpisodicMemory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.em = EpisodicMemory(
            persist_path=os.path.join(self.tmpdir, "episodes.json"),
            embedding_dim=3,
        )

    def test_store_and_count(self):
        ep = Episode(task="Test task", answer="Test answer", path_used="neural", confidence=0.8)
        self.em.store_episode(ep)
        assert len(self.em) == 1

    def test_retrieve_recent(self):
        for i in range(5):
            ep = Episode(task=f"Task {i}", answer=f"Answer {i}", path_used="neural", confidence=0.7)
            self.em.store_episode(ep)
        recent = self.em.get_recent(3)
        assert len(recent) == 3

    def test_retrieve_by_tag(self):
        ep1 = Episode(task="Logic task", answer="A", path_used="symbolic", confidence=0.9, tags=["logic"])
        ep2 = Episode(task="NLP task", answer="B", path_used="neural", confidence=0.8, tags=["nlp"])
        self.em.store_episode(ep1)
        self.em.store_episode(ep2)

        logic_eps = self.em.retrieve_by_tag("logic")
        assert len(logic_eps) == 1
        assert logic_eps[0].task == "Logic task"

    def test_retrieve_similar_with_embeddings(self):
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        ep1 = Episode(task="Task A", answer="A", path_used="neural", confidence=0.8)
        ep2 = Episode(task="Task B", answer="B", path_used="neural", confidence=0.8)
        self.em.store_episode(ep1, embedding=emb1)
        self.em.store_episode(ep2, embedding=emb2)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.em.retrieve_similar(query, top_k=1)
        assert len(results) >= 1

    def test_episode_persistence(self):
        path = os.path.join(self.tmpdir, "episodes2.json")
        em1 = EpisodicMemory(persist_path=path, embedding_dim=3)
        ep = Episode(task="Persistent task", answer="Ans", path_used="symbolic", confidence=0.9)
        em1.store_episode(ep)

        em2 = EpisodicMemory(persist_path=path, embedding_dim=3)
        assert len(em2) == 1
        assert em2._episodes[0].task == "Persistent task"

    def test_cosine_sim_zero_vectors(self):
        a = np.zeros(3)
        b = np.ones(3)
        sim = EpisodicMemory._cosine_sim(a, b)
        assert sim == 0.0

    def test_retrieve_similar_empty_memory(self):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = self.em.retrieve_similar(query, top_k=5)
        assert results == []


class TestMemoryManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mm = MemoryManager(
            working_memory_capacity=10,
            episodic_memory_path=os.path.join(self.tmpdir, "ep.json"),
            retrieval_top_k=3,
            similarity_threshold=0.5,
            embedding_dim=3,
        )

    def test_store_reasoning_step(self):
        item = self.mm.store_reasoning_step("step1", "Analysing the problem...")
        assert item.content == "Analysing the problem..."

    def test_store_episode(self):
        ep = self.mm.store_episode(
            task="Test task",
            answer="Test answer",
            path_used="hybrid",
            confidence=0.82,
        )
        assert ep.task == "Test task"
        assert ep.path_used == "hybrid"
        assert len(self.mm.em) == 1

    def test_get_context_no_embedding(self):
        self.mm.store_reasoning_step("step1", "context item")
        ctx = self.mm.get_context(query_embedding=None, recent_n=3)
        assert isinstance(ctx, MemoryContext)
        assert isinstance(ctx.context_summary, str)

    def test_get_context_with_embedding(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.mm.store_episode(
            task="A task",
            answer="An answer",
            path_used="neural",
            confidence=0.8,
            embedding=emb,
        )
        ctx = self.mm.get_context(query_embedding=emb)
        assert isinstance(ctx.similar_episodes, list)

    def test_update_from_critique(self):
        self.mm.update_from_critique(
            key="task_1",
            original="Wrong answer",
            corrected="Correct answer",
            critique="The original missed the key point.",
        )
        item = self.mm.wm.retrieve("task_1_corrected")
        assert item is not None
        assert item.content == "Correct answer"

    def test_clear_working_memory(self):
        self.mm.store_reasoning_step("k1", "content")
        self.mm.clear_working_memory()
        assert len(self.mm.wm) == 0

    def test_summary(self):
        self.mm.store_reasoning_step("k1", "step")
        summary = self.mm.summary()
        assert "working_memory" in summary
        assert "episodic_count" in summary
