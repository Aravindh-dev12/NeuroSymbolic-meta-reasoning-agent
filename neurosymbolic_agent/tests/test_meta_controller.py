"""
tests/test_meta_controller.py — Unit tests for MetaController routing logic.
Uses a mocked Embedder to avoid loading the full sentence-transformers model.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import json

from neural.classifier import TaskClassifier, TaskType, ClassificationResult
from agent.meta_controller import MetaController, RoutingDecision


class MockEmbedder:
    """Lightweight mock that returns random embeddings without loading a model."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**31)
        return np.random.rand(self.dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])


class TestTaskClassifier:
    def setup_method(self):
        self.embedder = MockEmbedder()
        self.classifier = TaskClassifier(self.embedder)

    def test_symbolic_keywords_detected(self):
        result = self.classifier.classify("Prove that all mammals breathe if whales are mammals")
        assert result.task_type in (TaskType.SYMBOLIC, TaskType.HYBRID)

    def test_neural_keywords_detected(self):
        result = self.classifier.classify("Classify the sentiment of this text")
        assert result.task_type in (TaskType.NEURAL, TaskType.HYBRID)

    def test_hybrid_keywords_detected(self):
        result = self.classifier.classify("Explain why the strategy works and how to plan it")
        assert result.task_type in (TaskType.HYBRID, TaskType.NEURAL)

    def test_confidence_in_range(self):
        result = self.classifier.classify("Some random task")
        assert 0.0 <= result.confidence <= 1.0

    def test_unknown_falls_back(self):
        result = self.classifier.classify("xyz 123 ???")
        assert result.task_type in TaskType.__members__.values()


class TestRoutingDecision:
    def test_routing_decision_fields(self):
        d = RoutingDecision(
            path="symbolic",
            confidence=0.9,
            task_type="syllogism",
            reasoning="Logical structure detected",
        )
        assert d.path == "symbolic"
        assert d.confidence == 0.9
        assert not d.needs_planning
        assert d.facts_extracted == []

    def test_routing_decision_with_planning(self):
        d = RoutingDecision(
            path="hybrid",
            confidence=0.75,
            task_type="planning",
            reasoning="Complex multi-step task",
            needs_planning=True,
            subtask_hints=["Analyse input", "Generate plan", "Validate"],
        )
        assert d.needs_planning
        assert len(d.subtask_hints) == 3


class TestMetaControllerFallback:
    def setup_method(self):
        self.embedder = MockEmbedder()
        self.classifier = TaskClassifier(self.embedder)

    def test_fallback_routing_symbolic(self):
        mc = MetaController.__new__(MetaController)
        mc.confidence_threshold = 0.75
        mc._embedder = self.embedder
        mc._classifier = self.classifier

        result = mc._fallback_routing(
            "prove theorem X",
            ClassificationResult(
                task_type=TaskType.SYMBOLIC,
                confidence=0.85,
                scores={"symbolic": 0.85},
                reasoning="keyword match",
            )
        )
        assert result.path == "symbolic"
        assert result.confidence <= 0.85 * 0.8 + 0.01  # with tolerance

    def test_is_confident(self):
        mc = MetaController.__new__(MetaController)
        mc.confidence_threshold = 0.75

        high = RoutingDecision(path="neural", confidence=0.9, task_type="x", reasoning="")
        low = RoutingDecision(path="neural", confidence=0.5, task_type="x", reasoning="")

        assert mc.is_confident(high)
        assert not mc.is_confident(low)
