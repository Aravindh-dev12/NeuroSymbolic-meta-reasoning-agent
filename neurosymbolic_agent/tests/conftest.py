"""
tests/conftest.py — Shared pytest fixtures.
"""
import sys
import os
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Add project root to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_embedding():
    """Return a deterministic sample embedding vector."""
    rng = np.random.default_rng(42)
    v = rng.random(384).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


@pytest.fixture
def small_embedding():
    """Return a 3-dimensional embedding for fast tests."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def sample_principles():
    """Return a minimal set of constitutional principles for testing."""
    return [
        {
            "id": "CP-001",
            "name": "Factual Grounding",
            "description": "No overconfident claims.",
            "check_type": "regex",
            "severity": "HIGH",
            "patterns": {"violations": ["I know for certain"]},
        },
        {
            "id": "CP-004",
            "name": "No Self-Modification",
            "description": "No blacklisted actions.",
            "check_type": "action_blacklist",
            "severity": "CRITICAL",
            "blacklisted_actions": ["disable_checker"],
        },
        {
            "id": "CP-007",
            "name": "Bounded Recursion",
            "description": "Max improvement rounds.",
            "check_type": "iteration_bound",
            "severity": "HIGH",
            "max_rounds": 3,
        },
    ]


@pytest.fixture
def memory_manager(tmp_dir):
    """Provide a MemoryManager backed by a temp directory."""
    from memory.memory_manager import MemoryManager
    return MemoryManager(
        working_memory_capacity=10,
        episodic_memory_path=os.path.join(tmp_dir, "ep.json"),
        embedding_dim=3,
    )


@pytest.fixture
def symbolic_solver():
    """Provide a SymbolicSolver instance."""
    from symbolic.solver import SymbolicSolver
    return SymbolicSolver(timeout_seconds=5)


@pytest.fixture
def constitutional_checker(sample_principles):
    """Provide a ConstitutionalChecker with sample principles."""
    from constitutional.principles import PrinciplesRegistry
    from constitutional.checker import ConstitutionalChecker
    registry = PrinciplesRegistry(principles=sample_principles)
    return ConstitutionalChecker(registry=registry)
