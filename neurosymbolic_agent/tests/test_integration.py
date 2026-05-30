"""
tests/test_integration.py — Integration tests for the full pipeline.
Tests the agent end-to-end with mocked LLM calls to avoid API costs.
"""
import sys
import os
import tempfile
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np

from symbolic.solver import SymbolicSolver, SolverStatus
from neural.embedder import Embedder
from neural.classifier import TaskClassifier, TaskType
from memory.memory_manager import MemoryManager
from constitutional.principles import PrinciplesRegistry
from constitutional.checker import ConstitutionalChecker
from constitutional.reward_hacking_detector import RewardHackingDetector
from planning.hierarchical_planner import HierarchicalPlanner, ExecutionPlan, Subtask
from planning.fallback_strategies import FallbackStrategyEngine, FallbackType
from utils.trace_recorder import TraceRecorder


# ─── Symbolic solver integration ─────────────────────────────────────────────

class TestSymbolicSolverIntegration:
    def setup_method(self):
        self.solver = SymbolicSolver()

    def test_three_step_syllogism(self):
        task = "All living things need energy. All animals are living things. Dogs are animals. Do dogs need energy?"
        result = self.solver.solve(task)
        assert "yes" in result.answer.lower() or "dog" in result.answer.lower()

    def test_arithmetic_precedence(self):
        result = self.solver.solve("2 + 3 * 4")
        # Python eval respects operator precedence: 3*4=12, 12+2=14
        assert "14" in result.answer

    def test_end_to_end_solver_returns_structured_result(self):
        result = self.solver.solve("All A are B. X is A. Is X B?")
        assert hasattr(result, "status")
        assert hasattr(result, "answer")
        assert hasattr(result, "confidence")
        assert hasattr(result, "proof_steps")
        assert 0.0 <= result.confidence <= 1.0


# ─── Memory pipeline integration ─────────────────────────────────────────────

class TestMemoryIntegration:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mm = MemoryManager(
            working_memory_capacity=10,
            episodic_memory_path=os.path.join(self.tmpdir, "ep.json"),
            embedding_dim=3,
            vector_db_type="sqlite",
            persist_directory=os.path.join(self.tmpdir, "vector_db"),
        )

    def test_store_then_retrieve_episode(self):
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.mm.store_episode(
            task="What is 2+2?",
            answer="4",
            path_used="symbolic",
            confidence=0.99,
            embedding=emb,
            tags=["arithmetic"],
        )
        ctx = self.mm.get_context(query_embedding=emb)
        assert len(ctx.similar_episodes) > 0
        score, ep = ctx.similar_episodes[0]
        assert ep.answer == "4"
        assert score > 0.8

    def test_working_memory_persists_corrections(self):
        self.mm.store_reasoning_step("step1", "Initial analysis")
        self.mm.update_from_critique(
            "step1",
            original="Initial analysis",
            corrected="Corrected analysis with full context",
            critique="Original was missing context",
        )
        item = self.mm.wm.retrieve("step1_corrected")
        assert item is not None
        assert item.metadata.get("is_correction") is True

    def test_full_episode_lifecycle(self):
        emb = np.random.rand(3).astype(np.float32)

        # Store multiple episodes
        for i in range(5):
            ep_emb = np.random.rand(3).astype(np.float32)
            self.mm.store_episode(
                task=f"Task {i}",
                answer=f"Answer {i}",
                path_used="neural" if i % 2 == 0 else "symbolic",
                confidence=0.7 + i * 0.02,
                embedding=ep_emb,
                tags=["test"],
            )

        assert len(self.mm.em) == 5
        recent = self.mm.em.get_recent(3)
        assert len(recent) == 3


# ─── Constitutional pipeline integration ─────────────────────────────────────

class TestConstitutionalIntegration:
    def setup_method(self):
        principles = [
            {
                "id": "CP-001",
                "name": "Factual Grounding",
                "description": "No overconfident claims",
                "check_type": "regex",
                "severity": "HIGH",
                "patterns": {"violations": ["I know for certain"]},
            },
            {
                "id": "CP-004",
                "name": "No Self-Modification",
                "description": "No blacklisted actions",
                "check_type": "action_blacklist",
                "severity": "CRITICAL",
                "blacklisted_actions": ["disable_checker"],
            },
            {
                "id": "CP-007",
                "name": "Bounded Recursion",
                "description": "Max rounds",
                "check_type": "iteration_bound",
                "severity": "HIGH",
                "max_rounds": 3,
            },
        ]
        registry = PrinciplesRegistry(principles=principles)
        self.checker = ConstitutionalChecker(registry=registry)
        self.hacking_detector = RewardHackingDetector()

    def test_clean_output_full_pipeline(self):
        output = "Based on evidence, I believe the answer is likely correct."
        check = self.checker.check(output=output, confidence=0.75)
        assert check.passed

    def test_violation_chain(self):
        # Violates CP-001 (regex) AND CP-004 (blacklist)
        check = self.checker.check(
            output="I know for certain this is right.",
            confidence=0.80,
            action="disable_checker completely",
        )
        violation_ids = {v.principle_id for v in check.violations}
        assert "CP-001" in violation_ids
        assert "CP-004" in violation_ids
        assert not check.passed
        assert check.has_critical_violations

    def test_iteration_bound_stops_at_limit(self):
        for round_n in range(1, 4):
            check = self.checker.check(
                output="Still working...",
                confidence=0.65,
                improvement_round=round_n,
            )
            violation_ids = {v.principle_id for v in check.violations}
            assert "CP-007" not in violation_ids  # Rounds 1-3 are fine

        # Round 4 exceeds limit
        check = self.checker.check(
            output="Still working...",
            confidence=0.65,
            improvement_round=4,
        )
        violation_ids = {v.principle_id for v in check.violations}
        assert "CP-007" in violation_ids

    def test_reward_hacking_and_constitutional_combined(self):
        # Perfect confidence after improvement round = reward hacking
        principles_with_rh = [
            {
                "id": "CP-005",
                "name": "No Reward Hacking",
                "description": "No gaming self-evaluation",
                "check_type": "reward_hacking",
                "severity": "CRITICAL",
                "heuristics": ["self_assigned_perfect_score"],
            }
        ]
        registry = PrinciplesRegistry(principles=principles_with_rh)
        checker = ConstitutionalChecker(registry=registry)

        check = checker.check(
            output="Perfect answer.",
            confidence=1.0,
            improvement_round=2,
        )
        violation_ids = {v.principle_id for v in check.violations}
        assert "CP-005" in violation_ids


# ─── Planning integration ─────────────────────────────────────────────────────

class TestPlanningIntegration:
    def test_fallback_strategy_timeout(self):
        engine = FallbackStrategyEngine(max_retries=2)
        decision = engine.decide(
            task="Prove theorem X",
            failure_reason="Z3 solver timeout",
            current_path="symbolic",
            confidence=0.40,
            attempt=0,
        )
        assert decision.fallback_type == FallbackType.PATH_SWITCH
        assert "neural" in decision.reason.lower() or "switching" in decision.reason.lower()

    def test_fallback_strategy_json_error(self):
        engine = FallbackStrategyEngine()
        decision = engine.decide(
            task="Some task",
            failure_reason="JSON parse error in response",
            current_path="neural",
            confidence=0.70,
            attempt=0,
        )
        assert decision.fallback_type == FallbackType.RETRY
        assert "json" in decision.modified_task.lower() or "format" in decision.modified_task.lower()

    def test_fallback_strategy_max_retries_defer(self):
        engine = FallbackStrategyEngine(max_retries=2)
        # Exhaust retries
        for _ in range(3):
            decision = engine.decide(
                task="Hard task",
                failure_reason="unknown error",
                current_path="hybrid",
                confidence=0.50,
                attempt=0,
            )
        assert decision.fallback_type == FallbackType.DEFER

    def test_simplify_task(self):
        engine = FallbackStrategyEngine()
        long_task = "This is a very long task. It has multiple sentences. And more details."
        simplified = engine._simplify_task(long_task)
        assert len(simplified) < len(long_task)
        assert simplified.endswith(".")


# ─── Trace recorder integration ──────────────────────────────────────────────

class TestTraceRecorder:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.recorder = TraceRecorder(
            trace_file=os.path.join(self.tmpdir, "traces.jsonl")
        )

    def test_full_trace_lifecycle(self):
        trace = self.recorder.start_trace("abc123", "What is 2+2?")
        assert trace.trace_id == "abc123"
        assert trace.task == "What is 2+2?"

        step = self.recorder.add_step(
            "routing",
            input_summary="What is 2+2?",
            output_summary="path=symbolic, confidence=0.95",
            path="symbolic",
        )
        assert step.step_type == "routing"

        self.recorder.add_step("reasoning", input_summary="solve", output_summary="4")
        self.recorder.add_step("self_improvement", rounds=0)

        completed = self.recorder.complete_trace(
            final_answer="4",
            final_confidence=0.99,
            path_used="symbolic",
            self_improvement_rounds=0,
            success=True,
        )

        assert completed.success
        assert completed.final_answer == "4"
        assert len(completed.steps) == 3

    def test_trace_persisted_to_jsonl(self):
        self.recorder.start_trace("xyz789", "Test task")
        self.recorder.add_step("routing", input_summary="input", output_summary="output")
        self.recorder.complete_trace("Answer", 0.8, "neural", success=True)

        trace_path = os.path.join(self.tmpdir, "traces.jsonl")
        assert os.path.exists(trace_path)

        with open(trace_path) as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["trace_id"] == "xyz789"
        assert data["final_answer"] == "Answer"

    def test_multiple_traces(self):
        for i in range(3):
            self.recorder.start_trace(f"t{i}", f"Task {i}")
            self.recorder.complete_trace(f"Answer {i}", 0.8, "neural", success=True)

        trace_path = os.path.join(self.tmpdir, "traces.jsonl")
        with open(trace_path) as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_no_active_trace_raises(self):
        with pytest.raises(RuntimeError):
            self.recorder.add_step("routing")

    def test_step_id_format(self):
        self.recorder.start_trace("id1", "task")
        step = self.recorder.add_step("routing")
        assert step.step_id.startswith("id1_")
        self.recorder.complete_trace("ans", 0.8, "neural")
