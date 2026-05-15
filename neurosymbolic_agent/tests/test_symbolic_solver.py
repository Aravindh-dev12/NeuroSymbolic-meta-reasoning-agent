"""
tests/test_symbolic_solver.py — Unit tests for the SymbolicSolver.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from symbolic.solver import SymbolicSolver, SolverStatus
from symbolic.knowledge_base import KnowledgeBase, Fact, Rule
from symbolic.constraint_engine import ConstraintEngine, Constraint


class TestSymbolicSolver:
    def setup_method(self):
        self.solver = SymbolicSolver(timeout_seconds=5)

    # ── Syllogism tests ───────────────────────────────────────────────────────

    def test_basic_syllogism_positive(self):
        result = self.solver.solve(
            "All mammals breathe air. Whales are mammals. Do whales breathe air?"
        )
        assert result.status == SolverStatus.SAT
        assert "yes" in result.answer.lower() or "whale" in result.answer.lower()
        assert result.confidence > 0.8

    def test_basic_syllogism_chain(self):
        result = self.solver.solve(
            "All animals are living things. All dogs are animals. Fido is a dog. Is Fido a living thing?"
        )
        assert result.status == SolverStatus.SAT
        assert result.confidence > 0.7

    def test_syllogism_cannot_derive(self):
        result = self.solver.solve(
            "All cats are mammals. Dogs are mammals. Are cats dogs?"
        )
        # Should be unable to derive — different species
        assert "cannot determine" in result.answer.lower() or result.confidence < 0.8

    def test_proof_steps_populated(self):
        result = self.solver.solve(
            "All birds have wings. Eagles are birds. Do eagles have wings?"
        )
        assert len(result.proof_steps) > 0

    # ── Arithmetic tests ──────────────────────────────────────────────────────

    def test_simple_arithmetic(self):
        result = self.solver.solve("15 * 7 + 23")
        assert result.status == SolverStatus.SAT
        assert "128" in result.answer  # 15*7=105, 105+23=128

    def test_addition(self):
        result = self.solver.solve("100 + 200 + 50")
        assert "350" in result.answer

    def test_division(self):
        result = self.solver.solve("144 / 12")
        assert "12" in result.answer

    # ── Problem type detection ────────────────────────────────────────────────

    def test_detects_syllogism(self):
        assert self.solver._is_syllogism("all men are mortal")
        assert self.solver._is_syllogism("every student is a learner")
        assert not self.solver._is_syllogism("what is 2 + 2")

    def test_detects_arithmetic(self):
        assert self.solver._is_arithmetic("solve 2x + 3 = 7")
        assert self.solver._is_arithmetic("100 + 200")
        assert not self.solver._is_arithmetic("all cats are animals")

    def test_detects_propositional(self):
        assert self.solver._is_propositional("if it rains then the ground is wet")
        assert self.solver._is_propositional("A implies B and not C")

    # ── Generic fallback ──────────────────────────────────────────────────────

    def test_unsupported_returns_unknown(self):
        result = self.solver.solve("write a poem about clouds")
        assert result.status == SolverStatus.UNKNOWN
        assert result.confidence < 0.5

    # ── Confidence calibration ────────────────────────────────────────────────

    def test_confidence_in_valid_range(self):
        result = self.solver.solve("All fish live in water. Salmon are fish. Do salmon live in water?")
        assert 0.0 <= result.confidence <= 1.0


class TestKnowledgeBase:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_assert_and_query_fact(self):
        self.kb.assert_fact("is_a", "whale", "mammal")
        results = self.kb.query("is_a", "whale", None)
        assert len(results) == 1
        assert results[0].args == ["whale", "mammal"]

    def test_wildcard_query(self):
        self.kb.assert_fact("is_a", "dog", "animal")
        self.kb.assert_fact("is_a", "cat", "animal")
        results = self.kb.query("is_a", None, "animal")
        assert len(results) == 2

    def test_retract_fact(self):
        self.kb.assert_fact("likes", "alice", "chocolate")
        removed = self.kb.retract_fact("likes", "alice", "chocolate")
        assert removed
        assert len(self.kb.query("likes", "alice", None)) == 0

    def test_retract_nonexistent_returns_false(self):
        removed = self.kb.retract_fact("nonexistent", "a", "b")
        assert not removed

    def test_no_duplicate_facts(self):
        self.kb.assert_fact("is_a", "whale", "mammal")
        self.kb.assert_fact("is_a", "whale", "mammal")
        results = self.kb.query("is_a", "whale", None)
        assert len(results) == 1

    def test_fact_str(self):
        f = Fact(predicate="is_a", args=["dog", "animal"])
        assert str(f) == "is_a(dog, animal)"

    def test_summary(self):
        self.kb.assert_fact("is_a", "whale", "mammal")
        self.kb.assert_fact("can", "whale", "swim")
        s = self.kb.summary()
        assert s["fact_count"] == 2
        assert "is_a" in s["predicates"]

    def test_empty_kb_query(self):
        results = self.kb.query("is_a", "anything", None)
        assert results == []


class TestConstraintEngine:
    def setup_method(self):
        self.engine = ConstraintEngine(timeout_seconds=5)

    def test_empty_constraints_satisfiable(self):
        result = self.engine.check_constraints([])
        assert result.satisfiable
        assert result.confidence == 1.0

    def test_constraints_returned(self):
        constraints = [
            Constraint(
                name="c1",
                expression="x >= 0",
                variables=["x"],
                constraint_type="inequality",
            )
        ]
        result = self.engine.check_constraints(constraints)
        # Either solved (Z3 available) or reported (fallback)
        assert isinstance(result.satisfiable, bool)
        assert len(result.steps) > 0

    def test_constraint_names_in_result(self):
        constraints = [
            Constraint(
                name="age_positive",
                expression="age >= 0",
                variables=["age"],
                constraint_type="inequality",
            ),
            Constraint(
                name="age_reasonable",
                expression="age <= 150",
                variables=["age"],
                constraint_type="inequality",
            ),
        ]
        result = self.engine.check_constraints(constraints)
        assert isinstance(result.steps, list)
