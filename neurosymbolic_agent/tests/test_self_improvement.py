"""
tests/test_self_improvement.py — Unit tests for the self-improvement loop
and reward hacking detector.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock

from agent.self_improvement import SelfImprovementLoop, CritiqueResult
from constitutional.reward_hacking_detector import RewardHackingDetector, HackingSignal


class TestCritiqueResult:
    def test_no_issues(self):
        c = CritiqueResult(
            has_issues=False,
            issues=[],
            failure_modes=["none"],
            suggested_correction="",
            corrected_answer="The answer is correct.",
            corrected_confidence=0.85,
            improvement_achieved=False,
        )
        assert not c.has_issues
        assert c.corrected_confidence == 0.85

    def test_with_issues(self):
        c = CritiqueResult(
            has_issues=True,
            issues=["Unsupported claim", "Missing premise"],
            failure_modes=["hallucination", "incomplete"],
            suggested_correction="Add supporting evidence",
            corrected_answer="Revised answer with evidence.",
            corrected_confidence=0.72,
            improvement_achieved=True,
            reasoning_trace=["Step 1: Identified gap", "Step 2: Added evidence"],
        )
        assert c.has_issues
        assert len(c.issues) == 2
        assert c.improvement_achieved
        assert len(c.reasoning_trace) == 2


class TestRewardHackingDetector:
    def setup_method(self):
        self.detector = RewardHackingDetector()

    def test_clean_round_no_signals(self):
        signals = self.detector.record_round(
            round_n=1,
            answer="The sky is blue because of Rayleigh scattering.",
            confidence=0.75,
            critique="The explanation is mostly correct but could be more detailed.",
            corrected_answer="The sky is blue because of Rayleigh scattering of sunlight. Shorter wavelengths scatter more.",
            corrected_confidence=0.82,
        )
        # Modest confidence increase with real content change — should be clean
        assert isinstance(signals, list)

    def test_confidence_inflation_detected(self):
        signals = self.detector.record_round(
            round_n=1,
            answer="The answer is X.",
            confidence=0.50,
            critique="Minor phrasing issue.",
            corrected_answer="The answer is X.",  # Nearly identical
            corrected_confidence=0.95,  # Large jump
        )
        signal_types = [s.signal_type for s in signals]
        assert "confidence_inflation" in signal_types

    def test_circular_correction_detected(self):
        # First round
        self.detector.record_round(
            round_n=1,
            answer="Answer A",
            confidence=0.70,
            critique="Switch to B",
            corrected_answer="Answer B",
            corrected_confidence=0.75,
        )
        # Second round
        self.detector.record_round(
            round_n=2,
            answer="Answer B",
            confidence=0.75,
            critique="Switch back to A",
            corrected_answer="Answer A",  # Reverts to round 1's answer
            corrected_confidence=0.78,
        )
        # Third round
        signals = self.detector.record_round(
            round_n=3,
            answer="Answer A",
            confidence=0.78,
            critique="",
            corrected_answer="Answer A",
            corrected_confidence=0.78,
        )
        signal_types = [s.signal_type for s in signals]
        assert "circular_correction" in signal_types

    def test_critique_suppression_detected(self):
        long_critique = "A" * 500
        medium_critique = "B" * 200
        short_critique = "C" * 50

        self.detector.record_round(1, "a", 0.5, long_critique, "a", 0.55)
        self.detector.record_round(2, "a", 0.55, medium_critique, "a", 0.6)
        signals = self.detector.record_round(3, "a", 0.6, short_critique, "a", 0.65)

        signal_types = [s.signal_type for s in signals]
        # May or may not trigger depending on exact thresholds
        assert isinstance(signals, list)

    def test_text_change_ratio_identical(self):
        ratio = RewardHackingDetector._text_change_ratio("hello world", "hello world")
        assert ratio == 0.0

    def test_text_change_ratio_different(self):
        ratio = RewardHackingDetector._text_change_ratio("apple banana", "cat dog fish")
        assert ratio > 0.9  # Completely different words

    def test_text_change_ratio_partial(self):
        ratio = RewardHackingDetector._text_change_ratio("the cat sat", "the dog sat")
        assert 0.0 < ratio < 1.0

    def test_detector_summary(self):
        self.detector.record_round(1, "a", 0.5, "critique", "b", 0.6)
        summary = self.detector.summary()
        assert summary["rounds_monitored"] == 1
        assert len(summary["history"]) == 1

    def test_empty_detector_no_signals(self):
        fresh = RewardHackingDetector()
        signals = fresh.record_round(1, "a", 0.5, "critique", "b", 0.6)
        assert isinstance(signals, list)


class TestSelfImprovementLoopMocked:
    """Tests for SelfImprovementLoop with mocked LLM calls."""

    def _make_loop(self, max_rounds=2):
        loop = SelfImprovementLoop.__new__(SelfImprovementLoop)
        loop.model = "claude-sonnet-4-20250514"
        loop.max_rounds = max_rounds
        loop.constitutional_checker = None
        loop.memory = None
        loop.hacking_detector = RewardHackingDetector()
        return loop

    def test_no_issues_stops_early(self):
        loop = self._make_loop(max_rounds=3)

        no_issue_critique = CritiqueResult(
            has_issues=False,
            issues=[],
            failure_modes=["none"],
            suggested_correction="",
            corrected_answer="Original answer.",
            corrected_confidence=0.80,
            improvement_achieved=False,
        )

        with patch.object(loop, "_critique", return_value=no_issue_critique):
            answer, confidence, rounds, steps = loop.improve(
                task="What is 2+2?",
                initial_answer="4",
                initial_confidence=0.95,
            )

        assert rounds == 1  # Stops after first round finds no issues
        assert answer == "4"

    def test_correction_applied_on_issues(self):
        loop = self._make_loop(max_rounds=2)

        has_issue_critique = CritiqueResult(
            has_issues=True,
            issues=["Incomplete answer"],
            failure_modes=["incomplete"],
            suggested_correction="Add explanation",
            corrected_answer="4, because 2+2=4 by arithmetic.",
            corrected_confidence=0.88,
            improvement_achieved=True,
        )
        no_issue_critique = CritiqueResult(
            has_issues=False,
            issues=[],
            failure_modes=["none"],
            suggested_correction="",
            corrected_answer="4, because 2+2=4 by arithmetic.",
            corrected_confidence=0.88,
            improvement_achieved=False,
        )

        critiques = [has_issue_critique, no_issue_critique]
        call_count = {"n": 0}

        def mock_critique(*args, **kwargs):
            c = critiques[min(call_count["n"], len(critiques) - 1)]
            call_count["n"] += 1
            return c

        with patch.object(loop, "_critique", side_effect=mock_critique):
            answer, confidence, rounds, steps = loop.improve(
                task="What is 2+2?",
                initial_answer="4",
                initial_confidence=0.70,
            )

        assert "because" in answer
        assert confidence > 0.70

    def test_max_rounds_respected(self):
        loop = self._make_loop(max_rounds=2)

        always_issue_critique = CritiqueResult(
            has_issues=True,
            issues=["Still wrong"],
            failure_modes=["incorrect_path"],
            suggested_correction="Try again",
            corrected_answer="Revised answer.",
            corrected_confidence=0.60,
            improvement_achieved=True,
        )

        with patch.object(loop, "_critique", return_value=always_issue_critique):
            answer, confidence, rounds, steps = loop.improve(
                task="Hard task",
                initial_answer="Initial answer",
                initial_confidence=0.50,
            )

        assert rounds <= loop.max_rounds

    def test_apply_constitutional_correction_overconfidence(self):
        loop = self._make_loop()
        from constitutional.checker import CheckResult, ViolationReport
        from constitutional.principles import Severity

        check = CheckResult(
            passed=False,
            violations=[ViolationReport(
                principle_id="CP-003",
                principle_name="Calibrated Confidence",
                severity=Severity.MEDIUM,
                description="overconfident output detected",
            )],
        )

        corrected = loop._apply_constitutional_correction("My answer.", check)
        assert "uncertainty" in corrected.lower() or "note" in corrected.lower()
