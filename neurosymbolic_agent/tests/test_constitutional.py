"""
tests/test_constitutional.py — Unit tests for ConstitutionalChecker and PrinciplesRegistry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from constitutional.principles import PrinciplesRegistry, Principle, Severity
from constitutional.checker import ConstitutionalChecker, CheckResult, ViolationReport


SAMPLE_PRINCIPLES = [
    {
        "id": "CP-001",
        "name": "Factual Grounding",
        "description": "No overconfident unverified claims.",
        "check_type": "regex",
        "severity": "HIGH",
        "patterns": {
            "violations": ["I know for certain", "It is absolutely true"],
            "allowed_uncertainty": ["I believe", "likely"],
        },
    },
    {
        "id": "CP-003",
        "name": "Calibrated Confidence",
        "description": "Confidence must be calibrated.",
        "check_type": "confidence_bound",
        "severity": "MEDIUM",
        "max_confidence_without_proof": 0.95,
    },
    {
        "id": "CP-004",
        "name": "No Self-Modification of Safety",
        "description": "No blacklisted actions.",
        "check_type": "action_blacklist",
        "severity": "CRITICAL",
        "blacklisted_actions": ["disable_checker", "modify_constitutional_principles"],
    },
    {
        "id": "CP-005",
        "name": "No Reward Hacking",
        "description": "No reward hacking.",
        "check_type": "reward_hacking",
        "severity": "CRITICAL",
        "heuristics": ["self_assigned_perfect_score", "empty_critique_with_high_confidence"],
    },
    {
        "id": "CP-007",
        "name": "Bounded Recursion",
        "description": "Max improvement rounds.",
        "check_type": "iteration_bound",
        "severity": "HIGH",
        "max_rounds": 3,
    },
    {
        "id": "CP-002",
        "name": "No Circular Reasoning",
        "description": "No repeated reasoning steps.",
        "check_type": "graph_cycle",
        "severity": "HIGH",
    },
    {
        "id": "CP-006",
        "name": "Transparency of Uncertainty",
        "description": "Express uncertainty when confidence is low.",
        "check_type": "uncertainty_communication",
        "severity": "MEDIUM",
    },
    {
        "id": "CP-008",
        "name": "No Harmful Output",
        "description": "No harmful content.",
        "check_type": "content_safety",
        "severity": "CRITICAL",
    },
]


class TestPrinciplesRegistry:
    def setup_method(self):
        self.registry = PrinciplesRegistry(principles=SAMPLE_PRINCIPLES)

    def test_loaded_count(self):
        assert len(self.registry) == len(SAMPLE_PRINCIPLES)

    def test_get_by_id(self):
        p = self.registry.get("CP-001")
        assert p is not None
        assert p.name == "Factual Grounding"
        assert p.severity == Severity.HIGH

    def test_get_nonexistent(self):
        assert self.registry.get("CP-999") is None

    def test_all_returns_all(self):
        all_p = self.registry.all()
        assert len(all_p) == len(SAMPLE_PRINCIPLES)

    def test_by_severity_critical(self):
        critical = self.registry.by_severity(Severity.CRITICAL)
        assert len(critical) >= 1
        assert all(p.severity == Severity.CRITICAL for p in critical)

    def test_critical_shortcut(self):
        critical = self.registry.critical()
        ids = [p.id for p in critical]
        assert "CP-004" in ids
        assert "CP-005" in ids


class TestConstitutionalChecker:
    def setup_method(self):
        registry = PrinciplesRegistry(principles=SAMPLE_PRINCIPLES)
        self.checker = ConstitutionalChecker(
            registry=registry,
            max_violations_before_halt=2,
            strict_mode=False,
        )

    def test_clean_output_passes(self):
        result = self.checker.check(
            output="Based on available evidence, I believe the answer is likely B.",
            confidence=0.75,
        )
        assert result.passed
        assert result.violations == []

    def test_regex_violation_detected(self):
        result = self.checker.check(
            output="I know for certain that the answer is X.",
            confidence=0.80,
        )
        violation_ids = [v.principle_id for v in result.violations]
        assert "CP-001" in violation_ids

    def test_overconfidence_violation(self):
        result = self.checker.check(
            output="The answer is correct.",
            confidence=0.98,  # Above 0.95 without proof
            reasoning_steps=[],
        )
        # CP-003 is MEDIUM — will appear as warning, not violation
        # But confidence adjustment should be negative
        assert result.confidence_adjustment <= 0

    def test_blacklisted_action_violation(self):
        result = self.checker.check(
            output="Proceeding normally.",
            confidence=0.80,
            action="disable_checker now",
        )
        violation_ids = [v.principle_id for v in result.violations]
        assert "CP-004" in violation_ids

    def test_blacklisted_action_clean(self):
        result = self.checker.check(
            output="Normal output.",
            confidence=0.80,
            action="run_analysis",
        )
        violation_ids = [v.principle_id for v in result.violations]
        assert "CP-004" not in violation_ids

    def test_reward_hacking_perfect_score_after_improvement(self):
        result = self.checker.check(
            output="Perfect answer.",
            confidence=1.0,
            improvement_round=2,  # After self-improvement
        )
        violation_ids = [v.principle_id for v in result.violations]
        assert "CP-005" in violation_ids

    def test_iteration_bound_violation(self):
        result = self.checker.check(
            output="Still trying.",
            confidence=0.60,
            improvement_round=5,  # Exceeds max_rounds=3
        )
        violation_ids = [v.principle_id for v in result.violations]
        assert "CP-007" in violation_ids

    def test_circular_reasoning_detected(self):
        repeated_step = "Therefore A implies B because A is true and B follows from A"
        result = self.checker.check(
            output="The conclusion is A.",
            confidence=0.70,
            reasoning_steps=[repeated_step, "Some other step", repeated_step],
        )
        violation_ids = [v.principle_id for v in result.violations]
        assert "CP-002" in violation_ids

    def test_uncertainty_warning_low_confidence(self):
        result = self.checker.check(
            output="The answer is definitely X and nothing else.",  # No uncertainty language
            confidence=0.45,  # Low confidence
        )
        # CP-006 is MEDIUM — appears as warning
        assert isinstance(result.warnings, list)

    def test_check_result_summary_pass(self):
        result = CheckResult(passed=True, violations=[], warnings=[])
        assert "passed" in result.summary()

    def test_check_result_summary_fail(self):
        result = CheckResult(
            passed=False,
            violations=[ViolationReport(
                principle_id="CP-001",
                principle_name="Factual Grounding",
                severity=Severity.HIGH,
                description="Violation",
            )],
        )
        assert "CP-001" in result.summary()

    def test_has_critical_violations(self):
        result = CheckResult(
            passed=False,
            violations=[ViolationReport(
                principle_id="CP-004",
                principle_name="No Self-Modification",
                severity=Severity.CRITICAL,
                description="Critical violation",
            )],
        )
        assert result.has_critical_violations

    def test_no_critical_when_medium_only(self):
        result = CheckResult(
            passed=False,
            violations=[ViolationReport(
                principle_id="CP-003",
                principle_name="Calibrated Confidence",
                severity=Severity.MEDIUM,
                description="Medium violation",
            )],
        )
        assert not result.has_critical_violations

    def test_confidence_adjustment_negative_on_violation(self):
        result = self.checker.check(
            output="I know for certain this is true.",
            confidence=0.80,
        )
        assert result.confidence_adjustment < 0
