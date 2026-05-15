"""
constitutional/checker.py — Constitutional AI output checker.
Validates agent outputs against constitutional principles and flags violations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .principles import Principle, PrinciplesRegistry, Severity


@dataclass
class ViolationReport:
    principle_id: str
    principle_name: str
    severity: Severity
    description: str
    evidence: str = ""


@dataclass
class CheckResult:
    passed: bool
    violations: list[ViolationReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence_adjustment: float = 0.0  # negative = reduce confidence

    @property
    def has_critical_violations(self) -> bool:
        return any(v.severity == Severity.CRITICAL for v in self.violations)

    def summary(self) -> str:
        if self.passed:
            return "✓ Constitutional check passed"
        return f"✗ {len(self.violations)} violation(s): " + ", ".join(
            f"{v.principle_id}({v.severity})" for v in self.violations
        )


class ConstitutionalChecker:
    """
    Checks agent outputs against constitutional principles.
    Each check_type has a dedicated validation method.
    """

    def __init__(
        self,
        registry: PrinciplesRegistry,
        max_violations_before_halt: int = 2,
        strict_mode: bool = False,
    ):
        self.registry = registry
        self.max_violations = max_violations_before_halt
        self.strict_mode = strict_mode
        logger.info(f"[ConstitutionalChecker] Ready with {len(registry)} principles")

    def check(
        self,
        output: str,
        confidence: float,
        reasoning_steps: list[str] | None = None,
        action: str | None = None,
        improvement_round: int = 0,
    ) -> CheckResult:
        """
        Run all constitutional checks on an output.
        """
        violations: list[ViolationReport] = []
        warnings: list[str] = []
        confidence_adjustment = 0.0

        for principle in self.registry.all():
            report = self._check_principle(
                principle, output, confidence, reasoning_steps or [], action, improvement_round
            )
            if report is not None:
                if principle.severity in (Severity.CRITICAL, Severity.HIGH):
                    violations.append(report)
                    confidence_adjustment -= 0.15
                else:
                    warnings.append(f"{principle.id}: {report.description}")
                    confidence_adjustment -= 0.05

        passed = len(violations) == 0
        if self.strict_mode and warnings:
            passed = False

        return CheckResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            confidence_adjustment=confidence_adjustment,
        )

    def _check_principle(
        self,
        p: Principle,
        output: str,
        confidence: float,
        steps: list[str],
        action: str | None,
        improvement_round: int,
    ) -> ViolationReport | None:
        """Dispatch to specific checker based on check_type."""
        check_type = p.check_type

        if check_type == "regex":
            return self._check_regex(p, output)
        elif check_type == "confidence_bound":
            return self._check_confidence(p, confidence, steps)
        elif check_type == "action_blacklist":
            return self._check_action_blacklist(p, action)
        elif check_type == "reward_hacking":
            return self._check_reward_hacking(p, output, confidence, improvement_round)
        elif check_type == "iteration_bound":
            return self._check_iteration_bound(p, improvement_round)
        elif check_type == "graph_cycle":
            return self._check_circular_reasoning(p, steps)
        elif check_type == "uncertainty_communication":
            return self._check_uncertainty(p, output, confidence)
        elif check_type == "content_safety":
            return self._check_content_safety(p, output)
        return None

    def _check_regex(self, p: Principle, output: str) -> ViolationReport | None:
        patterns = p.metadata.get("patterns", {})
        violations = patterns.get("violations", [])
        for pattern in violations:
            if re.search(pattern, output, re.IGNORECASE):
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description=f"Output contains prohibited phrase matching: '{pattern}'",
                    evidence=pattern,
                )
        return None

    def _check_confidence(self, p: Principle, confidence: float, steps: list[str]) -> ViolationReport | None:
        max_conf = p.metadata.get("max_confidence_without_proof", 0.95)
        has_proof = any("Q.E.D" in s or "Proof complete" in s or "Z3 SAT" in s for s in steps)
        if confidence > max_conf and not has_proof:
            return ViolationReport(
                principle_id=p.id,
                principle_name=p.name,
                severity=p.severity,
                description=f"Overconfident output: {confidence:.2%} exceeds {max_conf:.2%} without formal proof.",
                evidence=f"confidence={confidence:.2f}",
            )
        return None

    def _check_action_blacklist(self, p: Principle, action: str | None) -> ViolationReport | None:
        if action is None:
            return None
        blacklisted = p.metadata.get("blacklisted_actions", [])
        for banned in blacklisted:
            if banned.lower() in action.lower():
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description=f"Attempted blacklisted action: '{banned}'",
                    evidence=action,
                )
        return None

    def _check_reward_hacking(
        self,
        p: Principle,
        output: str,
        confidence: float,
        improvement_round: int,
    ) -> ViolationReport | None:
        heuristics = p.metadata.get("heuristics", [])
        output_lower = output.lower()

        if "self_assigned_perfect_score" in heuristics:
            if improvement_round > 0 and confidence >= 1.0:
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description="Self-assigned perfect confidence after self-improvement — potential reward hacking.",
                )

        if "empty_critique_with_high_confidence" in heuristics:
            if len(output.strip()) < 20 and confidence > 0.9:
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description="Empty/trivial output with high confidence.",
                )

        return None

    def _check_iteration_bound(self, p: Principle, improvement_round: int) -> ViolationReport | None:
        max_rounds = p.metadata.get("max_rounds", 3)
        if improvement_round > max_rounds:
            return ViolationReport(
                principle_id=p.id,
                principle_name=p.name,
                severity=p.severity,
                description=f"Self-improvement exceeded max rounds ({improvement_round} > {max_rounds}).",
            )
        return None

    def _check_circular_reasoning(self, p: Principle, steps: list[str]) -> ViolationReport | None:
        # Simple cycle detection: same step appearing twice
        seen = set()
        for step in steps:
            clean = step.strip().lower()
            if clean in seen and len(clean) > 20:
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description=f"Circular reasoning detected: repeated step '{clean[:60]}'",
                )
            seen.add(clean)
        return None

    def _check_uncertainty(self, p: Principle, output: str, confidence: float) -> ViolationReport | None:
        if confidence < 0.6:
            uncertainty_words = ["uncertain", "likely", "probably", "believe", "might", "possibly", "unclear"]
            if not any(w in output.lower() for w in uncertainty_words):
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description=f"Low confidence ({confidence:.2%}) but no uncertainty language in output.",
                )
        return None

    def _check_content_safety(self, p: Principle, output: str) -> ViolationReport | None:
        # Basic safety patterns
        harmful_patterns = [
            r"\bharm\b.*\bpeople\b", r"instructions.*\bexplosive\b",
            r"\bweapon\b.*\bmake\b",
        ]
        for pattern in harmful_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return ViolationReport(
                    principle_id=p.id,
                    principle_name=p.name,
                    severity=p.severity,
                    description=f"Potentially harmful content detected.",
                )
        return None
