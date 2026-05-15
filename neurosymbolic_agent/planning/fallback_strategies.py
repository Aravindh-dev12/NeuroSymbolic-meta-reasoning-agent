"""
planning/fallback_strategies.py — Fallback and recovery strategies for execution failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any

from loguru import logger


class FallbackType(str, Enum):
    RETRY = "retry"
    PATH_SWITCH = "path_switch"   # Switch neural ↔ symbolic
    SIMPLIFY = "simplify"          # Reduce task complexity
    DECOMPOSE = "decompose"        # Further decompose
    DEFER = "defer"                # Skip and continue
    ABORT = "abort"                # Cannot proceed


@dataclass
class FallbackDecision:
    fallback_type: FallbackType
    modified_task: str
    reason: str
    confidence_penalty: float = 0.1


class FallbackStrategyEngine:
    """
    Decides what to do when a reasoning step fails.
    Uses a priority queue of strategies based on failure type and context.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._retry_counts: dict[str, int] = {}

    def decide(
        self,
        task: str,
        failure_reason: str,
        current_path: str,  # "neural" or "symbolic"
        confidence: float,
        attempt: int = 0,
    ) -> FallbackDecision:
        """Choose the best fallback strategy."""
        task_id = hash(task[:50])

        self._retry_counts[task_id] = self._retry_counts.get(task_id, 0) + 1
        retry_count = self._retry_counts[task_id]

        logger.debug(f"[FallbackStrategy] Failure: {failure_reason[:60]}, attempt={attempt}")

        # Strategy selection logic
        if "timeout" in failure_reason.lower() or "z3" in failure_reason.lower():
            return FallbackDecision(
                fallback_type=FallbackType.PATH_SWITCH,
                modified_task=task,
                reason=f"Symbolic solver failed ({failure_reason}). Switching to neural path.",
                confidence_penalty=0.15,
            )

        if "json" in failure_reason.lower() or "parse" in failure_reason.lower():
            return FallbackDecision(
                fallback_type=FallbackType.RETRY,
                modified_task=f"{task}\n\nNote: Please respond ONLY in valid JSON format.",
                reason="Response parsing failed. Retrying with format reminder.",
                confidence_penalty=0.05,
            )

        if confidence < 0.4 and retry_count < self.max_retries:
            return FallbackDecision(
                fallback_type=FallbackType.SIMPLIFY,
                modified_task=self._simplify_task(task),
                reason=f"Low confidence ({confidence:.2%}). Simplifying task.",
                confidence_penalty=0.1,
            )

        if retry_count >= self.max_retries:
            return FallbackDecision(
                fallback_type=FallbackType.DEFER,
                modified_task=task,
                reason=f"Max retries ({self.max_retries}) reached. Deferring with partial answer.",
                confidence_penalty=0.3,
            )

        if attempt == 0:
            return FallbackDecision(
                fallback_type=FallbackType.RETRY,
                modified_task=task,
                reason="First failure — simple retry.",
                confidence_penalty=0.05,
            )

        return FallbackDecision(
            fallback_type=FallbackType.PATH_SWITCH,
            modified_task=task,
            reason=f"Multiple failures on {current_path} path. Switching.",
            confidence_penalty=0.2,
        )

    def _simplify_task(self, task: str) -> str:
        """Produce a simpler version of the task."""
        # Trim to first sentence/clause
        for sep in [".", "?", "!", ";"]:
            if sep in task:
                return task.split(sep)[0].strip() + sep
        return task[:100] + "..." if len(task) > 100 else task
