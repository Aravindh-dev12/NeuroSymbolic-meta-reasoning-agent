"""
constitutional/reward_hacking_detector.py — Detects reward hacking in the self-improvement loop.
Monitors patterns that indicate the agent is gaming its own evaluation criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class HackingSignal:
    signal_type: str
    severity: str  # "low", "medium", "high"
    description: str
    round_number: int
    evidence: dict[str, Any] = field(default_factory=dict)


class RewardHackingDetector:
    """
    Monitors the self-improvement loop for reward hacking patterns:

    1. Confidence inflation: confidence increases without answer improvement
    2. Critique suppression: critiques become shorter/less specific over rounds
    3. Circular correction: agent "corrects" to the same answer
    4. Metric gaming: agent focuses on measurable proxies rather than quality
    """

    def __init__(self):
        self._history: list[dict[str, Any]] = []
        logger.info("[RewardHackingDetector] Initialised")

    def record_round(
        self,
        round_n: int,
        answer: str,
        confidence: float,
        critique: str,
        corrected_answer: str,
        corrected_confidence: float,
    ) -> list[HackingSignal]:
        """
        Record a self-improvement round and detect hacking signals.
        Returns list of detected signals (empty = clean).
        """
        entry = {
            "round": round_n,
            "answer": answer,
            "confidence": confidence,
            "critique": critique,
            "corrected_answer": corrected_answer,
            "corrected_confidence": corrected_confidence,
        }
        self._history.append(entry)

        signals = []
        signals.extend(self._detect_confidence_inflation(entry))
        signals.extend(self._detect_critique_suppression())
        signals.extend(self._detect_circular_correction(entry))

        if signals:
            logger.warning(f"[RewardHackingDetector] {len(signals)} signal(s) in round {round_n}")

        return signals

    def _detect_confidence_inflation(self, entry: dict) -> list[HackingSignal]:
        """Confidence jump without substantial answer change."""
        signals = []
        delta_conf = entry["corrected_confidence"] - entry["confidence"]
        answer_change = self._text_change_ratio(entry["answer"], entry["corrected_answer"])

        if delta_conf > 0.2 and answer_change < 0.1:
            signals.append(HackingSignal(
                signal_type="confidence_inflation",
                severity="high",
                description=(
                    f"Confidence jumped +{delta_conf:.2%} but answer barely changed ({answer_change:.1%} diff). "
                    "Suspected self-assigned score inflation."
                ),
                round_number=entry["round"],
                evidence={"delta_conf": delta_conf, "answer_change": answer_change},
            ))
        return signals

    def _detect_critique_suppression(self) -> list[HackingSignal]:
        """Critiques getting shorter (less rigorous) over rounds."""
        signals = []
        if len(self._history) < 2:
            return signals

        critique_lengths = [len(h["critique"]) for h in self._history[-3:]]
        if len(critique_lengths) >= 2 and all(
            critique_lengths[i] > critique_lengths[i + 1] * 1.5
            for i in range(len(critique_lengths) - 1)
        ):
            signals.append(HackingSignal(
                signal_type="critique_suppression",
                severity="medium",
                description="Critique length decreasing rapidly — possible self-critique gaming.",
                round_number=self._history[-1]["round"],
                evidence={"critique_lengths": critique_lengths},
            ))
        return signals

    def _detect_circular_correction(self, entry: dict) -> list[HackingSignal]:
        """Answer reverts to a previous answer."""
        signals = []
        if len(self._history) < 3:
            return signals

        current = entry["corrected_answer"].strip().lower()
        for prev_entry in self._history[:-1]:
            prev = prev_entry["answer"].strip().lower()
            if current == prev and entry["round"] > 1:
                signals.append(HackingSignal(
                    signal_type="circular_correction",
                    severity="high",
                    description=f"Answer in round {entry['round']} reverts to round {prev_entry['round']} answer.",
                    round_number=entry["round"],
                ))
                break
        return signals

    @staticmethod
    def _text_change_ratio(a: str, b: str) -> float:
        """Estimate how much text changed (0 = identical, 1 = completely different)."""
        if not a and not b:
            return 0.0
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words and not b_words:
            return 0.0
        intersection = len(a_words & b_words)
        union = len(a_words | b_words)
        return 1.0 - (intersection / union) if union > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "rounds_monitored": len(self._history),
            "history": self._history,
        }
