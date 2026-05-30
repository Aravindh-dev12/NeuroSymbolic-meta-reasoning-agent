"""
agent/self_improvement.py — Recursive self-improvement loop.
The agent critiques its own outputs, detects failure modes,
generates corrective reasoning traces, and updates working memory. Upgraded to support multiple backends (Anthropic, OpenAI, Local) dynamically!
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from constitutional.checker import ConstitutionalChecker, CheckResult
from constitutional.reward_hacking_detector import RewardHackingDetector, HackingSignal
from memory.memory_manager import MemoryManager


@dataclass
class CritiqueResult:
    has_issues: bool
    issues: list[str]
    failure_modes: list[str]
    suggested_correction: str
    corrected_answer: str
    corrected_confidence: float
    improvement_achieved: bool
    reasoning_trace: list[str] = field(default_factory=list)


CRITIC_SYSTEM_PROMPT = """You are the Self-Critic module of a NeuroSymbolic Meta-Reasoning Agent.

Your role is to critically evaluate a reasoning output and:
1. Identify any logical errors, unsupported claims, or reasoning failures
2. Classify failure modes: ["hallucination", "circular_reasoning", "overconfidence", "incomplete", "incorrect_path", "none"]
3. If issues exist, generate a corrected answer
4. Assign a corrected confidence score (be well-calibrated — do NOT inflate confidence)

Be strict but fair. If the answer is good, say so clearly.

Respond ONLY in this JSON format:
{
  "has_issues": true|false,
  "issues": ["<issue 1>", "<issue 2>"],
  "failure_modes": ["hallucination"|"circular_reasoning"|"overconfidence"|"incomplete"|"incorrect_path"|"none"],
  "suggested_correction": "<what to fix>",
  "corrected_answer": "<improved answer or original if no issues>",
  "corrected_confidence": <float 0.0-1.0>,
  "improvement_achieved": true|false,
  "reasoning_trace": ["<step 1>", "<step 2>"]
}"""


class SelfImprovementLoop:
    """
    Recursive self-improvement with:
    - Multi-round critique and correction
    - Constitutional constraint checking at each round
    - Reward hacking detection
    - Working memory updates with corrected traces
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        backend: str = "anthropic",
        max_rounds: int = 3,
        constitutional_checker: ConstitutionalChecker | None = None,
        memory_manager: MemoryManager | None = None,
        hacking_detector: RewardHackingDetector | None = None,
        local_llm: Optional[Any] = None,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model = model
        self.local_llm = local_llm
        self.api_key = api_key
        self._client = None

        # Setup backend client
        if backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif backend == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self.max_rounds = max_rounds
        self.constitutional_checker = constitutional_checker
        self.memory = memory_manager
        self.hacking_detector = hacking_detector or RewardHackingDetector()
        logger.info(f"[SelfImprovementLoop] Ready | backend={backend} | max_rounds={max_rounds}")

    def _llm_generate(self, prompt: str, system_prompt: str) -> str:
        """Call LLM client in a backend-agnostic way."""
        if self.backend == "local":
            if self.local_llm:
                return self.local_llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=1000,
                    temperature=0.2,
                )
            else:
                raise ValueError("Local LLM manager not provided in local backend mode")
        elif self.backend == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.content[0].text.strip()
        elif self.backend == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unsupported backend in SelfImprovementLoop: {self.backend}")

    def improve(
        self,
        task: str,
        initial_answer: str,
        initial_confidence: float,
        reasoning_steps: list[str] | None = None,
        path_used: str = "unknown",
    ) -> tuple[str, float, int, list[str]]:
        """
        Run the self-improvement loop.
        Returns: (final_answer, final_confidence, rounds_used, all_reasoning_steps)
        """
        answer = initial_answer
        confidence = initial_confidence
        steps = list(reasoning_steps or [])
        rounds_used = 0
        all_violations: list[str] = []

        for round_n in range(1, self.max_rounds + 1):
            logger.debug(f"[SelfImprovement] Round {round_n}/{self.max_rounds}")

            # Step 1: Constitutional check BEFORE critique
            if self.constitutional_checker:
                check_result = self.constitutional_checker.check(
                    output=answer,
                    confidence=confidence,
                    reasoning_steps=steps,
                    improvement_round=round_n,
                )
                confidence = max(0.05, confidence + check_result.confidence_adjustment)

                if check_result.has_critical_violations:
                    violation_summary = "; ".join(v.principle_id for v in check_result.violations)
                    all_violations.append(f"Round {round_n}: {violation_summary}")
                    logger.warning(f"[SelfImprovement] Critical violations: {violation_summary}")

                    # Attempt constitutional correction
                    answer = self._apply_constitutional_correction(answer, check_result)
                    steps.append(f"[Constitutional] Applied corrections for: {violation_summary}")

            # Step 2: Run self-critique
            critique = self._critique(task, answer, confidence, steps, path_used)
            rounds_used = round_n

            steps.append(f"[Critique Round {round_n}]")
            steps.extend(critique.reasoning_trace)

            # Step 3: Reward hacking check
            hacking_signals = self.hacking_detector.record_round(
                round_n=round_n,
                answer=answer,
                confidence=confidence,
                critique=str(critique.issues),
                corrected_answer=critique.corrected_answer,
                corrected_confidence=critique.corrected_confidence,
            )
            if hacking_signals:
                for signal in hacking_signals:
                    if signal.severity == "high":
                        logger.warning(f"[SelfImprovement] Reward hacking detected: {signal.signal_type}")
                        steps.append(f"[RewardHacking] Signal: {signal.description}")
                        # Penalise confidence
                        critique.corrected_confidence = min(
                            critique.corrected_confidence, confidence * 0.9
                        )

            # Step 4: Apply correction if improvement achieved
            if critique.has_issues and critique.improvement_achieved:
                if self.memory:
                    self.memory.update_from_critique(
                        key=f"task_{round_n}",
                        original=answer,
                        corrected=critique.corrected_answer,
                        critique=str(critique.issues),
                    )
                answer = critique.corrected_answer
                confidence = critique.corrected_confidence
                steps.append(f"[Applied correction] New confidence: {confidence:.2%}")
            elif not critique.has_issues:
                # No issues — stop early
                logger.debug(f"[SelfImprovement] No issues found in round {round_n}. Stopping.")
                steps.append(f"[Critique Round {round_n}] No issues found. Early stop.")
                break

        return answer, confidence, rounds_used, steps

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _critique(
        self,
        task: str,
        answer: str,
        confidence: float,
        steps: list[str],
        path_used: str,
    ) -> CritiqueResult:
        """Run one round of self-critique."""
        user_content = (
            f"Original task: {task}\n\n"
            f"Agent's answer: {answer}\n\n"
            f"Confidence: {confidence:.2%}\n"
            f"Path used: {path_used}\n\n"
            f"Reasoning steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps[-10:]))
        )

        raw = self._llm_generate(user_content, CRITIC_SYSTEM_PROMPT)

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return CritiqueResult(
                has_issues=bool(data.get("has_issues", False)),
                issues=data.get("issues", []),
                failure_modes=data.get("failure_modes", ["none"]),
                suggested_correction=data.get("suggested_correction", ""),
                corrected_answer=data.get("corrected_answer", answer),
                corrected_confidence=float(data.get("corrected_confidence", confidence)),
                improvement_achieved=bool(data.get("improvement_achieved", False)),
                reasoning_trace=data.get("reasoning_trace", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[SelfImprovement] Critique parse error: {e}")
            return CritiqueResult(
                has_issues=False,
                issues=[],
                failure_modes=["none"],
                suggested_correction="",
                corrected_answer=answer,
                corrected_confidence=confidence,
                improvement_achieved=False,
            )

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return CritiqueResult(
                has_issues=bool(data.get("has_issues", False)),
                issues=data.get("issues", []),
                failure_modes=data.get("failure_modes", ["none"]),
                suggested_correction=data.get("suggested_correction", ""),
                corrected_answer=data.get("corrected_answer", answer),
                corrected_confidence=float(data.get("corrected_confidence", confidence)),
                improvement_achieved=bool(data.get("improvement_achieved", False)),
                reasoning_trace=data.get("reasoning_trace", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[SelfImprovement] Critique parse error: {e}")
            return CritiqueResult(
                has_issues=False,
                issues=[],
                failure_modes=["none"],
                suggested_correction="",
                corrected_answer=answer,
                corrected_confidence=confidence,
                improvement_achieved=False,
            )

    def _apply_constitutional_correction(
        self,
        answer: str,
        check_result: CheckResult,
    ) -> str:
        """Apply simple corrections for constitutional violations."""
        corrected = answer

        for violation in check_result.violations:
            if "overconfident" in violation.description.lower():
                corrected += "\n\n[Note: I should clarify this answer involves some uncertainty.]"
            elif "harmful" in violation.description.lower():
                corrected = "[Constitutional constraint: Cannot provide this output.]"
            elif "circular" in violation.description.lower():
                corrected += "\n\n[Note: The above reasoning may have a circular element — please verify independently.]"

        return corrected
