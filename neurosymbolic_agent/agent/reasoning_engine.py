"""
agent/reasoning_engine.py — Orchestrates neural and symbolic reasoning paths.
Receives a routing decision and executes the appropriate reasoning pipeline.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from neural.inference import NeuralInferencePipeline, NeuralInferenceResult
from symbolic.solver import SymbolicSolver, SolverResult
from symbolic.knowledge_base import KnowledgeBase


@dataclass
class ReasoningResult:
    answer: str
    confidence: float
    path_used: str
    reasoning_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


HYBRID_SYSTEM = """You are the Hybrid Reasoning Engine. Combine neural and symbolic insights to answer the task.
Use the symbolic results for formal/logical components and neural results for natural language understanding.
Be clear, accurate, and well-calibrated in your confidence.

Respond in JSON:
{
  "answer": "<final synthesised answer>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<how you combined both approaches>"
}"""


class ReasoningEngine:
    """
    Executes reasoning on the selected path (neural/symbolic/hybrid).
    Handles path-specific logic and produces a unified ReasoningResult.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        backend: str = "anthropic",
        neural_pipeline: NeuralInferencePipeline | None = None,
        symbolic_solver: SymbolicSolver | None = None,
        knowledge_base: KnowledgeBase | None = None,
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

        self.kb = knowledge_base or KnowledgeBase()

        # Pass backend config down
        self.neural = neural_pipeline or NeuralInferencePipeline(
            model=model,
            backend=backend,
            local_llm=local_llm,
            api_key=api_key,
        )
        self.symbolic = symbolic_solver or SymbolicSolver(
            backend=backend,
            model=model,
            local_llm=local_llm,
            api_key=api_key,
        )
        logger.info(f"[ReasoningEngine] Initialised | backend={backend}")

    def _llm_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call LLM client in a backend-agnostic way."""
        if self.backend == "local":
            if self.local_llm:
                return self.local_llm.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=1000,
                    temperature=0.3,
                )
            else:
                raise ValueError("Local LLM manager not provided in local backend mode")
        elif self.backend == "anthropic":
            kwargs = {
                "model": self.model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = self._client.messages.create(**kwargs)
            return response.content[0].text.strip()
        elif self.backend == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=messages,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unsupported backend in ReasoningEngine: {self.backend}")

    def execute(
        self,
        task: str,
        path: str,
        memory_context: str = "",
        extracted_facts: list[str] | None = None,
    ) -> ReasoningResult:
        """Execute reasoning on the given path."""
        extracted_facts = extracted_facts or []

        # Add extracted facts to KB
        for fact_str in extracted_facts:
            self._parse_and_assert_fact(fact_str)

        if path == "symbolic":
            return self._symbolic_path(task, extracted_facts)
        elif path == "neural":
            return self._neural_path(task, memory_context)
        elif path == "hybrid":
            return self._hybrid_path(task, memory_context, extracted_facts)
        else:
            logger.warning(f"[ReasoningEngine] Unknown path '{path}', defaulting to hybrid")
            return self._hybrid_path(task, memory_context, extracted_facts)

    # ─── Symbolic Path ────────────────────────────────────────────────────────

    def _symbolic_path(self, task: str, facts: list[str]) -> ReasoningResult:
        logger.debug("[ReasoningEngine] Running symbolic path")
        result: SolverResult = self.symbolic.solve(task, facts=facts)
        steps = result.proof_steps.copy()

        if result.confidence < 0.6:
            # Symbolic failed — augment with LLM
            logger.debug("[ReasoningEngine] Symbolic confidence low, augmenting with LLM")
            llm_result = self._llm_symbolic_fallback(task, steps)
            steps.append("[LLM augmentation applied]")
            return ReasoningResult(
                answer=llm_result,
                confidence=0.65,
                path_used="symbolic+llm",
                reasoning_steps=steps,
            )

        return ReasoningResult(
            answer=result.answer,
            confidence=result.confidence,
            path_used="symbolic",
            reasoning_steps=steps,
            metadata={"solver_status": result.status.value, "model": result.model},
        )

    # ─── Neural Path ─────────────────────────────────────────────────────────

    def _neural_path(self, task: str, context: str) -> ReasoningResult:
        logger.debug("[ReasoningEngine] Running neural path")
        result: NeuralInferenceResult = self.neural.run(task, context=context)
        return ReasoningResult(
            answer=result.output,
            confidence=result.confidence,
            path_used="neural",
            reasoning_steps=[result.metadata.get("reasoning", "Neural inference")],
            metadata={"task_type": result.task_type},
        )

    # ─── Hybrid Path ─────────────────────────────────────────────────────────

    def _hybrid_path(
        self,
        task: str,
        context: str,
        facts: list[str],
    ) -> ReasoningResult:
        logger.debug("[ReasoningEngine] Running hybrid path")
        steps = []

        # Run both paths
        sym_result = self.symbolic.solve(task, facts=facts)
        neu_result = self.neural.run(task, context=context)

        steps.append(f"[Symbolic] {sym_result.answer[:100]}")
        steps.append(f"[Neural] {neu_result.output[:100]}")

        # Combine via LLM
        combined = self._synthesise(task, sym_result, neu_result, steps)
        return combined

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _synthesise(
        self,
        task: str,
        sym: SolverResult,
        neu: NeuralInferenceResult,
        steps: list[str],
    ) -> ReasoningResult:
        """Use LLM to synthesise symbolic and neural results."""
        user_content = (
            f"Task: {task}\n\n"
            f"Symbolic reasoning result:\n{sym.answer}\n"
            f"Proof steps: {sym.proof_steps}\n\n"
            f"Neural reasoning result:\n{neu.output}\n"
            f"Neural confidence: {neu.confidence:.2%}"
        )

        raw = self._llm_generate(user_content, HYBRID_SYSTEM)

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            steps.append(f"[Synthesis] {data.get('reasoning', '')[:100]}")
            return ReasoningResult(
                answer=data.get("answer", neu.output),
                confidence=float(data.get("confidence", (sym.confidence + neu.confidence) / 2)),
                path_used="hybrid",
                reasoning_steps=steps,
            )
        except Exception:
            # Fallback: use whichever had higher confidence
            if sym.confidence > neu.confidence:
                return ReasoningResult(
                    answer=sym.answer, confidence=sym.confidence,
                    path_used="hybrid(sym)", reasoning_steps=steps,
                )
            return ReasoningResult(
                answer=neu.output, confidence=neu.confidence,
                path_used="hybrid(neu)", reasoning_steps=steps,
            )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def _llm_symbolic_fallback(self, task: str, symbolic_steps: list[str]) -> str:
        """When symbolic solver produces low-confidence output, use LLM to augment."""
        prompt = (
            f"Task: {task}\n\n"
            f"Partial symbolic reasoning: {symbolic_steps}\n\n"
            "Please complete the reasoning and provide a final answer. "
            "Be explicit about any uncertainty."
        )
        return self._llm_generate(prompt)

    def _parse_and_assert_fact(self, fact_str: str) -> None:
        """Parse a natural language fact and assert it into the knowledge base."""
        import re
        # "X is a Y" / "X are Y"
        m = re.match(r"(\w+)\s+(?:is|are)\s+(?:a\s+)?(\w+)", fact_str.lower().strip())
        if m:
            self.kb.assert_fact("is_a", m.group(1), m.group(2))
        # "All X are Y"
        m2 = re.match(r"(?:all|every)\s+(\w+)\s+(?:are|is)\s+(\w+)", fact_str.lower().strip())
        if m2:
            self.kb.assert_fact("all_are", m2.group(1), m2.group(2))
