"""
agent/meta_controller.py — Production-grade LLM-powered meta-controller.
Analyses tasks, selects reasoning paths, estimates confidence, and orchestrates execution.
Supports local LLMs (llama.cpp, Ollama) and cloud LLMs (Anthropic, OpenAI).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from llm.local_llm_manager import LocalLLMManager, create_llm_manager
from neural.classifier import TaskClassifier, TaskType, ClassificationResult
from neural.embedder import Embedder


@dataclass
class RoutingDecision:
    path: str  # "neural", "symbolic", "hybrid"
    confidence: float
    task_type: str
    reasoning: str
    subtask_hints: list[str] = field(default_factory=list)
    needs_planning: bool = False
    facts_extracted: list[str] = field(default_factory=list)


META_CONTROLLER_SYSTEM = """You are the Meta-Controller of a NeuroSymbolic Meta-Reasoning Agent.

Your role is to analyse incoming tasks and decide:
1. Which reasoning path to use: "neural" (NLP, pattern matching, generation) OR "symbolic" (logic, math, formal reasoning) OR "hybrid" (both)
2. Your confidence in this routing decision (0.0 to 1.0)  
3. Whether the task needs hierarchical planning (complex multi-step tasks)
4. Any facts or rules to extract for the symbolic knowledge base

Guidelines:
- Use SYMBOLIC for: syllogisms, formal logic, arithmetic, constraint problems, theorem proving
- Use NEURAL for: sentiment, classification, NLU, generation, summarisation, similarity
- Use HYBRID for: question answering, planning, analogical reasoning, multi-step problems

Respond ONLY in this JSON format:
{
  "path": "neural|symbolic|hybrid",
  "confidence": <float 0.0-1.0>,
  "task_type": "<specific type, e.g. syllogism, sentiment_analysis, planning>",
  "reasoning": "<1-2 sentences explaining your choice>",
  "needs_planning": <true|false>,
  "facts_extracted": ["<fact 1>", "<fact 2>"],
  "subtask_hints": ["<hint 1>", "<hint 2>"]
}"""


class MetaController:
    """
    Production-grade LLM-based meta-controller that routes tasks to appropriate reasoning paths.
    Supports local LLMs (llama.cpp, Ollama) and cloud LLMs (Anthropic, OpenAI).
    Combines LLM judgment with neural classifier for robust routing.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        backend: str = "anthropic",
        confidence_threshold: float = 0.75,
        embedder: Embedder | None = None,
        classifier: TaskClassifier | None = None,
        local_llm: Optional[LocalLLMManager] = None,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model = model
        self.confidence_threshold = confidence_threshold
        self._embedder = embedder or Embedder()
        self._classifier = classifier or TaskClassifier(self._embedder)
        
        # Initialize LLM backend
        if backend == "local":
            self.local_llm = local_llm or create_llm_manager(model_name=model)
            self.client = None
        elif backend == "anthropic":
            api_key = api_key or anthropic.api_key
            if not api_key:
                raise ValueError("Anthropic API key required for anthropic backend")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.local_llm = None
        elif backend == "openai":
            try:
                import openai
                api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key required for openai backend")
                self.client = openai.OpenAI(api_key=api_key)
                self.local_llm = None
            except ImportError:
                raise ValueError("openai package required for openai backend")
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        logger.info(f"[MetaController] Ready | backend={backend} | model={model} | threshold={confidence_threshold}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def route(self, task: str, memory_context: str = "") -> RoutingDecision:
        """
        Analyse a task and decide the reasoning path.
        Combines neural classifier signal with LLM meta-reasoning.
        """
        # Step 1: Fast neural classification
        classification = self._classifier.classify(task)
        logger.debug(f"[MetaController] Classifier: {classification.task_type} ({classification.confidence:.2%})")

        # Step 2: LLM meta-reasoning (more nuanced)
        user_content = f"Task: {task}"
        if memory_context:
            user_content += f"\n\nRelevant context from memory:\n{memory_context}"
        if classification.task_type != TaskType.UNKNOWN:
            user_content += f"\n\nHint: Neural classifier suggests: {classification.task_type.value} (confidence: {classification.confidence:.2%})"

        # Call appropriate LLM backend
        if self.backend == "local":
            raw = self.local_llm.generate(
                prompt=user_content,
                system_prompt=META_CONTROLLER_SYSTEM,
                max_tokens=800,
                temperature=0.3,
                json_mode=True,
            )
        elif self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                system=META_CONTROLLER_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
        elif self.backend == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": META_CONTROLLER_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[MetaController] JSON parse failed, using classifier result")
            return self._fallback_routing(task, classification)

        # Step 3: Merge LLM and classifier signals
        llm_confidence = float(data.get("confidence", 0.7))

        # If they disagree significantly, lower confidence
        llm_path = data.get("path", "hybrid")
        classifier_path = classification.task_type.value
        if llm_path != classifier_path and classifier_path != "unknown":
            llm_confidence = max(llm_confidence * 0.85, 0.4)  # slight penalty for disagreement

        # Active Inference Scaling: Force planning and critique for highly complex keywords
        needs_planning = bool(data.get("needs_planning", False))
        task_lower = task.lower()
        complex_keywords = ["einstein", "knights and knaves", "puzzle", "schedule", "plan", "complex", "optimize", "system of equations", "derivative", "integral"]
        if any(w in task_lower for w in complex_keywords):
            needs_planning = True
            llm_confidence = min(llm_confidence, 0.65)  # Force self-critique refiner loop for safety and precision

        return RoutingDecision(
            path=llm_path,
            confidence=llm_confidence,
            task_type=data.get("task_type", "unknown"),
            reasoning=data.get("reasoning", ""),
            needs_planning=needs_planning,
            facts_extracted=data.get("facts_extracted", []),
            subtask_hints=data.get("subtask_hints", []),
        )

    def embed_task(self, task: str):
        """Embed task for memory retrieval."""
        return self._embedder.embed(task)

    def _fallback_routing(
        self,
        task: str,
        classification: ClassificationResult,
    ) -> RoutingDecision:
        """Fallback when LLM routing fails."""
        path_map = {
            TaskType.SYMBOLIC: "symbolic",
            TaskType.NEURAL: "neural",
            TaskType.HYBRID: "hybrid",
            TaskType.UNKNOWN: "hybrid",
        }
        return RoutingDecision(
            path=path_map[classification.task_type],
            confidence=classification.confidence * 0.8,
            task_type=classification.task_type.value,
            reasoning=f"Fallback to neural classifier: {classification.reasoning}",
        )

    def is_confident(self, decision: RoutingDecision) -> bool:
        return decision.confidence >= self.confidence_threshold
