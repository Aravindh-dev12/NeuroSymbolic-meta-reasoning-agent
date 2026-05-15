"""
neural/inference.py — Neural inference pipeline.
Handles NLP tasks: classification, sentiment, summarisation, similarity, NER.
Uses the LLM as the core neural engine, wrapped with structured prompting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class NeuralInferenceResult:
    task: str
    output: str
    task_type: str
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""


NEURAL_SYSTEM_PROMPT = """You are the Neural Inference Engine of a NeuroSymbolic Meta-Reasoning Agent.
Your role is to handle NLP tasks that benefit from neural pattern recognition:
- Sentiment analysis
- Text classification
- Named entity recognition
- Summarisation
- Semantic similarity
- Text generation
- Paraphrasing

For each task, provide:
1. Your output/answer
2. Your confidence score (0.0 to 1.0) — be well-calibrated, not overconfident
3. Brief reasoning

Respond ONLY in this JSON format:
{
  "output": "<your answer>",
  "confidence": <float 0.0-1.0>,
  "task_type": "<detected subtask>",
  "reasoning": "<1-2 sentence explanation>"
}"""


class NeuralInferencePipeline:
    """Neural inference pipeline backed by LLM."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
        logger.info("[NeuralInference] Pipeline ready")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def run(self, task: str, context: str = "") -> NeuralInferenceResult:
        """Run neural inference on a task."""
        user_content = f"Task: {task}"
        if context:
            user_content += f"\n\nAdditional context:\n{context}"

        logger.debug(f"[NeuralInference] Running inference on: {task[:80]}")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=NEURAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text.strip()

        try:
            # Strip markdown fences if present
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
            return NeuralInferenceResult(
                task=task,
                output=data.get("output", ""),
                task_type=data.get("task_type", "neural"),
                confidence=float(data.get("confidence", 0.7)),
                metadata={"reasoning": data.get("reasoning", "")},
                raw_response=raw,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[NeuralInference] JSON parse error: {e}. Using raw response.")
            return NeuralInferenceResult(
                task=task,
                output=raw,
                task_type="neural",
                confidence=0.5,
                raw_response=raw,
            )
