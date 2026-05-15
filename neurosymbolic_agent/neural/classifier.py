"""
neural/classifier.py — Task-type and confidence classifier using PyTorch.
Classifies whether a task should go to the neural or symbolic path,
and estimates the model's confidence in that classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .embedder import Embedder


class TaskType(str, Enum):
    SYMBOLIC = "symbolic"
    NEURAL = "neural"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    scores: dict[str, float]
    reasoning: str


# Keyword heuristics for fast classification before neural model
SYMBOLIC_KEYWORDS = [
    r"\bprove\b", r"\btheorem\b", r"\bdeduct\b", r"\blogic\b", r"\biff\b",
    r"\ball .* are\b", r"\bsome .* are\b", r"\bif .* then\b", r"\bconstraint\b",
    r"\bsolve\b.*\bequat", r"\bverif", r"\bformal\b", r"\bsatisf", r"\bz3\b",
    r"\bprolog\b", r"\bfirst.order\b", r"\bpredicate\b",
]
NEURAL_KEYWORDS = [
    r"\bsentiment\b", r"\bclassif", r"\bsummariz", r"\btranslat", r"\bgenerat",
    r"\bwrite\b", r"\bparaphras", r"\bnamed.entity\b", r"\bner\b", r"\bembedd",
    r"\bsimilar", r"\bcluster", r"\btopic model",
]
HYBRID_KEYWORDS = [
    r"\bplan\b", r"\bstrateg", r"\breason\b", r"\banalog", r"\bq&a\b",
    r"\banswer\b", r"\bexplain\b", r"\bwhy\b", r"\bhow\b",
]


class TaskClassifierNet(nn.Module):
    """Small MLP for task-type classification."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskClassifier:
    """
    Classifies tasks into symbolic/neural/hybrid using:
    1. Keyword heuristics (fast path)
    2. Neural MLP on sentence embeddings (slow path)
    3. Confidence calibration via temperature scaling
    """

    def __init__(
        self,
        embedder: Embedder,
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        self.embedder = embedder
        self.device = torch.device(device)
        self.labels = [TaskType.SYMBOLIC, TaskType.NEURAL, TaskType.HYBRID]

        self.model = TaskClassifierNet(
            input_dim=embedder.embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=len(self.labels),
        ).to(self.device)
        self.temperature = 1.5  # Temperature scaling for calibrated confidence

        logger.info("[TaskClassifier] Initialised (untrained — using heuristic fallback)")

    def classify(self, task: str) -> ClassificationResult:
        """Classify a task and return type + confidence."""
        # Step 1: Keyword heuristics
        heuristic_result = self._heuristic_classify(task)
        if heuristic_result is not None:
            return heuristic_result

        # Step 2: Neural classification
        return self._neural_classify(task)

    def _heuristic_classify(self, task: str) -> ClassificationResult | None:
        task_lower = task.lower()
        sym_score = sum(1 for p in SYMBOLIC_KEYWORDS if re.search(p, task_lower))
        neu_score = sum(1 for p in NEURAL_KEYWORDS if re.search(p, task_lower))
        hyb_score = sum(1 for p in HYBRID_KEYWORDS if re.search(p, task_lower))

        total = sym_score + neu_score + hyb_score
        if total == 0:
            return None  # No clear signal, fall through to neural

        # Normalise
        scores = {
            TaskType.SYMBOLIC: sym_score / total,
            TaskType.NEURAL: neu_score / total,
            TaskType.HYBRID: hyb_score / total,
        }
        best = max(scores, key=lambda k: scores[k])
        confidence = scores[best]

        # Only use heuristic if confident
        if confidence < 0.5:
            return None

        return ClassificationResult(
            task_type=best,
            confidence=min(confidence + 0.1, 0.9),  # slight boost for heuristic match
            scores={k.value: v for k, v in scores.items()},
            reasoning=f"Keyword heuristic: sym={sym_score}, neu={neu_score}, hyb={hyb_score}",
        )

    def _neural_classify(self, task: str) -> ClassificationResult:
        embedding = self.embedder.embed(task)
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            # Temperature scaling for calibration
            calibrated_logits = logits / self.temperature
            probs = F.softmax(calibrated_logits, dim=-1).squeeze(0).cpu().numpy()

        scores = {label.value: float(probs[i]) for i, label in enumerate(self.labels)}
        best_idx = int(np.argmax(probs))
        best_label = self.labels[best_idx]
        confidence = float(probs[best_idx])

        # If model is untrained, confidence will be ~0.33 — flag as uncertain
        if confidence < 0.45:
            best_label = TaskType.HYBRID
            confidence = 0.45

        return ClassificationResult(
            task_type=best_label,
            confidence=confidence,
            scores=scores,
            reasoning=f"Neural classifier (temp={self.temperature})",
        )
