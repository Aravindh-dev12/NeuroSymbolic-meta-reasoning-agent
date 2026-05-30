"""
utils/trace_recorder.py — Records full reasoning traces to JSONL for analysis.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    step_id: str
    step_type: str  # "routing", "neural", "symbolic", "critique", "memory_update", "constitutional"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    input_summary: str = ""
    output_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningTrace(BaseModel):
    trace_id: str
    task: str
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str = ""
    steps: list[ReasoningStep] = Field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    path_used: str = ""
    self_improvement_rounds: int = 0
    constitutional_violations: list[str] = Field(default_factory=list)
    success: bool = False


class TraceRecorder:
    """Records and persists reasoning traces."""

    def __init__(self, trace_file: str = "logs/reasoning_traces.jsonl"):
        self.trace_file = trace_file
        os.makedirs(Path(trace_file).parent, exist_ok=True)
        self._current_trace: ReasoningTrace | None = None

    def start_trace(self, trace_id: str, task: str) -> ReasoningTrace:
        self._current_trace = ReasoningTrace(trace_id=trace_id, task=task)
        return self._current_trace

    def add_step(
        self,
        step_type: str,
        input_summary: str = "",
        output_summary: str = "",
        **metadata,
    ) -> ReasoningStep:
        if self._current_trace is None:
            raise RuntimeError("No active trace. Call start_trace() first.")

        step = ReasoningStep(
            step_id=f"{self._current_trace.trace_id}_{len(self._current_trace.steps):03d}",
            step_type=step_type,
            input_summary=input_summary[:500],
            output_summary=output_summary[:500],
            metadata=metadata,
        )
        self._current_trace.steps.append(step)
        return step

    def complete_trace(
        self,
        final_answer: str,
        final_confidence: float,
        path_used: str,
        self_improvement_rounds: int = 0,
        constitutional_violations: list[str] | None = None,
        success: bool = True,
    ) -> ReasoningTrace:
        if self._current_trace is None:
            raise RuntimeError("No active trace.")

        self._current_trace.completed_at = datetime.utcnow().isoformat()
        self._current_trace.final_answer = final_answer
        self._current_trace.final_confidence = final_confidence
        self._current_trace.path_used = path_used
        self._current_trace.self_improvement_rounds = self_improvement_rounds
        self._current_trace.constitutional_violations = constitutional_violations or []
        self._current_trace.success = success

        self._persist(self._current_trace)
        trace = self._current_trace
        self._current_trace = None
        return trace

    def _persist(self, trace: ReasoningTrace) -> None:
        with open(self.trace_file, "a", encoding="utf-8") as f:
            f.write(trace.model_dump_json() + "\n")

    @property
    def current_trace(self) -> ReasoningTrace | None:
        return self._current_trace
