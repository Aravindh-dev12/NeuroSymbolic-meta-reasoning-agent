"""
planning/hierarchical_planner.py — Recursive hierarchical task decomposition.
Breaks complex tasks into subtasks, plans execution order, and manages dependencies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    id: str
    description: str
    task_type: str  # "neural", "symbolic", "hybrid"
    depth: int = 0
    status: SubtaskStatus = SubtaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)  # IDs of prerequisite subtasks
    result: str = ""
    children: list["Subtask"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    task: str
    subtasks: list[Subtask]
    estimated_complexity: str  # "low", "medium", "high"
    requires_symbolic: bool
    requires_neural: bool
    max_depth: int = 0
    raw_plan: str = ""


PLANNER_SYSTEM_PROMPT = """You are the Hierarchical Task Planner of a NeuroSymbolic Meta-Reasoning Agent.
Your job is to decompose complex tasks into a structured execution plan.

For each task:
1. Break it into 2-5 concrete, executable subtasks
2. Classify each subtask as: "symbolic" (logic/math/formal), "neural" (NLP/pattern), or "hybrid"
3. Identify dependencies between subtasks
4. Estimate overall complexity: "low", "medium", or "high"

Respond ONLY in this JSON format:
{
  "subtasks": [
    {
      "id": "s1",
      "description": "<what to do>",
      "task_type": "symbolic|neural|hybrid",
      "dependencies": []
    }
  ],
  "estimated_complexity": "low|medium|high",
  "requires_symbolic": true|false,
  "requires_neural": true|false,
  "reasoning": "<why this decomposition>"
}"""


class HierarchicalPlanner:
    """
    Decomposes complex tasks into executable subtask hierarchies.
    Uses LLM for decomposition, then validates and orders subtasks.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_depth: int = 5,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_depth = max_depth
        logger.info("[HierarchicalPlanner] Ready")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def decompose(self, task: str, depth: int = 0) -> ExecutionPlan:
        """Decompose a task into an execution plan."""
        if depth >= self.max_depth:
            # Leaf node — execute directly
            return ExecutionPlan(
                task=task,
                subtasks=[Subtask(id="s0", description=task, task_type="hybrid", depth=depth)],
                estimated_complexity="low",
                requires_symbolic=False,
                requires_neural=True,
            )

        logger.debug(f"[Planner] Decomposing (depth={depth}): {task[:60]}")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Task: {task}"}],
        )

        raw = response.content[0].text.strip()

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[Planner] JSON parse failed, using simple plan")
            return self._simple_plan(task, depth)

        subtasks = []
        for s in data.get("subtasks", []):
            subtasks.append(Subtask(
                id=s["id"],
                description=s["description"],
                task_type=s.get("task_type", "hybrid"),
                depth=depth,
                dependencies=s.get("dependencies", []),
            ))

        plan = ExecutionPlan(
            task=task,
            subtasks=subtasks,
            estimated_complexity=data.get("estimated_complexity", "medium"),
            requires_symbolic=data.get("requires_symbolic", False),
            requires_neural=data.get("requires_neural", True),
            max_depth=depth,
            raw_plan=raw,
        )

        logger.debug(f"[Planner] Plan: {len(subtasks)} subtasks, complexity={plan.estimated_complexity}")
        return plan

    def _simple_plan(self, task: str, depth: int) -> ExecutionPlan:
        """Fallback: single-step plan."""
        return ExecutionPlan(
            task=task,
            subtasks=[Subtask(id="s0", description=task, task_type="hybrid", depth=depth)],
            estimated_complexity="medium",
            requires_symbolic=False,
            requires_neural=True,
        )

    def topological_sort(self, subtasks: list[Subtask]) -> list[Subtask]:
        """Return subtasks in dependency order (topological sort)."""
        id_to_task = {s.id: s for s in subtasks}
        visited: set[str] = set()
        order: list[Subtask] = []

        def visit(task_id: str) -> None:
            if task_id in visited:
                return
            visited.add(task_id)
            task = id_to_task.get(task_id)
            if task is None:
                return
            for dep in task.dependencies:
                visit(dep)
            order.append(task)

        for subtask in subtasks:
            visit(subtask.id)

        return order
