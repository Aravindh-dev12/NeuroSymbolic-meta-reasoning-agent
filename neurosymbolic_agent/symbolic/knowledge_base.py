"""
symbolic/knowledge_base.py — Prolog-style fact and rule store.
Supports asserting facts, defining rules, and querying via forward chaining.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Fact:
    predicate: str
    args: list[str]

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"

    def matches(self, predicate: str, args: list[str | None]) -> bool:
        if self.predicate != predicate:
            return False
        for fact_arg, query_arg in zip(self.args, args):
            if query_arg is not None and fact_arg != query_arg:
                return False
        return True


@dataclass
class Rule:
    head_predicate: str
    head_args: list[str]
    body: list[tuple[str, list[str]]]  # list of (predicate, args) conditions
    description: str = ""

    def __str__(self) -> str:
        head = f"{self.head_predicate}({', '.join(self.head_args)})"
        body = ", ".join(f"{p}({', '.join(a)})" for p, a in self.body)
        return f"{head} :- {body}"


class KnowledgeBase:
    """
    Prolog-inspired knowledge base supporting:
    - Assert/retract facts
    - Define rules
    - Forward-chain queries
    - Persistence to JSON
    """

    def __init__(self, persist_path: str | None = None):
        self.facts: list[Fact] = []
        self.rules: list[Rule] = []
        self.persist_path = persist_path
        if persist_path and Path(persist_path).exists():
            self._load(persist_path)

    def assert_fact(self, predicate: str, *args: str) -> Fact:
        """Add a ground fact."""
        fact = Fact(predicate=predicate, args=list(args))
        if fact not in self.facts:
            self.facts.append(fact)
        return fact

    def retract_fact(self, predicate: str, *args: str) -> bool:
        """Remove a ground fact."""
        original_len = len(self.facts)
        self.facts = [
            f for f in self.facts
            if not (f.predicate == predicate and f.args == list(args))
        ]
        return len(self.facts) < original_len

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def query(self, predicate: str, *args: str | None) -> list[Fact]:
        """Return all facts matching the query pattern (None = wildcard)."""
        return [f for f in self.facts if f.matches(predicate, list(args))]

    def derive(self, predicate: str, entity: str, max_depth: int = 10) -> list[str]:
        """
        Derive all properties of an entity via forward chaining.
        Returns list of derivation steps.
        """
        derived: set[str] = set()
        steps: list[str] = []

        # Direct facts
        for fact in self.facts:
            if fact.predicate == predicate and entity in fact.args:
                derived.add(str(fact))
                steps.append(f"Fact: {fact}")

        # Rule-based inference (simplified forward chaining)
        changed = True
        depth = 0
        while changed and depth < max_depth:
            changed = False
            depth += 1
            for rule in self.rules:
                bindings = self._try_bind_rule(rule, entity)
                if bindings:
                    for binding in bindings:
                        derived_fact = str(binding)
                        if derived_fact not in derived:
                            derived.add(derived_fact)
                            self.facts.append(binding)
                            steps.append(f"Derived (rule: {rule.description}): {binding}")
                            changed = True

        return steps

    def _try_bind_rule(self, rule: Rule, entity: str) -> list[Fact]:
        """Attempt to fire a rule with a given entity binding. Simplified."""
        # Check if all body conditions are satisfied
        variable_bindings: dict[str, str] = {}
        for cond_pred, cond_args in rule.body:
            resolved_args = [variable_bindings.get(a, a) for a in cond_args]
            matches = self.query(cond_pred, *[a if a != "?" else None for a in resolved_args])
            if not matches:
                return []
            # Bind variables
            for match in matches:
                for arg, val in zip(cond_args, match.args):
                    if arg.startswith("?"):
                        variable_bindings[arg] = val

        # Fire rule head
        head_args = [variable_bindings.get(a, a) for a in rule.head_args]
        new_fact = Fact(predicate=rule.head_predicate, args=head_args)
        if new_fact not in self.facts:
            return [new_fact]
        return []

    def summary(self) -> dict[str, Any]:
        return {
            "fact_count": len(self.facts),
            "rule_count": len(self.rules),
            "predicates": list({f.predicate for f in self.facts}),
        }

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for f in data.get("facts", []):
            self.facts.append(Fact(**f))

    def save(self) -> None:
        if not self.persist_path:
            return
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(
                {"facts": [{"predicate": f.predicate, "args": f.args} for f in self.facts]},
                f, indent=2,
            )
