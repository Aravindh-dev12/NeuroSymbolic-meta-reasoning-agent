"""
symbolic/solver.py — Z3-based symbolic solver for logic, constraints, and formal reasoning.
Handles propositional logic, first-order logic fragments, arithmetic constraints,
and syllogistic reasoning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class SolverStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


@dataclass
class SolverResult:
    status: SolverStatus
    answer: str
    proof_steps: list[str] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasoning: str = ""


class SymbolicSolver:
    """
    Symbolic reasoning engine using Z3 SMT solver.
    Handles:
    - Syllogistic reasoning (All A are B, X is A => X is B)
    - Arithmetic constraint solving
    - Propositional logic (SAT/UNSAT)
    - Simple first-order queries
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_ms = timeout_seconds * 1000
        self._z3_available = self._check_z3()
        logger.info(f"[SymbolicSolver] Z3 available: {self._z3_available}")

    def _check_z3(self) -> bool:
        try:
            import z3  # noqa: F401
            return True
        except ImportError:
            logger.warning("[SymbolicSolver] Z3 not installed. Using pattern-based fallback.")
            return False

    def solve(self, task: str, facts: list[str] | None = None) -> SolverResult:
        """
        Solve a symbolic reasoning task.
        Auto-detects problem type and routes to appropriate solver.
        """
        facts = facts or []
        task_lower = task.lower()

        # Route to appropriate solver
        if self._is_syllogism(task_lower):
            return self._solve_syllogism(task, facts)
        elif self._is_arithmetic(task_lower):
            return self._solve_arithmetic(task, facts)
        elif self._is_propositional(task_lower):
            return self._solve_propositional(task, facts)
        else:
            return self._solve_generic_logic(task, facts)

    # ─── Problem Type Detection ───────────────────────────────────────────────

    def _is_syllogism(self, task: str) -> bool:
        patterns = [
            r"all .+ are", r"some .+ are", r"no .+ are",
            r"every .+ is", r"is .+ a\b", r"are .+ a\b",
        ]
        return any(re.search(p, task) for p in patterns)

    def _is_arithmetic(self, task: str) -> bool:
        patterns = [r"\d+\s*[\+\-\*\/]\s*\d+", r"solve.*equation", r"find.*x", r"x\s*="]
        return any(re.search(p, task) for p in patterns)

    def _is_propositional(self, task: str) -> bool:
        patterns = [r"\bif\b.*\bthen\b", r"\band\b.*\bor\b", r"\bnot\b.*\band\b", r"\bimplies\b"]
        return any(re.search(p, task) for p in patterns)

    # ─── Syllogistic Reasoning ────────────────────────────────────────────────

    def _solve_syllogism(self, task: str, facts: list[str]) -> SolverResult:
        """
        Handles syllogisms like:
        "All mammals breathe. Whales are mammals. Do whales breathe?"
        """
        steps = []

        # Extract premises and conclusion
        sentences = re.split(r'[.?!]\s*', task)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return SolverResult(
                status=SolverStatus.UNKNOWN,
                answer="Insufficient premises for syllogistic reasoning.",
                confidence=0.3,
            )

        # Build a simple forward-chaining reasoner
        knowledge_base: dict[str, set[str]] = {}  # category -> properties/supersets
        entities: dict[str, set[str]] = {}  # entity -> categories

        for sentence in sentences[:-1]:  # Last is the question
            self._parse_and_add_fact(sentence, knowledge_base, entities, steps)

        # The question is the last sentence
        question = sentences[-1]
        answer, confidence = self._answer_syllogistic_query(
            question, knowledge_base, entities, steps
        )

        return SolverResult(
            status=SolverStatus.SAT if answer else SolverStatus.UNSAT,
            answer=answer,
            proof_steps=steps,
            confidence=confidence,
            reasoning="Forward-chaining syllogistic reasoner",
        )

    def _parse_and_add_fact(
        self,
        sentence: str,
        kb: dict[str, set[str]],
        entities: dict[str, set[str]],
        steps: list[str],
    ) -> None:
        """Parse a natural language fact and add to KB or entity store.

        Handles patterns like:
          "All mammals breathe air"       -> rule:  mammals -> breathe air
          "All animals are living things" -> rule:  animals -> living things
          "All living things need energy" -> rule:  living things -> need energy
          "Dogs are animals"              -> fact:  dogs is-a animals
          "Fido is a dog"                 -> fact:  fido is-a dog
        """
        s = sentence.lower().strip()

        # ── Universal rules: All/Every <subject> <copula/verb> <predicate> ──
        # First try with explicit copula (are/is) to correctly split subject/pred
        m = re.match(r"(?:all|every)\s+(.+?)\s+(?:are|is)\s+(?:a\s+|an\s+)?(.+)", s)
        if m:
            subj = m.group(1).strip()
            pred = m.group(2).strip()
            kb.setdefault(subj, set()).add(pred)
            steps.append(f"Rule: \u2200x. {subj}(x) \u2192 {pred}(x)")
            return

        # Then try without copula: "All X verb Y"
        # Split at first verb word (non-article) after subject
        m = re.match(r"(?:all|every)\s+(.+?)\s+(\w+(?:\s+\w+)*?)$", s)
        if m:
            # Heuristic: subject is everything up to a known action verb or 2-word noun
            full = s[len("all "):].strip() if s.startswith("all") else s[len("every "):].strip()
            words = full.split()
            # Find best split: try each word boundary
            # If word[i] looks like a verb (not part of subject noun), split there
            VERBS = {"breathe","need","have","can","do","does","eat","live","grow",
                     "make","take","produce","contain","require","use"}
            split_at = None
            for i, w in enumerate(words):
                if w in VERBS and i > 0:
                    split_at = i
                    break
            if split_at is not None:
                subj = " ".join(words[:split_at])
                pred = " ".join(words[split_at:])
            else:
                # Default: first word is subject, rest is predicate
                subj = words[0]
                pred = " ".join(words[1:])
            if subj and pred:
                kb.setdefault(subj, set()).add(pred)
                steps.append(f"Rule: \u2200x. {subj}(x) \u2192 {pred}(x)")
            return

        # ── Instance/subclass facts: <entity> are/is [a/an] <category> ──
        m = re.match(r"(.+?)\s+(?:are|is)\s+(?:a\s+|an\s+)?(.+)", s)
        if m:
            entity = m.group(1).strip()
            category = m.group(2).strip()
            # Only short subjects are entities (avoid mis-parsing rules)
            if len(entity.split()) <= 3:
                entities.setdefault(entity, set()).add(category)
                steps.append(f"Fact: {category}({entity})")

    def _answer_syllogistic_query(
        self,
        question: str,
        kb: dict[str, set[str]],
        entities: dict[str, set[str]],
        steps: list[str],
    ) -> tuple[str, float]:
        q = question.lower().strip().rstrip("?")

        # "Do/Does X Y?" or "Is X a Y?"
        m = re.match(r"(?:do|does|is|are)\s+([\w]+)\s+(?:a\s+)?([\w][\w ]*)", q)
        if not m:
            return "Unable to parse question.", 0.4

        entity = m.group(1).strip()
        target_property = m.group(2).strip().rstrip("?").strip()

        # BFS through knowledge base
        categories = set(entities.get(entity, []))
        visited = set()
        frontier = list(categories)
        derivation_path: list[str] = [entity]

        while frontier:
            current = frontier.pop()
            if current in visited:
                continue
            visited.add(current)

            if current == target_property:
                steps.append(f"Conclusion: {target_property}({entity}) ✓")
                path_str = " → ".join(derivation_path + [target_property])
                return (
                    f"Yes, {entity} {target_property}. "
                    f"(Derived via chain: {path_str})"
                ), 0.95

            # Expand via rules
            for implied in kb.get(current, []):
                if implied not in visited:
                    frontier.append(implied)
                    steps.append(f"Chain: {current} → {implied}")
                    derivation_path.append(current)

        return f"Cannot determine if {entity} {target_property} from given premises.", 0.5

    # ─── Arithmetic Constraint Solving ───────────────────────────────────────

    def _solve_arithmetic(self, task: str, facts: list[str]) -> SolverResult:
        if self._z3_available:
            return self._z3_arithmetic(task)
        return self._fallback_arithmetic(task)

    def _z3_arithmetic(self, task: str) -> SolverResult:
        try:
            import z3
            steps = []

            # Try to extract a simple linear equation: "solve 2x + 3 = 7"
            m = re.search(r"(\d+)\s*\*?\s*x\s*([+-]\s*\d+)?\s*=\s*(\d+)", task.lower())
            if m:
                coeff = int(m.group(1))
                const = int(m.group(2).replace(" ", "")) if m.group(2) else 0
                rhs = int(m.group(3))

                x = z3.Real("x")
                solver = z3.Solver()
                solver.set("timeout", self.timeout_ms)
                solver.add(coeff * x + const == rhs)

                steps.append(f"Z3 constraint: {coeff}*x + {const} == {rhs}")

                if solver.check() == z3.sat:
                    model = solver.model()
                    val = model[x]
                    steps.append(f"Z3 SAT: x = {val}")
                    return SolverResult(
                        status=SolverStatus.SAT,
                        answer=f"x = {val}",
                        proof_steps=steps,
                        model={"x": str(val)},
                        confidence=1.0,
                        reasoning="Z3 SMT solver",
                    )

            # Try pure arithmetic expression
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s]", "", task)
            if expr.strip():
                result = eval(expr.strip(), {"__builtins__": {}})  # safe-ish for digits only
                return SolverResult(
                    status=SolverStatus.SAT,
                    answer=str(result),
                    proof_steps=[f"Evaluated: {expr.strip()} = {result}"],
                    confidence=1.0,
                    reasoning="Direct arithmetic evaluation",
                )

        except Exception as e:
            logger.warning(f"[Z3Arithmetic] Error: {e}")

        return SolverResult(
            status=SolverStatus.UNKNOWN,
            answer="Could not solve arithmetic problem.",
            confidence=0.3,
        )

    def _fallback_arithmetic(self, task: str) -> SolverResult:
        """Safe fallback arithmetic evaluator."""
        try:
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s]", "", task)
            if expr.strip():
                result = eval(expr.strip(), {"__builtins__": {}})
                return SolverResult(
                    status=SolverStatus.SAT,
                    answer=f"= {result}",
                    proof_steps=[f"Evaluated: {expr.strip()} = {result}"],
                    confidence=0.9,
                )
        except Exception:
            pass
        return SolverResult(
            status=SolverStatus.UNKNOWN,
            answer="Could not evaluate expression.",
            confidence=0.3,
        )

    # ─── Propositional Logic ─────────────────────────────────────────────────

    def _solve_propositional(self, task: str, facts: list[str]) -> SolverResult:
        if not self._z3_available:
            return SolverResult(
                status=SolverStatus.UNSUPPORTED,
                answer="Z3 not available for propositional logic.",
                confidence=0.0,
            )

        try:
            import z3
            steps = ["Translating to propositional logic..."]

            # Extract "if P then Q" style rules
            p, q = None, None
            m = re.search(r"if\s+(.+?)\s+then\s+(.+?)[\.\?]", task.lower())
            if m:
                p_text, q_text = m.group(1).strip(), m.group(2).strip()
                P = z3.Bool("P")
                Q = z3.Bool("Q")
                solver = z3.Solver()
                solver.add(z3.Implies(P, Q))
                solver.add(P)

                steps.append(f"Rule: {p_text} → {q_text}")
                steps.append("Premise: P is True")

                if solver.check() == z3.sat:
                    model = solver.model()
                    q_val = model[Q]
                    steps.append(f"Z3: Q = {q_val}")
                    return SolverResult(
                        status=SolverStatus.SAT,
                        answer=f"Given '{p_text}', it follows that '{q_text}' (modus ponens).",
                        proof_steps=steps,
                        confidence=0.95,
                        reasoning="Z3 propositional satisfiability",
                    )

        except Exception as e:
            logger.warning(f"[PropositionalSolver] Error: {e}")

        return SolverResult(
            status=SolverStatus.UNKNOWN,
            answer="Could not resolve propositional query.",
            confidence=0.4,
        )

    # ─── Generic Logic Fallback ───────────────────────────────────────────────

    def _solve_generic_logic(self, task: str, facts: list[str]) -> SolverResult:
        return SolverResult(
            status=SolverStatus.UNKNOWN,
            answer="Task type not recognised by symbolic solver. Routing to LLM.",
            confidence=0.3,
            reasoning="No matching symbolic pattern found",
        )
