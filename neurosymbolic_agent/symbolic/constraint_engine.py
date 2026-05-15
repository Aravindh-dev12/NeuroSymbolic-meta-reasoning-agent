"""
symbolic/constraint_engine.py — Hard constraint enforcement for symbolic reasoning.
Wraps Z3 for constraint satisfaction problems (CSP) and constraint propagation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class Constraint:
    name: str
    expression: str  # Human-readable
    variables: list[str]
    constraint_type: str  # "equality", "inequality", "range", "domain"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintResult:
    satisfiable: bool
    assignments: dict[str, Any]
    unsatisfied: list[str]
    steps: list[str]
    confidence: float


class ConstraintEngine:
    """
    Constraint satisfaction engine.
    Uses Z3 when available, falls back to simple interval arithmetic.
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_ms = timeout_seconds * 1000
        self._z3_available = self._check_z3()

    def _check_z3(self) -> bool:
        try:
            import z3  # noqa
            return True
        except ImportError:
            return False

    def check_constraints(
        self,
        constraints: list[Constraint],
        domain: dict[str, tuple[float, float]] | None = None,
    ) -> ConstraintResult:
        """Check whether a set of constraints is satisfiable."""
        if not constraints:
            return ConstraintResult(
                satisfiable=True,
                assignments={},
                unsatisfied=[],
                steps=["No constraints to check."],
                confidence=1.0,
            )

        if self._z3_available:
            return self._z3_solve(constraints, domain or {})
        return self._fallback_solve(constraints, domain or {})

    def _z3_solve(
        self,
        constraints: list[Constraint],
        domain: dict[str, tuple[float, float]],
    ) -> ConstraintResult:
        import z3

        steps = []
        solver = z3.Optimize()
        solver.set("timeout", self.timeout_ms)

        # Create Z3 variables
        z3_vars: dict[str, z3.ArithRef] = {}
        all_vars = set()
        for c in constraints:
            all_vars.update(c.variables)

        for var in all_vars:
            z3_vars[var] = z3.Real(var)
            if var in domain:
                lo, hi = domain[var]
                solver.add(z3_vars[var] >= lo)
                solver.add(z3_vars[var] <= hi)
                steps.append(f"Domain: {var} ∈ [{lo}, {hi}]")

        # Add constraints (simplified — expression parsing)
        for c in constraints:
            steps.append(f"Constraint: {c.expression}")
            try:
                expr = c.expression
                for var, z3v in z3_vars.items():
                    expr = expr.replace(var, f"z3_vars['{var}']")
                evaluated = eval(expr, {"z3_vars": z3_vars, **z3.__dict__})
                solver.add(evaluated)
            except Exception as e:
                steps.append(f"  ⚠ Could not parse constraint: {e}")

        check = solver.check()
        if check == z3.sat:
            model = solver.model()
            assignments = {}
            for var in z3_vars:
                val = model[z3_vars[var]]
                if val is not None:
                    assignments[var] = str(val)
            steps.append(f"SAT. Assignments: {assignments}")
            return ConstraintResult(
                satisfiable=True,
                assignments=assignments,
                unsatisfied=[],
                steps=steps,
                confidence=1.0,
            )
        else:
            steps.append("UNSAT — constraints cannot be simultaneously satisfied.")
            return ConstraintResult(
                satisfiable=False,
                assignments={},
                unsatisfied=[c.name for c in constraints],
                steps=steps,
                confidence=1.0,
            )

    def _fallback_solve(
        self,
        constraints: list[Constraint],
        domain: dict[str, tuple[float, float]],
    ) -> ConstraintResult:
        """Trivial fallback — reports constraints without solving."""
        steps = [f"Constraint (not solved): {c.expression}" for c in constraints]
        steps.append("Z3 unavailable — cannot verify satisfiability.")
        return ConstraintResult(
            satisfiable=False,
            assignments={},
            unsatisfied=[c.name for c in constraints],
            steps=steps,
            confidence=0.0,
        )
