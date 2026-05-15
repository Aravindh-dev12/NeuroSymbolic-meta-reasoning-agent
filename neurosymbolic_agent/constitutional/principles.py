"""
constitutional/principles.py — Registry of constitutional principles.
Loads principles from YAML and provides a queryable interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Principle:
    id: str
    name: str
    description: str
    check_type: str
    severity: Severity
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Principle":
        severity = Severity(d.get("severity", "MEDIUM"))
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description", ""),
            check_type=d["check_type"],
            severity=severity,
            metadata={k: v for k, v in d.items() if k not in ("id", "name", "description", "check_type", "severity")},
        )


class PrinciplesRegistry:
    """Manages constitutional principles."""

    def __init__(self, principles: list[dict] | None = None):
        self._principles: dict[str, Principle] = {}
        if principles:
            for p in principles:
                principle = Principle.from_dict(p)
                self._principles[principle.id] = principle
        logger.info(f"[PrinciplesRegistry] Loaded {len(self._principles)} principles")

    def get(self, principle_id: str) -> Principle | None:
        return self._principles.get(principle_id)

    def all(self) -> list[Principle]:
        return list(self._principles.values())

    def by_severity(self, severity: Severity) -> list[Principle]:
        return [p for p in self._principles.values() if p.severity == severity]

    def critical(self) -> list[Principle]:
        return self.by_severity(Severity.CRITICAL)

    def __len__(self) -> int:
        return len(self._principles)
