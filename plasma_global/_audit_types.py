"""Immutable report values shared by the audit pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any

from plasma_global.core.result import to_plain_mapping


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """One actionable problem found by an audit."""

    level: str
    code: str
    message: str
    quantity: str = ""

    def __post_init__(self) -> None:
        level = str(self.level).upper()
        if level not in {"ERROR", "WARNING"}:
            raise ValueError(f"Unsupported audit issue level {self.level!r}")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "code", str(self.code).strip())
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "quantity", str(self.quantity))


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Read-only checks performed on an existing simulation result."""

    issues: tuple[AuditIssue, ...] = ()
    conservation_max_abs_residual: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        maxima = {
            str(name): float(value)
            for name, value in dict(self.conservation_max_abs_residual or {}).items()
        }
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(
            self, "conservation_max_abs_residual", MappingProxyType(maxima)
        )

    @property
    def passed(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)

    @property
    def max_conservation_residual(self) -> float | None:
        if not self.conservation_max_abs_residual:
            return None
        return max(self.conservation_max_abs_residual.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_conservation_residual": self.max_conservation_residual,
            "conservation_max_abs_residual": dict(self.conservation_max_abs_residual),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class CaseAssessment:
    """Supported feature envelope and actionable compile-time warnings."""

    classification: str
    issues: tuple[AuditIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class CaseAuditReport:
    """Case assessment, provenance, and completed simulation information."""

    assessment: CaseAssessment
    model_ids: Mapping[str, Any]
    provenance: Mapping[str, Any]
    inventory: Mapping[str, int]
    simulation: Mapping[str, Any]

    @property
    def issues(self) -> tuple[AuditIssue, ...]:
        return self.assessment.issues

    @property
    def classification(self) -> str:
        return self.assessment.classification

    @property
    def passed(self) -> bool:
        return bool(self.simulation["status"]["success"]) and not any(
            issue.level == "ERROR" for issue in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            **self.assessment.to_dict(),
            "model_ids": to_plain_mapping(self.model_ids),
            "provenance": to_plain_mapping(self.provenance),
            "inventory": dict(self.inventory),
            "simulation": to_plain_mapping(self.simulation),
        }
