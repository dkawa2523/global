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
class CaseAuditReport:
    """Detailed, read-only audit of one compiled schema-v3 case."""

    issues: tuple[AuditIssue, ...]
    model_ids: Mapping[str, Any]
    provenance: Mapping[str, Any]
    inventory: Mapping[str, int]
    conservation_max_abs_residual: Mapping[str, float]
    simulation: Mapping[str, Any] = field(default_factory=dict)
    classification: str = "standard"
    production_qualified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        for name in (
            "model_ids",
            "provenance",
            "inventory",
            "conservation_max_abs_residual",
            "simulation",
        ):
            value = dict(getattr(self, name))
            object.__setattr__(self, name, MappingProxyType(value))
        if self.classification not in {"standard", "experimental"}:
            raise ValueError("classification must be standard or experimental")
        if self.production_qualified and (
            self.classification != "standard" or not self.passed
        ):
            raise ValueError("production_qualified requires a passing standard audit")

    @property
    def passed(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "classification": self.classification,
            "production_qualified": self.production_qualified,
            "model_ids": to_plain_mapping(self.model_ids),
            "provenance": to_plain_mapping(self.provenance),
            "inventory": to_plain_mapping(self.inventory),
            "conservation_max_abs_residual": to_plain_mapping(
                self.conservation_max_abs_residual
            ),
            "simulation": to_plain_mapping(self.simulation),
            "issues": [asdict(issue) for issue in self.issues],
        }
