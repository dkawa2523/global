"""Pure validation of an already computed simulation result."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from plasma_global._audit_types import AuditIssue, AuditReport
from plasma_global.core.result import SimulationResult


def _series_or_issue(
    result: SimulationResult,
    name: str,
    issues: list[AuditIssue],
    *,
    code: str,
) -> np.ndarray | None:
    try:
        return result.series(name)
    except KeyError:
        issues.append(
            AuditIssue(
                "ERROR", code, f"Required result series {name!r} is missing.", name
            )
        )
        return None


def _validate_tolerances(
    conservation_tolerances: Mapping[str, float] | None,
    default_conservation_tolerance: float | None,
) -> tuple[dict[str, float], float | None]:
    tolerances = dict(conservation_tolerances or {})
    if any(value < 0.0 or not np.isfinite(value) for value in tolerances.values()):
        raise ValueError("conservation tolerances must be finite and non-negative")
    default = default_conservation_tolerance
    if default is not None and (default < 0.0 or not np.isfinite(default)):
        raise ValueError(
            "default conservation tolerance must be finite and non-negative"
        )
    return tolerances, default


def _result_health_issues(result: SimulationResult) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    if not result.status.success:
        issues.append(
            AuditIssue(
                "ERROR",
                "SIMULATION_NOT_SUCCESSFUL",
                f"Simulation status is {result.status.code!r}: {result.status.message}",
                "status",
            )
        )
    if result.n_times == 0:
        issues.append(
            AuditIssue(
                "WARNING",
                "EMPTY_RESULT",
                "Simulation result contains no time points.",
                "time_s",
            )
        )

    return issues


def _series_issues(
    result: SimulationResult,
    required_series: Iterable[str],
    nonnegative_series: Iterable[str],
    negative_tolerance: float,
) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for name in dict.fromkeys(required_series):
        _series_or_issue(result, name, issues, code="REQUIRED_SERIES_MISSING")

    for name in dict.fromkeys(nonnegative_series):
        values = _series_or_issue(
            result, name, issues, code="NONNEGATIVE_SERIES_MISSING"
        )
        if values is None or values.size == 0:
            continue
        minimum = float(np.min(values))
        if minimum < -negative_tolerance:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "NEGATIVE_SERIES_VALUE",
                    (
                        f"Series {name!r} has minimum {minimum}, "
                        f"below allowed {-negative_tolerance}."
                    ),
                    name,
                )
            )

    return issues


def _residual_maximum(
    result: SimulationResult,
    name: str,
    raw_values: Any,
) -> float | AuditIssue:
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"conservation residual {name!r} must be numeric") from exc
    if values.ndim == 1 and values.shape != (result.n_times,):
        return AuditIssue(
            "ERROR",
            "CONSERVATION_RESIDUAL_SHAPE",
            (
                f"Conservation residual {name!r} must be scalar or have shape "
                f"({result.n_times},), got {values.shape}."
            ),
            name,
        )
    if values.ndim not in {0, 1}:
        return AuditIssue(
            "ERROR",
            "CONSERVATION_RESIDUAL_SHAPE",
            f"Conservation residual {name!r} must be scalar or one-dimensional.",
            name,
        )
    if not np.all(np.isfinite(values)):
        return AuditIssue(
            "ERROR",
            "CONSERVATION_RESIDUAL_NONFINITE",
            f"Conservation residual {name!r} contains non-finite values.",
            name,
        )
    return float(np.max(np.abs(values))) if values.size else 0.0


def _conservation_audit(
    result: SimulationResult,
    conservation_residuals: Mapping[str, Any] | None,
    tolerances: Mapping[str, float],
    default_tolerance: float | None,
) -> tuple[dict[str, float], list[AuditIssue]]:
    maxima: dict[str, float] = {}
    issues: list[AuditIssue] = []
    for raw_name, raw_values in dict(conservation_residuals or {}).items():
        name = raw_name.strip()
        if not name:
            raise ValueError("conservation residual names must not be empty")
        maximum = _residual_maximum(result, name, raw_values)
        if isinstance(maximum, AuditIssue):
            issues.append(maximum)
            continue
        maxima[name] = maximum
        tolerance = tolerances.get(name, default_tolerance)
        if tolerance is not None and maximum > tolerance:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "CONSERVATION_RESIDUAL_EXCEEDED",
                    (
                        f"Conservation residual {name!r} maximum {maximum} "
                        f"exceeds tolerance {tolerance}."
                    ),
                    name,
                )
            )
    return maxima, issues


def audit_result(
    result: SimulationResult,
    *,
    conservation_residuals: Mapping[str, Any] | None = None,
    conservation_tolerances: Mapping[str, float] | None = None,
    default_conservation_tolerance: float | None = None,
    required_series: Iterable[str] = (),
    nonnegative_series: Iterable[str] = (),
    negative_tolerance: float = 0.0,
) -> AuditReport:
    """Audit a result without modifying it or performing I/O.

    Conservation residuals are explicit because their definitions are model-specific.
    They may be scalars or arrays with one value per result time.
    """

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    if negative_tolerance < 0.0 or not np.isfinite(negative_tolerance):
        raise ValueError("negative_tolerance must be finite and non-negative")
    tolerances, default_tolerance = _validate_tolerances(
        conservation_tolerances,
        default_conservation_tolerance,
    )

    issues = _result_health_issues(result)
    issues.extend(
        _series_issues(result, required_series, nonnegative_series, negative_tolerance)
    )
    maxima, conservation_issues = _conservation_audit(
        result,
        conservation_residuals,
        tolerances,
        default_tolerance,
    )
    issues.extend(conservation_issues)
    return AuditReport(issues=tuple(issues), conservation_max_abs_residual=maxima)
