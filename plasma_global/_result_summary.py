"""Result-level audit interpretation and human-readable summary output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plasma_global._result_common import (
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    ensure_parent,
    plain_value,
)
from plasma_global.audit import AuditReport, audit_result
from plasma_global.core.result import SimulationResult, to_plain_mapping


def _string_sequence(metadata: Mapping[str, Any], name: str) -> tuple[str, ...]:
    raw = metadata.get(name, ())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError(f"result metadata {name} must be a sequence of names")
    return tuple(dict.fromkeys(str(value) for value in raw))


def _optional_dict(values: Mapping[str, Any], name: str, error: str) -> dict[str, Any]:
    value = values.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(error)
    return value


def _optional_dict_or_none(
    values: Mapping[str, Any], name: str, error: str
) -> dict[str, Any] | None:
    value = values.get(name)
    if value is not None and not isinstance(value, dict):
        raise TypeError(error)
    return value


def audit_from_metadata(result: SimulationResult) -> AuditReport:
    """Audit using only constraints stored with the result metadata."""

    metadata = to_plain_mapping(result.metadata)
    conservation_names = _string_sequence(metadata, "conservation_observables")
    residuals = {name: result.series(name) for name in conservation_names}
    provenance = _optional_dict(
        metadata,
        "provenance",
        "result metadata provenance must be a mapping",
    )
    runtime = _optional_dict(
        provenance,
        "runtime_diagnostics",
        "result provenance runtime_diagnostics must be a mapping",
    )
    stored_maxima = _optional_dict(
        runtime,
        "conservation_max_abs_residual",
        "runtime diagnostic conservation_max_abs_residual must be a mapping",
    )
    for name, value in stored_maxima.items():
        residuals.setdefault(str(name), np.asarray(float(value)))
    tolerances = _optional_dict_or_none(
        metadata,
        "conservation_tolerances",
        "result metadata conservation_tolerances must be a mapping",
    )
    return audit_result(
        result,
        conservation_residuals=residuals,
        conservation_tolerances=tolerances,
        default_conservation_tolerance=metadata.get("default_conservation_tolerance"),
    )


def build_summary(
    result: SimulationResult, *, audit: AuditReport | None = None
) -> dict[str, Any]:
    """Build the fixed summary from result-declared quantities only."""

    metadata = to_plain_mapping(result.metadata)
    final_names = _string_sequence(metadata, "summary_series")
    report = audit or audit_from_metadata(result)
    return {
        "format": RESULT_FORMAT,
        "format_version": RESULT_FORMAT_VERSION,
        "status": {
            "success": result.status.success,
            "code": result.status.code,
            "message": result.status.message,
        },
        "time": {
            "count": result.n_times,
            "start_s": float(result.time_s[0]) if result.n_times else None,
            "end_s": float(result.time_s[-1]) if result.n_times else None,
        },
        "model_ids": plain_value(metadata.get("model_ids", {})),
        "final": {name: result.final_value(name) for name in final_names},
        "conservation": {
            "max_abs_residual": report.max_conservation_residual,
            "max_abs_residual_by_quantity": dict(report.conservation_max_abs_residual),
        },
        "audit": {"passed": report.passed, "issue_count": len(report.issues)},
        "solver": to_plain_mapping(result.solver_stats),
    }


def write_summary_yaml(
    path: str | Path,
    result: SimulationResult,
    *,
    audit: AuditReport | None = None,
) -> Path:
    target = Path(path)
    ensure_parent(target)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            build_summary(result, audit=audit),
            stream,
            sort_keys=False,
            allow_unicode=True,
        )
    return target


__all__ = ["audit_from_metadata", "build_summary", "write_summary_yaml"]
