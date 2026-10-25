"""Human-readable summary derived entirely from the canonical result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plasma_global._result_common import (
    RESULT_FORMAT,
    RESULT_FORMAT_VERSION,
    ensure_parent,
    plain_value,
)
from plasma_global.core.result import SimulationResult, series_unit, to_plain_mapping


def build_summary(result: SimulationResult) -> dict[str, Any]:
    """Build the fixed summary from result-declared quantities only."""

    metadata = to_plain_mapping(result.metadata)
    final_names = metadata.get("summary_series", ())
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
        "assessment": plain_value(metadata.get("assessment")),
        "solver": to_plain_mapping(result.solver_stats),
        **(
            {
                "probes": {
                    name: {
                        "value": result.final_value(name) if result.n_times else None,
                        "unit": series_unit(name, metadata.get("series_units", {})),
                        **definition,
                    }
                    for name, definition in metadata.get("probes", {}).items()
                }
            }
            if metadata.get("probes")
            else {}
        ),
    }


def write_series_yaml(path: str | Path, result: SimulationResult) -> Path:
    """Describe exported columns without requiring an HDF5 reader."""
    from plasma_global._result_csv import _result_csv_columns

    units = result.metadata.get("series_units", {})
    columns = [{"column": "time_s", "series": "time_s", "unit": "s"}]
    for column, _, _ in _result_csv_columns(result):
        name = column.partition(":")[2]
        columns.append(
            {
                "column": column,
                "series": name,
                "unit": series_unit(name, units),
                **dict(result.metadata.get("probes", {}).get(name, {})),
            }
        )
    payload = {
        "columns": columns,
        "composition_basis": result.metadata.get("composition_basis"),
        "event_integration": plain_value(result.metadata.get("event_integration")),
        "observable_time_convention": result.metadata.get("observable_time_convention"),
        "model_geometry": plain_value(result.metadata.get("model_geometry")),
    }
    target = Path(path)
    ensure_parent(target)
    with target.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)
    return target


def write_summary_yaml(
    path: str | Path,
    result: SimulationResult,
) -> Path:
    target = Path(path)
    ensure_parent(target)
    with target.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(
            build_summary(result),
            stream,
            sort_keys=False,
            allow_unicode=True,
        )
    return target


__all__ = ["build_summary", "write_summary_yaml"]
