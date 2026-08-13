"""Reversible numeric CSV codec for explicit result export."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global._result_common import ensure_parent
from plasma_global.core.result import SimulationResult, SimulationStatus

_CSV_TIME = "time_s"
_CSV_STATE = "state:"
_CSV_OBSERVABLE = "observable:"
_CSV_SCALAR_OBSERVABLE = "observable_scalar:"


def _result_csv_columns(
    result: SimulationResult,
) -> list[tuple[str, np.ndarray, bool]]:
    columns = [
        (f"{_CSV_STATE}{label}", result.state[:, index], False)
        for index, label in enumerate(result.state_labels)
    ]
    for name, values in result.observables.items():
        prefix = _CSV_SCALAR_OBSERVABLE if values.ndim == 0 else _CSV_OBSERVABLE
        columns.append((f"{prefix}{name}", values, values.ndim == 0))
    return columns


def write_result_csv(path: str | Path, result: SimulationResult) -> Path:
    """Export a canonical result to one reversible numeric CSV."""

    if result.n_times == 0 and any(
        values.ndim == 0 for values in result.observables.values()
    ):
        raise ValueError("CSV cannot preserve scalar observables without time rows")
    target = Path(path)
    ensure_parent(target)
    columns = _result_csv_columns(result)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([_CSV_TIME, *(name for name, _values, _scalar in columns)])
        for row_index, time_s in enumerate(result.time_s):
            row: list[float] = [float(time_s)]
            for _name, values, scalar in columns:
                row.append(float(values) if scalar else float(values[row_index]))
            writer.writerow(row)
    return target


def _read_csv_rows(source: Path) -> tuple[list[str], list[list[str]]]:
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Result CSV is empty: {source}") from exc
        return header, list(reader)


def _numeric_csv_values(
    source: Path, header: list[str], rows: list[list[str]]
) -> np.ndarray:
    if not header or header[0] != _CSV_TIME:
        raise ValueError(f"Result CSV first column must be {_CSV_TIME!r}")
    if len(set(header)) != len(header):
        raise ValueError("Result CSV column names must be unique")
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Result CSV rows do not match its header")
    try:
        values = np.asarray(
            [[float(cell) for cell in row] for row in rows], dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Result CSV contains a non-numeric cell: {source}") from exc
    if not rows:
        return np.empty((0, len(header)), dtype=float)
    return values


def _decode_csv_columns(
    header: list[str], values: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
    time_s = values[:, 0]
    labels: list[str] = []
    state_columns: list[np.ndarray] = []
    observables: dict[str, np.ndarray] = {}
    for index, column in enumerate(header[1:], start=1):
        data = values[:, index]
        if column.startswith(_CSV_STATE):
            labels.append(column.removeprefix(_CSV_STATE))
            state_columns.append(data)
            continue
        if column.startswith(_CSV_SCALAR_OBSERVABLE):
            name = column.removeprefix(_CSV_SCALAR_OBSERVABLE)
            if data.size == 0 or not np.allclose(data, data[0], equal_nan=True):
                raise ValueError(f"Invalid scalar observable column {name!r}")
            observables[name] = np.asarray(data[0], dtype=float)
            continue
        if column.startswith(_CSV_OBSERVABLE):
            observables[column.removeprefix(_CSV_OBSERVABLE)] = data
            continue
        raise ValueError(f"Unsupported result CSV column {column!r}")
    state = (
        np.column_stack(state_columns)
        if state_columns
        else np.empty((time_s.size, 0), dtype=float)
    )
    return state, tuple(labels), observables


def read_result_csv(
    path: str | Path,
    *,
    status: SimulationStatus | None = None,
    solver_stats: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Read a CSV produced by :func:`write_result_csv`."""

    source = Path(path)
    header, rows = _read_csv_rows(source)
    values = _numeric_csv_values(source, header, rows)
    state, state_labels, observables = _decode_csv_columns(header, values)
    return SimulationResult(
        time_s=values[:, 0],
        state=state,
        state_labels=state_labels,
        observables=observables,
        status=status
        or SimulationStatus(
            success=True,
            code="imported_csv",
            message=f"Loaded from {source.name}",
        ),
        solver_stats=solver_stats or {},
        metadata=metadata or {},
    )


__all__ = ["read_result_csv", "write_result_csv"]
