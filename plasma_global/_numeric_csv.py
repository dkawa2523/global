"""Strict numeric CSV input with optional explicit physical-column bindings."""

from __future__ import annotations

import csv
from collections.abc import Container, Mapping
from pathlib import Path

import numpy as np


def _validate_header(source: Path, header: list[str]) -> None:
    if not header or any(not key.strip() for key in header):
        raise ValueError(f"{source}: column names must be nonempty")
    if len(set(header)) != len(header):
        raise ValueError(f"{source}: duplicate columns in CSV")


def _bindings(
    source: Path,
    header: list[str],
    columns: Mapping[str, str] | None,
    allowed: Container[str] | None,
) -> dict[str, int]:
    _validate_header(source, header)
    selected = dict(columns) if columns else {key: key for key in header}
    if missing := set(selected.values()) - set(header):
        raise ValueError(f"{source}: missing source columns {sorted(missing)}")
    if allowed is not None and (
        unknown := [key for key in selected if key not in allowed]
    ):
        raise ValueError(f"{source}: unknown columns {unknown}")
    return {key: header.index(value) for key, value in selected.items()}


def _number(source: Path, line: int, column: str, value: str) -> float:
    if not value.strip():
        raise ValueError(
            f"{source}:{line}:{column}: missing values; numeric value required"
        )
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{source}:{line}:{column}: a numeric value is required"
        ) from exc
    if not np.isfinite(number):
        raise ValueError(f"{source}:{line}:{column}: value must be finite")
    return number


def read_numeric_csv(
    path: str | Path,
    *,
    columns: Mapping[str, str] | None = None,
    allowed: Container[str] | None = None,
) -> dict[str, np.ndarray]:
    source = Path(path)
    rows: list[list[float]] = []
    try:
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            empty: list[str] = []
            header = next(reader, empty)
            selected = _bindings(source, header, columns, allowed)
            for line, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    reason = (
                        "extra CSV fields"
                        if len(row) > len(header)
                        else "missing values"
                    )
                    raise ValueError(f"{source}:{line}: {reason}; header mismatch")
                rows.append(
                    [
                        _number(source, line, header[index], row[index])
                        for index in selected.values()
                    ]
                )
    except csv.Error as exc:
        raise ValueError(f"{source}: malformed CSV: {exc}") from exc
    if not rows:
        raise ValueError(f"{source}: CSV contains no data rows")
    values = np.array(rows, dtype=float)
    return {key: values[:, index] for index, key in enumerate(selected)}
