"""Prescribed electron-density profiles with an explicit interpolation policy."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


def _validated_profile_time(value: np.ndarray) -> np.ndarray:
    time = np.array(value, dtype=float, copy=True)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("profile time_s must be a nonempty 1D array")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("profile time_s must be finite and strictly increasing")
    time.setflags(write=False)
    return time


def _validated_profile_densities(
    values: Mapping[str, np.ndarray], time_shape: tuple[int, ...]
) -> Mapping[str, np.ndarray]:
    if not values:
        raise ValueError("at least one zone density series is required")
    densities: dict[str, np.ndarray] = {}
    for zone_id, raw_density in values.items():
        key = str(zone_id)
        density = np.array(raw_density, dtype=float, copy=True)
        if not key:
            raise ValueError("profile zone IDs must not be empty")
        if density.shape != time_shape:
            raise ValueError(f"density profile for {key!r} has the wrong shape")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError(
                f"density profile for {key!r} must be finite and nonnegative"
            )
        density.setflags(write=False)
        densities[key] = density
    return MappingProxyType(densities)


def _read_profile_csv(
    source: Path,
) -> tuple[tuple[str, ...], list[dict[str | None, Any]]]:
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"electron profile {source} is malformed CSV: {exc}") from exc
    if len(set(columns)) != len(columns):
        raise ValueError(f"electron profile {source} has duplicate columns")
    if "time_s" not in columns:
        raise ValueError(f"electron profile {source} requires a time_s column")
    if not rows:
        raise ValueError(f"electron profile {source} contains no data rows")
    return columns, rows


def _selected_profile_columns(
    source: Path,
    columns: tuple[str, ...],
    zone_columns: Mapping[str, str] | None,
) -> dict[str, str]:
    if zone_columns is None:
        value_columns = tuple(column for column in columns if column != "time_s")
        selected = (
            {"*": "electron_density_m3"}
            if value_columns == ("electron_density_m3",)
            else {column: column for column in value_columns}
        )
    else:
        selected = {str(zone): str(column) for zone, column in zone_columns.items()}
    missing = sorted(set(selected.values()) - set(columns))
    if missing:
        raise ValueError(
            f"electron profile {source} is missing columns: {', '.join(missing)}"
        )
    if zone_columns is not None:
        extra = sorted(set(columns) - {"time_s", *selected.values()})
        if extra:
            raise ValueError(
                f"electron profile {source} has extra columns: {', '.join(extra)}"
            )
    return selected


def _validate_profile_rows(
    source: Path,
    columns: tuple[str, ...],
    rows: list[dict[str | None, Any]],
) -> None:
    for line, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(
                f"electron profile {source}:{line} contains extra CSV fields"
            )
        missing_values = [
            column
            for column in columns
            if row.get(column) is None or not str(row[column]).strip()
        ]
        if missing_values:
            raise ValueError(
                f"electron profile {source}:{line} has missing values for: "
                f"{', '.join(missing_values)}"
            )


def _numeric_profile_data(
    source: Path,
    rows: list[dict[str | None, Any]],
    selected: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        time = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
        density = {
            zone: np.asarray([float(row[column]) for row in rows], dtype=float)
            for zone, column in selected.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"electron profile {source} must contain numeric data"
        ) from exc
    return time, density


@dataclass(frozen=True, slots=True)
class PrescribedElectronProfile:
    """Immutable electron density time series keyed by zone.

    A ``"*"`` entry is a deliberate fallback for every zone.  Negative or
    non-finite density input is rejected instead of being silently clipped.
    """

    time_s: np.ndarray
    density_m3_by_zone: Mapping[str, np.ndarray]
    interpolation: Literal["linear", "previous"] = "linear"
    bounds: Literal["hold", "error"] = "hold"
    source: Path | None = None

    def __post_init__(self) -> None:
        time = _validated_profile_time(self.time_s)
        if self.interpolation not in {"linear", "previous"}:
            raise ValueError("interpolation must be 'linear' or 'previous'")
        if self.bounds not in {"hold", "error"}:
            raise ValueError("bounds must be 'hold' or 'error'")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(
            self,
            "density_m3_by_zone",
            _validated_profile_densities(self.density_m3_by_zone, time.shape),
        )
        if self.source is not None:
            object.__setattr__(self, "source", Path(self.source))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        zone_columns: Mapping[str, str] | None = None,
        interpolation: Literal["linear", "previous"] = "linear",
        bounds: Literal["hold", "error"] = "hold",
    ) -> PrescribedElectronProfile:
        """Read ``time_s`` plus numeric zone columns from a CSV file."""

        source = Path(path)
        columns, rows = _read_profile_csv(source)
        selected = _selected_profile_columns(source, columns, zone_columns)
        _validate_profile_rows(source, columns, rows)
        time, density = _numeric_profile_data(source, rows, selected)

        return cls(
            time_s=time,
            density_m3_by_zone=density,
            interpolation=interpolation,
            bounds=bounds,
            source=source,
        )

    def density(self, time_s: float, zone_id: str) -> float:
        query = float(time_s)
        key = str(zone_id)
        if not math.isfinite(query):
            raise ValueError("profile query time must be finite")
        values = self.density_m3_by_zone.get(key)
        if values is None:
            values = self.density_m3_by_zone.get("*")
        if values is None:
            raise KeyError(f"no prescribed electron density for zone {key!r}")
        if self.bounds == "error" and (
            query < self.time_s[0] or query > self.time_s[-1]
        ):
            raise ValueError(
                f"time_s={query:g} is outside [{self.time_s[0]:g}, {self.time_s[-1]:g}]"
            )
        query = float(np.clip(query, self.time_s[0], self.time_s[-1]))
        if self.interpolation == "previous":
            index = int(np.searchsorted(self.time_s, query, side="right") - 1)
            return float(values[max(index, 0)])
        return float(np.interp(query, self.time_s, values))

    def density_by_zone(
        self, time_s: float, zone_ids: Iterable[str]
    ) -> dict[str, float]:
        return {
            str(zone_id): self.density(time_s, str(zone_id)) for zone_id in zone_ids
        }


__all__ = ["PrescribedElectronProfile"]
