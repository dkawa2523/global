"""Strict, preloaded external power-table data and interpolation bindings.

The reader accepts the canonical v3 power columns and the migration-friendly
``absorbed_power_W`` or voltage/current form.  A store owns the per-compilation
cache, so neither recipe compilation nor the ODE right-hand side reopens files.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from plasma_global.errors import CaseValidationError, ModelDomainError


def _readonly(values: object) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ExternalTableData:
    source: Path
    time_s: np.ndarray
    electron_power_W: np.ndarray
    gas_power_W: np.ndarray
    reduced_field_Td: np.ndarray | None
    voltage_V: np.ndarray | None
    current_A: np.ndarray | None


def load_external_table(path: str | Path) -> ExternalTableData:
    """Read one strict circuit/power CSV into immutable numeric arrays."""

    source = Path(path).resolve()
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
    except OSError as exc:
        raise CaseValidationError(
            f"cannot read external power table {source}: {exc}"
        ) from exc
    except csv.Error as exc:
        raise CaseValidationError(
            f"external power table {source} is malformed CSV: {exc}"
        ) from exc

    if len(set(columns)) != len(columns):
        raise CaseValidationError(
            f"external power table {source} has duplicate columns"
        )
    allowed = {
        "time_s",
        "electron_power_W",
        "gas_power_W",
        "absorbed_power_W",
        "voltage_V",
        "current_A",
        "reduced_field_Td",
    }
    if unknown := sorted(set(columns) - allowed):
        raise CaseValidationError(
            f"external power table {source} has unknown columns: {', '.join(unknown)}"
        )
    if "time_s" not in columns:
        raise CaseValidationError(f"external power table {source} is missing time_s")
    direct_columns = {"electron_power_W", "absorbed_power_W"} & set(columns)
    if len(direct_columns) > 1:
        raise CaseValidationError(
            f"external power table {source} has ambiguous power columns"
        )
    has_voltage_current = {"voltage_V", "current_A"} <= set(columns)
    if not direct_columns and not has_voltage_current:
        raise CaseValidationError(
            f"external power table {source} needs electron_power_W, "
            "absorbed_power_W, or both voltage_V and current_A"
        )
    if len(rows) < 2:
        raise CaseValidationError(
            f"external power table {source} needs at least two rows"
        )
    for line, row in enumerate(rows, start=2):
        if None in row:
            raise CaseValidationError(
                f"external power table {source}:{line} contains extra CSV fields"
            )
        missing_values = [
            column
            for column in columns
            if row.get(column) is None or not str(row[column]).strip()
        ]
        if missing_values:
            raise CaseValidationError(
                f"external power table {source}:{line} has missing values for: "
                f"{', '.join(missing_values)}"
            )

    def column(name: str, *, default: float | None = None) -> np.ndarray | None:
        if name not in columns:
            if default is None:
                return None
            return np.full(len(rows), default, dtype=float)
        try:
            return np.asarray([float(row[name]) for row in rows], dtype=float)
        except (TypeError, ValueError) as exc:
            raise CaseValidationError(
                f"external power table {source} column {name!r} must be numeric"
            ) from exc

    time = column("time_s")
    voltage = column("voltage_V")
    current = column("current_A")
    direct_name = next(iter(direct_columns), None)
    electron = column(direct_name) if direct_name is not None else voltage * current
    gas = column("gas_power_W", default=0.0)
    field_values = column("reduced_field_Td")
    assert time is not None and electron is not None and gas is not None
    arrays = [time, electron, gas]
    arrays.extend(
        value for value in (field_values, voltage, current) if value is not None
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise CaseValidationError(
            f"external power table {source} contains non-finite values"
        )
    if np.any(np.diff(time) <= 0.0):
        raise CaseValidationError(
            f"external power table {source} time_s must be strictly increasing"
        )
    if np.any(electron < 0.0) or np.any(gas < 0.0):
        raise CaseValidationError(
            f"external power table {source} contains negative power"
        )
    if field_values is not None and np.any(field_values < 0.0):
        raise CaseValidationError(
            f"external power table {source} contains negative reduced field"
        )
    return ExternalTableData(
        source=source,
        time_s=_readonly(time),
        electron_power_W=_readonly(electron),
        gas_power_W=_readonly(gas),
        reduced_field_Td=None if field_values is None else _readonly(field_values),
        voltage_V=None if voltage is None else _readonly(voltage),
        current_A=None if current is None else _readonly(current),
    )


class ExternalTableStore:
    """Per-compilation cache for external power tables."""

    def __init__(self) -> None:
        self._tables: dict[Path, ExternalTableData] = {}

    def get(self, path: str | Path) -> ExternalTableData:
        source = Path(path).resolve()
        if source not in self._tables:
            self._tables[source] = load_external_table(source)
        return self._tables[source]


@dataclass(frozen=True, slots=True)
class ExternalTableBinding:
    time_dependent: ClassVar[bool] = True

    data: ExternalTableData
    interpolation: str
    bounds: str
    power_scale: float
    voltage_scale: float
    current_scale: float
    gap_m: float | None
    total_density_m3: float | None
    plasma_potential_V: float
    time_offset_s: float = 0.0

    @property
    def produces_reduced_field(self) -> bool:
        return self.data.reduced_field_Td is not None or (
            self.data.voltage_V is not None and self.gap_m is not None
        )

    def bind_previous(self, time_s: float) -> ExternalTableSample:
        """Compile one zero-order-hold interval to a fixed table row."""

        if self.interpolation != "previous":
            raise ValueError("bind_previous requires previous interpolation")
        query = float(time_s) - self.time_offset_s
        lower = float(self.data.time_s[0])
        upper = float(self.data.time_s[-1])
        if query < lower or query > upper:
            if self.bounds == "error":
                raise ModelDomainError(
                    f"external table time {query:g} is outside [{lower:g}, {upper:g}]"
                )
            query = float(np.clip(query, lower, upper))
        index = max(
            int(np.searchsorted(self.data.time_s, query, side="right") - 1),
            0,
        )

        def value(values: np.ndarray | None, scale: float) -> float | None:
            return None if values is None else scale * float(values[index])

        return ExternalTableSample(
            electron_power_W=self.power_scale
            * float(self.data.electron_power_W[index]),
            gas_power_W=self.power_scale * float(self.data.gas_power_W[index]),
            reduced_field_Td=value(self.data.reduced_field_Td, 1.0),
            voltage_V=value(self.data.voltage_V, self.voltage_scale),
            current_A=value(self.data.current_A, self.current_scale),
            gap_m=self.gap_m,
            total_density_m3=self.total_density_m3,
            plasma_potential_V=self.plasma_potential_V,
        )

    def _at(self, values: np.ndarray, time_s: float) -> float:
        query = float(time_s) - self.time_offset_s
        lower = float(self.data.time_s[0])
        upper = float(self.data.time_s[-1])
        if query < lower or query > upper:
            if self.bounds == "error":
                raise ModelDomainError(
                    f"external table time {query:g} is outside [{lower:g}, {upper:g}]"
                )
            query = float(np.clip(query, lower, upper))
        if self.interpolation == "previous":
            index = int(np.searchsorted(self.data.time_s, query, side="right") - 1)
            return float(values[max(index, 0)])
        return float(np.interp(query, self.data.time_s, values))


@dataclass(frozen=True, slots=True)
class ExternalTableSample:
    """One compiled zero-order-hold row used throughout a smooth segment."""

    time_dependent: ClassVar[bool] = False

    electron_power_W: float
    gas_power_W: float
    reduced_field_Td: float | None
    voltage_V: float | None
    current_A: float | None
    gap_m: float | None
    total_density_m3: float | None
    plasma_potential_V: float

    @property
    def produces_reduced_field(self) -> bool:
        return self.reduced_field_Td is not None or (
            self.voltage_V is not None and self.gap_m is not None
        )


__all__ = [
    "ExternalTableBinding",
    "ExternalTableData",
    "ExternalTableSample",
    "ExternalTableStore",
    "load_external_table",
]
