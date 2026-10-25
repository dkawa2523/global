"""Strict, preloaded external power-table data and interpolation bindings.

The reader accepts explicit electron or absorbed power, or a voltage/current
pair. A store owns the per-compilation cache, so bindings share preloaded data
and the ODE right-hand side never reopens files.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import numpy as np

from plasma_global._numeric_csv import read_numeric_csv
from plasma_global.errors import CaseValidationError, ModelDomainError

_ALLOWED_COLUMNS = frozenset(
    {
        "time_s",
        "electron_power_W",
        "gas_power_W",
        "absorbed_power_W",
        "voltage_V",
        "current_A",
        "reduced_field_Td",
    }
)
_DIRECT_POWER_COLUMNS = frozenset({"electron_power_W", "absorbed_power_W"})
_CsvRow = dict[str | None, str | list[str] | None]


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
    power_from_vi: bool = False


def _validate_column_schema(source: Path, columns: tuple[str, ...]) -> str | None:
    if len(set(columns)) != len(columns):
        raise CaseValidationError(
            f"external power table {source} has duplicate columns"
        )
    if unknown := sorted(set(columns) - _ALLOWED_COLUMNS):
        raise CaseValidationError(
            f"external power table {source} has unknown columns: {', '.join(unknown)}"
        )
    if "time_s" not in columns:
        raise CaseValidationError(f"external power table {source} is missing time_s")
    direct_columns = _DIRECT_POWER_COLUMNS & set(columns)
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
    return next(iter(direct_columns), None)


def _validate_numeric_table(table: ExternalTableData) -> None:
    arrays = [table.time_s, table.electron_power_W, table.gas_power_W]
    arrays.extend(
        value
        for value in (table.reduced_field_Td, table.voltage_V, table.current_A)
        if value is not None
    )
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise CaseValidationError(
            f"external power table {table.source} contains non-finite values"
        )
    if np.any(np.diff(table.time_s) <= 0.0):
        raise CaseValidationError(
            f"external power table {table.source} time_s must be strictly increasing"
        )
    if np.any(table.electron_power_W < 0.0) or np.any(table.gas_power_W < 0.0):
        raise CaseValidationError(
            f"external power table {table.source} contains negative power"
        )
    if table.reduced_field_Td is not None and np.any(table.reduced_field_Td < 0.0):
        raise CaseValidationError(
            f"external power table {table.source} contains negative reduced field"
        )


def _readonly_table(table: ExternalTableData) -> ExternalTableData:
    return ExternalTableData(
        source=table.source,
        time_s=_readonly(table.time_s),
        electron_power_W=_readonly(table.electron_power_W),
        gas_power_W=_readonly(table.gas_power_W),
        reduced_field_Td=(
            None
            if table.reduced_field_Td is None
            else _readonly(table.reduced_field_Td)
        ),
        voltage_V=None if table.voltage_V is None else _readonly(table.voltage_V),
        current_A=None if table.current_A is None else _readonly(table.current_A),
        power_from_vi=table.power_from_vi,
    )


def load_external_table(
    path: str | Path, *, columns: Mapping[str, str] | None = None
) -> ExternalTableData:
    """Read one strict circuit/power CSV into immutable numeric arrays."""

    source = Path(path).resolve()
    try:
        values = read_numeric_csv(source, columns=columns, allowed=_ALLOWED_COLUMNS)
    except (OSError, ValueError, csv.Error) as exc:
        raise CaseValidationError(
            f"cannot read external power table {source}: {exc}"
        ) from exc
    direct = _validate_column_schema(source, tuple(values))
    time = values["time_s"]
    if time.size < 2:
        raise CaseValidationError(
            f"external power table {source} needs at least two rows"
        )
    voltage, current = values.get("voltage_V"), values.get("current_A")
    electron = (
        values[direct]
        if direct is not None
        else cast(np.ndarray, voltage) * cast(np.ndarray, current)
    )
    gas = values.get("gas_power_W", np.zeros_like(time))
    if direct != "electron_power_W":
        electron = electron - gas
    table = ExternalTableData(
        source,
        time,
        electron,
        gas,
        values.get("reduced_field_Td"),
        voltage,
        current,
        direct is None,
    )
    _validate_numeric_table(table)
    return _readonly_table(table)


class ExternalTableStore:
    """Per-compilation cache for external power tables."""

    def __init__(self) -> None:
        self._tables: dict[
            tuple[Path, tuple[tuple[str, str], ...]], ExternalTableData
        ] = {}

    def get(
        self, path: str | Path, *, columns: Mapping[str, str] | None = None
    ) -> ExternalTableData:
        source = Path(path).resolve()
        identity = (source, tuple(sorted((columns or {}).items())))
        if identity not in self._tables:
            self._tables[identity] = (
                load_external_table(source, columns=columns)
                if columns
                else load_external_table(source)
            )
        return self._tables[identity]


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

    def __post_init__(self) -> None:
        self._validate_controls()
        self._validate_columns()
        if self.data.power_from_vi and self.power_scale > 0.0:
            self._validate_vi_power()

    def _validate_controls(self) -> None:
        if self.interpolation not in {"linear", "previous"}:
            raise CaseValidationError("interpolation must be linear or previous")
        if self.bounds not in {"error", "hold"}:
            raise CaseValidationError("bounds must be error or hold")
        scales = (self.power_scale, self.voltage_scale, self.current_scale)
        if not np.isfinite(scales).all() or min(scales) < 0.0:
            raise CaseValidationError(
                "external table scales must be finite and nonnegative"
            )
        if not np.isfinite((self.time_offset_s, self.plasma_potential_V)).all():
            raise CaseValidationError(
                "external table offset and potential must be finite"
            )
        for name, value in (
            ("gap_m", self.gap_m),
            ("total_density_m3", self.total_density_m3),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise CaseValidationError(f"{name} must be finite and positive")

    def _validate_vi_power(self) -> None:
        data = self.data
        if data.voltage_V is None or data.current_A is None:
            raise CaseValidationError("V*I power requires voltage_V and current_A")
        voltage = data.voltage_V * self.voltage_scale
        current = data.current_A * self.current_scale
        power = voltage * current - data.gas_power_W
        if self.interpolation == "linear":
            dv, di, dg = np.diff(voltage), np.diff(current), np.diff(data.gas_power_W)
            quadratic = dv * di
            linear = voltage[:-1] * di + current[:-1] * dv - dg
            position = np.divide(
                -linear,
                2.0 * quadratic,
                out=np.zeros_like(linear),
                where=quadratic > 0.0,
            )
            active = (quadratic > 0.0) & (position > 0.0) & (position < 1.0)
            position = position[active]
            interior = (voltage[:-1][active] + position * dv[active]) * (
                current[:-1][active] + position * di[active]
            ) - (data.gas_power_W[:-1][active] + position * dg[active])
            power = np.concatenate((power, interior))
        if not np.isfinite(power).all() or np.any(power < 0.0):
            raise CaseValidationError(
                "scaled/interpolated V*I must be finite and cover gas_power_W; "
                "electron power cannot be negative"
            )

    def _validate_columns(self) -> None:
        for column, scale in (
            (self.data.voltage_V, self.voltage_scale),
            (self.data.current_A, self.current_scale),
        ):
            if column is None and scale != 1.0:
                raise CaseValidationError(
                    "voltage_scale/current_scale requires the corresponding CSV column"
                )
        if self.gap_m is not None and (
            self.data.voltage_V is None or self.data.reduced_field_Td is not None
        ):
            raise CaseValidationError(
                "gap_m requires voltage_V without an explicit reduced_field_Td column"
            )
        if self.total_density_m3 is not None and self.gap_m is None:
            raise CaseValidationError("total_density_m3 requires gap_m")

    def validate_interval(self, start_s: float, end_s: float) -> None:
        """Reject a recipe outside the table before integration starts."""

        if self.bounds == "error" and (
            start_s - self.time_offset_s < self.data.time_s[0]
            or end_s - self.time_offset_s > self.data.time_s[-1]
        ):
            raise CaseValidationError(
                f"external table {self.data.source} does not cover recipe interval "
                f"[{start_s:g}, {end_s:g}] s with time_offset_s={self.time_offset_s:g}"
            )

    @property
    def produces_reduced_field(self) -> bool:
        return self.data.reduced_field_Td is not None or (
            self.data.voltage_V is not None and self.gap_m is not None
        )

    def bind_previous(self, time_s: float) -> ExternalTableSample:
        """Compile one zero-order-hold interval to a fixed table row."""

        if self.interpolation != "previous":
            raise ValueError("bind_previous requires previous interpolation")
        return self.sample(time_s)

    def sample(self, time_s: float) -> ExternalTableSample:
        """Interpolate physical inputs, then calculate power from scaled V and I."""

        def value(values: np.ndarray | None, scale: float) -> float | None:
            return None if values is None else scale * self._at(values, time_s)

        voltage = value(self.data.voltage_V, self.voltage_scale)
        current = value(self.data.current_A, self.current_scale)
        gas = self._at(self.data.gas_power_W, time_s)
        electron = self._at(self.data.electron_power_W, time_s)
        if self.data.power_from_vi:
            if voltage is None or current is None:
                raise CaseValidationError("V*I power requires voltage_V and current_A")
            electron = voltage * current - gas

        return ExternalTableSample(
            electron_power_W=self.power_scale * electron,
            gas_power_W=self.power_scale * gas,
            reduced_field_Td=value(self.data.reduced_field_Td, 1.0),
            voltage_V=voltage,
            current_A=current,
            gap_m=self.gap_m,
            total_density_m3=self.total_density_m3,
            plasma_potential_V=self.plasma_potential_V,
        )

    def _at(self, values: np.ndarray, time_s: float) -> float:
        if not np.isfinite(time_s):
            raise ModelDomainError("external table time must be finite")
        query = time_s - self.time_offset_s
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
