"""Prepared electron-rate and transport lookups used outside the ODE state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np

from plasma_global.errors import CaseValidationError, ModelDomainError


@dataclass(frozen=True, slots=True)
class _ElectronTableArrays:
    mean_energy_eV: np.ndarray
    mobility_m2_V_s: np.ndarray
    effective_field_Td: np.ndarray
    rate_tables: dict[str, np.ndarray]


def _read_rate_tables(
    group: Any,
    *,
    source: Path,
    required_rate_ids: tuple[str, ...],
    optional_rate_ids: tuple[str, ...],
) -> dict[str, np.ndarray]:
    missing = sorted(set(required_rate_ids) - set(group))
    if missing:
        raise CaseValidationError(
            f"electron table {source} is missing rate coefficients: "
            f"{', '.join(missing)}"
        )
    selected = tuple(
        dict.fromkeys(
            (
                *required_rate_ids,
                *(name for name in optional_rate_ids if name in group),
            )
        )
    ) or tuple(str(name) for name in group)
    return {name: np.asarray(group[name][:], dtype=float) for name in selected}


def _read_electron_table(
    source: Path,
    *,
    required_rate_ids: tuple[str, ...],
    optional_rate_ids: tuple[str, ...],
) -> _ElectronTableArrays:
    with h5py.File(source, "r") as handle:
        required = {"mean_energy_eV", "mobility_m2_V_s", "effective_field_Td"}
        unknown = sorted(set(handle) - {*required, "rate_coefficients"})
        if unknown:
            raise CaseValidationError(
                f"electron table {source} has unknown datasets: {', '.join(unknown)}"
            )
        missing = sorted(name for name in required if name not in handle)
        if missing:
            raise CaseValidationError(
                f"electron table {source} is missing: {', '.join(missing)}"
            )
        if "rate_coefficients" not in handle:
            raise CaseValidationError(
                f"electron table {source} is missing rate_coefficients"
            )
        return _ElectronTableArrays(
            mean_energy_eV=np.asarray(handle["mean_energy_eV"][:], dtype=float),
            mobility_m2_V_s=np.asarray(handle["mobility_m2_V_s"][:], dtype=float),
            effective_field_Td=np.asarray(handle["effective_field_Td"][:], dtype=float),
            rate_tables=_read_rate_tables(
                handle["rate_coefficients"],
                source=source,
                required_rate_ids=required_rate_ids,
                optional_rate_ids=optional_rate_ids,
            ),
        )


def _table_arrays(
    data: _ElectronTableArrays, axis: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "axis": axis,
        "mean_energy_eV": data.mean_energy_eV,
        "mobility_m2_V_s": data.mobility_m2_V_s,
        "effective_field_Td": data.effective_field_Td,
        **{f"rate:{name}": values for name, values in data.rate_tables.items()},
    }


def _validate_table_array_contract(
    source: Path, arrays: Mapping[str, np.ndarray], axis: np.ndarray
) -> None:
    size = axis.size
    if size < 2 or any(
        values.ndim != 1 or values.size != size for values in arrays.values()
    ):
        raise CaseValidationError(
            f"electron table {source} datasets must be equal 1-D arrays with at "
            "least two points"
        )
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise CaseValidationError(f"electron table {source} contains non-finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise CaseValidationError(
            f"electron table {source} lookup axis must be strictly increasing "
            "and unique"
        )


def _validate_table_values(source: Path, data: _ElectronTableArrays) -> None:
    if (
        np.any(data.mean_energy_eV < 0.0)
        or np.any(data.mobility_m2_V_s <= 0.0)
        or np.any(data.effective_field_Td < 0.0)
    ):
        raise CaseValidationError(
            f"electron table {source} contains invalid transport values"
        )
    if any(np.any(values < 0.0) for values in data.rate_tables.values()):
        raise CaseValidationError(f"electron table {source} contains negative rates")


@dataclass(frozen=True, slots=True)
class ElectronKineticsResult:
    mean_energy_eV: float
    electron_temperature_eV: float
    mobility_m2_V_s: float
    effective_field_Td: float
    rate_coefficients: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (
            self.mean_energy_eV,
            self.electron_temperature_eV,
            self.mobility_m2_V_s,
            self.effective_field_Td,
        )
        if not np.isfinite(values).all() or min(values) < 0.0:
            raise ModelDomainError("electron kinetics returned invalid transport data")
        object.__setattr__(
            self, "rate_coefficients", MappingProxyType(dict(self.rate_coefficients))
        )


@dataclass(frozen=True, slots=True)
class TabulatedElectronKinetics:
    """One prevalidated HDF5 table; interpolation never sorts or scans metadata."""

    source: Path
    lookup: str
    bounds: str
    axis: np.ndarray
    mean_energy_eV: np.ndarray
    mobility_m2_V_s: np.ndarray
    effective_field_Td: np.ndarray
    rate_tables: Mapping[str, np.ndarray]

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        lookup: str,
        required_rate_ids: tuple[str, ...] = (),
        optional_rate_ids: tuple[str, ...] = (),
        bounds: str = "error",
    ) -> TabulatedElectronKinetics:
        source = Path(path).resolve()
        if lookup not in {"mean_energy", "local_field"}:
            raise CaseValidationError("table lookup must be mean_energy or local_field")
        if bounds not in {"error", "clip"}:
            raise CaseValidationError("table bounds must be error or clip")
        if not source.is_file():
            raise CaseValidationError(f"electron table does not exist: {source}")
        data = _read_electron_table(
            source,
            required_rate_ids=required_rate_ids,
            optional_rate_ids=optional_rate_ids,
        )
        axis = (
            data.mean_energy_eV if lookup == "mean_energy" else data.effective_field_Td
        )
        arrays = _table_arrays(data, axis)
        _validate_table_array_contract(source, arrays, axis)
        _validate_table_values(source, data)
        for values in arrays.values():
            values.setflags(write=False)
        return cls(
            source=source,
            lookup=lookup,
            bounds=bounds,
            axis=axis,
            mean_energy_eV=data.mean_energy_eV,
            mobility_m2_V_s=data.mobility_m2_V_s,
            effective_field_Td=data.effective_field_Td,
            rate_tables=MappingProxyType(data.rate_tables),
        )

    def _query(self, value: float) -> float:
        query = value
        if not math.isfinite(query):
            raise ModelDomainError("electron table lookup value must be finite")
        if query < self.axis[0] or query > self.axis[-1]:
            if self.bounds == "error":
                raise ModelDomainError(
                    f"electron table lookup {query:g} is outside "
                    f"[{self.axis[0]:g}, {self.axis[-1]:g}]"
                )
            query = float(np.clip(query, self.axis[0], self.axis[-1]))
        return query

    def evaluate(
        self,
        *,
        mean_energy_eV: float | None = None,
        reduced_field_Td: float | None = None,
    ) -> ElectronKineticsResult:
        supplied = mean_energy_eV if self.lookup == "mean_energy" else reduced_field_Td
        if supplied is None:
            raise ModelDomainError(
                f"{self.lookup} electron table lookup value is required"
            )
        query = self._query(supplied)

        def interpolate(values: np.ndarray) -> float:
            return float(np.interp(query, self.axis, values))

        mean_energy = interpolate(self.mean_energy_eV)
        return ElectronKineticsResult(
            mean_energy_eV=mean_energy,
            electron_temperature_eV=(2.0 / 3.0) * mean_energy,
            mobility_m2_V_s=interpolate(self.mobility_m2_V_s),
            effective_field_Td=interpolate(self.effective_field_Td),
            rate_coefficients={
                name: interpolate(values) for name, values in self.rate_tables.items()
            },
        )

    def mean_energy_from_field(self, reduced_field_Td: float) -> float:
        if self.lookup != "local_field":
            raise ModelDomainError(
                "mean_energy_from_field requires a local_field table"
            )
        return self.evaluate(reduced_field_Td=reduced_field_Td).mean_energy_eV


__all__ = ["ElectronKineticsResult", "TabulatedElectronKinetics"]
