"""Prepared electron-rate and transport lookups used outside the ODE state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import h5py
import numpy as np

from plasma_global.errors import CaseValidationError, ModelDomainError


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
        with h5py.File(source, "r") as handle:
            required_datasets = {
                "mean_energy_eV",
                "mobility_m2_V_s",
                "effective_field_Td",
            }
            allowed_entries = {*required_datasets, "rate_coefficients"}
            unknown = sorted(set(handle) - allowed_entries)
            if unknown:
                raise CaseValidationError(
                    f"electron table {source} has unknown datasets: "
                    f"{', '.join(unknown)}"
                )
            missing = sorted(name for name in required_datasets if name not in handle)
            if missing:
                raise CaseValidationError(
                    f"electron table {source} is missing: {', '.join(missing)}"
                )
            mean_energy = np.asarray(handle["mean_energy_eV"][:], dtype=float)
            mobility = np.asarray(handle["mobility_m2_V_s"][:], dtype=float)
            field_values = np.asarray(handle["effective_field_Td"][:], dtype=float)
            if "rate_coefficients" not in handle:
                raise CaseValidationError(
                    f"electron table {source} is missing rate_coefficients"
                )
            group = handle["rate_coefficients"]
            missing_rates = sorted(set(required_rate_ids) - set(group))
            if missing_rates:
                raise CaseValidationError(
                    f"electron table {source} is missing rate coefficients: {', '.join(missing_rates)}"
                )
            selected = tuple(
                dict.fromkeys(
                    (
                        *required_rate_ids,
                        *(name for name in optional_rate_ids if name in group),
                    )
                )
            ) or tuple(str(name) for name in group)
            rate_tables = {
                name: np.asarray(group[name][:], dtype=float) for name in selected
            }
        axis = mean_energy if lookup == "mean_energy" else field_values
        arrays = {
            "axis": axis,
            "mean_energy_eV": mean_energy,
            "mobility_m2_V_s": mobility,
            "effective_field_Td": field_values,
            **{f"rate:{name}": values for name, values in rate_tables.items()},
        }
        size = axis.size
        if size < 2 or any(
            values.ndim != 1 or values.size != size for values in arrays.values()
        ):
            raise CaseValidationError(
                f"electron table {source} datasets must be equal 1-D arrays with at least two points"
            )
        if any(not np.all(np.isfinite(values)) for values in arrays.values()):
            raise CaseValidationError(
                f"electron table {source} contains non-finite values"
            )
        if np.any(np.diff(axis) <= 0.0):
            raise CaseValidationError(
                f"electron table {source} lookup axis must be strictly increasing and unique"
            )
        if (
            np.any(mean_energy < 0.0)
            or np.any(mobility <= 0.0)
            or np.any(field_values < 0.0)
        ):
            raise CaseValidationError(
                f"electron table {source} contains invalid transport values"
            )
        if any(np.any(values < 0.0) for values in rate_tables.values()):
            raise CaseValidationError(
                f"electron table {source} contains negative rates"
            )
        for values in arrays.values():
            values.setflags(write=False)
        return cls(
            source=source,
            lookup=lookup,
            bounds=bounds,
            axis=axis,
            mean_energy_eV=mean_energy,
            mobility_m2_V_s=mobility,
            effective_field_Td=field_values,
            rate_tables=MappingProxyType(rate_tables),
        )

    def _query(self, value: float) -> float:
        query = float(value)
        if not math.isfinite(query):
            raise ModelDomainError("electron table lookup value must be finite")
        if query < self.axis[0] or query > self.axis[-1]:
            if self.bounds == "error":
                raise ModelDomainError(
                    f"electron table lookup {query:g} is outside [{self.axis[0]:g}, {self.axis[-1]:g}]"
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
