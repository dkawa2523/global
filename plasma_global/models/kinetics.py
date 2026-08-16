"""Prepared electron-rate and transport lookups used outside the ODE state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models._kinetics_table import (
    read_electron_table,
    validate_table_modes,
    validated_table_arrays,
)


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
    """Immutable electron table with one contract for file and direct inputs."""

    source: Path
    lookup: str
    bounds: str
    axis: np.ndarray
    mean_energy_eV: np.ndarray
    mobility_m2_V_s: np.ndarray
    effective_field_Td: np.ndarray
    rate_tables: Mapping[str, np.ndarray]
    mobility_reference_neutral_density_m3: float | None = None

    def __post_init__(self) -> None:
        try:
            source = Path(self.source)
        except (TypeError, ValueError) as exc:
            raise CaseValidationError("electron table source must be a path") from exc
        validate_table_modes(self.lookup, self.bounds)
        axis, data = validated_table_arrays(
            source,
            self.lookup,
            self.axis,
            self.mean_energy_eV,
            self.mobility_m2_V_s,
            self.effective_field_Td,
            self.rate_tables,
        )
        reference = self.mobility_reference_neutral_density_m3
        if reference is not None:
            if isinstance(reference, bool) or not isinstance(reference, Real):
                raise CaseValidationError(
                    "electron mobility reference neutral density must be finite and "
                    "positive"
                )
            try:
                reference = float(reference)
            except (OverflowError, ValueError) as exc:
                raise CaseValidationError(
                    "electron mobility reference neutral density must be finite and "
                    "positive"
                ) from exc
            if not math.isfinite(reference) or reference <= 0.0:
                raise CaseValidationError(
                    "electron mobility reference neutral density must be finite and "
                    "positive"
                )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "mean_energy_eV", data.mean_energy_eV)
        object.__setattr__(self, "mobility_m2_V_s", data.mobility_m2_V_s)
        object.__setattr__(self, "effective_field_Td", data.effective_field_Td)
        object.__setattr__(self, "rate_tables", MappingProxyType(data.rate_tables))
        object.__setattr__(self, "mobility_reference_neutral_density_m3", reference)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        *,
        lookup: str,
        required_rate_ids: tuple[str, ...] = (),
        optional_rate_ids: tuple[str, ...] = (),
        bounds: str = "error",
        mobility_reference_neutral_density_m3: float | None = None,
    ) -> TabulatedElectronKinetics:
        source = Path(path).resolve()
        validate_table_modes(lookup, bounds)
        if not source.is_file():
            raise CaseValidationError(f"electron table does not exist: {source}")
        data = read_electron_table(
            source,
            required_rate_ids=required_rate_ids,
            optional_rate_ids=optional_rate_ids,
        )
        axis = (
            data.mean_energy_eV if lookup == "mean_energy" else data.effective_field_Td
        )
        return cls(
            source=source,
            lookup=lookup,
            bounds=bounds,
            axis=axis,
            mean_energy_eV=data.mean_energy_eV,
            mobility_m2_V_s=data.mobility_m2_V_s,
            effective_field_Td=data.effective_field_Td,
            rate_tables=data.rate_tables,
            mobility_reference_neutral_density_m3=(
                mobility_reference_neutral_density_m3
            ),
        )

    def _mobility_scale(self, neutral_density_m3: float | None) -> float:
        reference = self.mobility_reference_neutral_density_m3
        if reference is None:
            return 1.0
        if neutral_density_m3 is None:
            raise ModelDomainError(
                "electron mobility with a reference density requires current "
                "neutral_density_m3"
            )
        if not math.isfinite(neutral_density_m3) or neutral_density_m3 <= 0.0:
            raise ModelDomainError(
                "electron mobility requires finite positive neutral density"
            )
        return reference / neutral_density_m3

    def mobility_curve(self, neutral_density_m3: float) -> np.ndarray:
        """Return mobility at current density while preserving reduced mobility."""

        return self.mobility_m2_V_s * self._mobility_scale(neutral_density_m3)

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
        neutral_density_m3: float | None = None,
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
            mobility_m2_V_s=(
                interpolate(self.mobility_m2_V_s)
                * self._mobility_scale(neutral_density_m3)
            ),
            effective_field_Td=interpolate(self.effective_field_Td),
            rate_coefficients={
                name: interpolate(values) for name, values in self.rate_tables.items()
            },
        )

    def zero_field_result(
        self, *, neutral_density_m3: float | None = None
    ) -> ElectronKineticsResult:
        """Return the explicit cold-electron boundary used when a port is off."""

        if self.lookup != "local_field":
            raise ModelDomainError("zero-field kinetics require a local_field table")
        return ElectronKineticsResult(
            mean_energy_eV=0.0,
            electron_temperature_eV=0.0,
            # Zero-field mobility remains finite in a swarm.  The first
            # resolved table value is the least-assumptive boundary value and
            # is only used by electrical ports; rates and energy stay zero.
            mobility_m2_V_s=(
                float(self.mobility_m2_V_s[0])
                * self._mobility_scale(neutral_density_m3)
            ),
            effective_field_Td=0.0,
            rate_coefficients=dict.fromkeys(self.rate_tables, 0.0),
        )

    def mean_energy_from_field(self, reduced_field_Td: float) -> float:
        if self.lookup != "local_field":
            raise ModelDomainError(
                "mean_energy_from_field requires a local_field table"
            )
        query = self._query(reduced_field_Td)
        return float(np.interp(query, self.axis, self.mean_energy_eV))


__all__ = ["ElectronKineticsResult", "TabulatedElectronKinetics"]
