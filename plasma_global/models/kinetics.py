"""Prepared electron-rate and transport lookups used outside the ODE state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

from plasma_global._numeric_csv import read_numeric_csv
from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models._kinetics_table import (
    validate_table_modes,
    validated_table_arrays,
)


def _csv_mobility(
    columns: dict[str, np.ndarray],
    mobility_reference_neutral_density_m3: float | None,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    reduced = columns.get("reduced_mobility_1_V_m_s")
    mobility = columns.get("mobility_m2_V_s")
    reference = mobility_reference_neutral_density_m3
    if reduced is not None:
        if mobility is not None or reference is not None:
            raise CaseValidationError(
                "reduced mobility cannot be combined with absolute mobility "
                "or its reference density"
            )
        mobility, reference = None, None
    elif mobility is not None and reference is None:
        raise CaseValidationError(
            "absolute mobility requires mobility_reference_neutral_density_m3 "
            "from table generation"
        )
    elif mobility is None and reference is not None:
        raise CaseValidationError(
            "mobility reference density is unused without a mobility column"
        )
    return mobility, reduced, reference


def _reference_density(reference: float | None) -> float | None:
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
    return reference


def _csv_coordinates(
    source: Path, values: dict[str, np.ndarray], lookup: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    standard = {
        "mean_energy_eV",
        "reduced_field_Td",
        "mobility_m2_V_s",
        "reduced_mobility_1_V_m_s",
    }
    unknown = [
        name
        for name in values
        if name not in standard and not name.startswith(("k:", "momentum:"))
    ]
    if unknown:
        raise CaseValidationError(
            f"electron table {source} has unknown columns: {unknown}"
        )
    mean = values.get("mean_energy_eV")
    if mean is None:
        raise CaseValidationError(f"electron table {source} requires mean_energy_eV")
    field = values.get("reduced_field_Td")
    axis = mean if lookup == "mean_energy" else field
    if axis is None:
        raise CaseValidationError(f"electron table {source} requires reduced_field_Td")
    return axis, mean, field


def _csv_rates(
    source: Path, values: dict[str, np.ndarray], required: tuple[str, ...]
) -> dict[str, np.ndarray]:
    names = [name for name in values if name.startswith(("k:", "momentum:"))]
    rates = {name.removeprefix("k:"): values[name] for name in names}
    if len(names) != len(rates):
        raise CaseValidationError("duplicate coefficient IDs after removing k: prefix")
    if missing := sorted(set(required) - set(rates)):
        raise CaseValidationError(
            f"electron table {source} is missing rate coefficients: "
            + ", ".join(missing)
        )
    return rates


@dataclass(frozen=True, slots=True)
class ElectronKineticsResult:
    mean_energy_eV: float
    electron_temperature_eV: float
    mobility_m2_V_s: float | None
    effective_field_Td: float | None
    rate_coefficients: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (
            self.mean_energy_eV,
            self.electron_temperature_eV,
            self.mobility_m2_V_s,
            self.effective_field_Td,
        )
        finite = tuple(value for value in values if value is not None)
        if not np.isfinite(finite).all() or min(finite) < 0.0:
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
    mobility_m2_V_s: np.ndarray | None
    effective_field_Td: np.ndarray | None
    rate_tables: Mapping[str, np.ndarray]
    mobility_reference_neutral_density_m3: float | None = None
    reduced_mobility_1_V_m_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        try:
            source = Path(self.source)
        except (TypeError, ValueError) as exc:
            raise CaseValidationError("electron table source must be a path") from exc
        validate_table_modes(self.lookup, self.bounds)
        reduced = self.reduced_mobility_1_V_m_s
        if reduced is not None and (
            self.mobility_m2_V_s is not None
            or self.mobility_reference_neutral_density_m3 is not None
        ):
            raise CaseValidationError(
                "reduced mobility has no absolute mobility/reference density"
            )
        axis, data = validated_table_arrays(
            source,
            self.lookup,
            self.axis,
            self.mean_energy_eV,
            self.mobility_m2_V_s if reduced is None else reduced,
            self.effective_field_Td,
            self.rate_tables,
        )
        reference = _reference_density(self.mobility_reference_neutral_density_m3)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "mean_energy_eV", data.mean_energy_eV)
        object.__setattr__(
            self, "mobility_m2_V_s", data.mobility_m2_V_s if reduced is None else None
        )
        object.__setattr__(
            self,
            "reduced_mobility_1_V_m_s",
            None if reduced is None else data.mobility_m2_V_s,
        )
        object.__setattr__(self, "effective_field_Td", data.effective_field_Td)
        object.__setattr__(self, "rate_tables", MappingProxyType(data.rate_tables))
        object.__setattr__(self, "mobility_reference_neutral_density_m3", reference)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        lookup: str,
        required_rate_ids: tuple[str, ...] = (),
        bounds: str = "error",
        mobility_reference_neutral_density_m3: float | None = None,
        require_mobility: bool = False,
        columns: Mapping[str, str] | None = None,
    ) -> TabulatedElectronKinetics:
        source = Path(path).resolve()
        validate_table_modes(lookup, bounds)
        values = read_numeric_csv(source, columns=columns)
        axis, mean, field = _csv_coordinates(source, values, lookup)
        mobility, reduced, reference = _csv_mobility(
            values, mobility_reference_neutral_density_m3
        )
        if require_mobility and mobility is None and reduced is None:
            raise CaseValidationError(
                "the selected power model requires electron mobility"
            )
        rates = _csv_rates(source, values, required_rate_ids)
        return cls(
            source,
            lookup,
            bounds,
            axis,
            mean,
            mobility,
            field,
            rates,
            reference,
            reduced,
        )

    def _mobility_scale(self, neutral_density_m3: float | None) -> float:
        reference = self.mobility_reference_neutral_density_m3
        if reference is None and self.reduced_mobility_1_V_m_s is None:
            return 1.0
        if neutral_density_m3 is None:
            raise ModelDomainError(
                "electron mobility requires current neutral_density_m3"
            )
        if not math.isfinite(neutral_density_m3) or neutral_density_m3 <= 0.0:
            raise ModelDomainError(
                "electron mobility requires finite positive neutral density"
            )
        if self.reduced_mobility_1_V_m_s is not None:
            return 1.0 / neutral_density_m3
        return 1.0 if reference is None else reference / neutral_density_m3

    def mobility_curve(self, neutral_density_m3: float) -> np.ndarray:
        """Return mobility at current density while preserving reduced mobility."""

        values = self._mobility_values()
        if values is None:
            raise ModelDomainError("electron mobility was not supplied")
        return values * self._mobility_scale(neutral_density_m3)

    @property
    def mobility_quantity(self) -> str | None:
        if self.reduced_mobility_1_V_m_s is not None:
            return "reduced_mobility"
        return None if self.mobility_m2_V_s is None else "mobility"

    def _mobility_values(self) -> np.ndarray | None:
        return (
            self.mobility_m2_V_s
            if self.reduced_mobility_1_V_m_s is None
            else self.reduced_mobility_1_V_m_s
        )

    def _mobility_at(self, query: float, density: float | None) -> float | None:
        values = self._mobility_values()
        if values is None:
            return None
        return float(np.interp(query, self.axis, values)) * self._mobility_scale(
            density
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
            mobility_m2_V_s=self._mobility_at(query, neutral_density_m3),
            effective_field_Td=(
                None
                if self.effective_field_Td is None
                else interpolate(self.effective_field_Td)
            ),
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
            mobility_m2_V_s=self._mobility_at(float(self.axis[0]), neutral_density_m3),
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
