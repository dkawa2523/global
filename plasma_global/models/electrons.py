"""Mutually exclusive electron closures for the compact global model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from plasma_global.core.exceptions import (
    ModelConfigurationError,
    QuasineutralityError,
    StateDomainError,
)

ELEMENTARY_CHARGE_C = 1.602176634e-19


@dataclass(frozen=True)
class ElectronState:
    density_m3: float
    mean_energy_eV: float
    temperature_eV: float
    energy_density_J_m3: float
    reduced_field_Td: float | None


class ElectronClosure(Protocol):
    evolves_energy: bool
    mode: str
    mean_energy_from_field: FieldMeanEnergy | None

    def evaluate(
        self,
        *,
        net_heavy_charge_density_m3: float,
        energy_density_J_m3: float | None,
        reduced_field_Td: float | None,
        electron_density_m3: float | None = None,
    ) -> ElectronState: ...


def quasineutral_electron_density(net_heavy_charge_density_m3: float) -> float:
    """Return ``n_e = sum(z_s n_s)`` without a hidden numerical floor."""

    value = float(net_heavy_charge_density_m3)
    if not math.isfinite(value):
        raise StateDomainError("Net heavy-particle charge density must be finite")
    if value < 0.0:
        raise QuasineutralityError(
            "Quasineutral closure produced a negative electron density "
            f"({value:.6e} m^-3); use a charge-continuity closure for this regime"
        )
    return value


def resolved_electron_density(
    net_heavy_charge_density_m3: float,
    prescribed_electron_density_m3: float | None,
) -> float:
    """Select quasineutral or explicitly prescribed electron density."""

    if prescribed_electron_density_m3 is None:
        return quasineutral_electron_density(net_heavy_charge_density_m3)
    value = float(prescribed_electron_density_m3)
    if not math.isfinite(value) or value < 0.0:
        raise StateDomainError(
            "Prescribed electron density must be finite and nonnegative"
        )
    return value


@dataclass(frozen=True)
class ElectronEnergyClosure:
    """Evolve electron energy density and derive both mean energy and Te."""

    mode: str = "electron_energy"
    evolves_energy: bool = True
    mean_energy_from_field: None = None

    def evaluate(
        self,
        *,
        net_heavy_charge_density_m3: float,
        energy_density_J_m3: float | None,
        reduced_field_Td: float | None,
        electron_density_m3: float | None = None,
    ) -> ElectronState:
        density = resolved_electron_density(
            net_heavy_charge_density_m3, electron_density_m3
        )
        if energy_density_J_m3 is None:
            raise StateDomainError(
                "electron_energy closure requires an electron-energy state"
            )
        energy = float(energy_density_J_m3)
        if not math.isfinite(energy) or energy < 0.0:
            raise StateDomainError(
                "Electron energy density must be finite and non-negative"
            )
        if density == 0.0:
            if energy != 0.0:
                raise StateDomainError(
                    "Positive electron energy is undefined at zero electron density"
                )
            mean_energy = 0.0
        else:
            mean_energy = energy / (density * ELEMENTARY_CHARGE_C)
        return ElectronState(
            density_m3=density,
            mean_energy_eV=mean_energy,
            temperature_eV=(2.0 / 3.0) * mean_energy,
            energy_density_J_m3=energy,
            reduced_field_Td=reduced_field_Td,
        )


class FieldMeanEnergy(Protocol):
    def __call__(self, reduced_field_Td: float) -> float: ...


@dataclass(frozen=True)
class TabulatedMeanEnergy:
    """Strict, non-extrapolating E/N-to-mean-energy interpolation."""

    reduced_field_Td: tuple[float, ...]
    mean_energy_eV: tuple[float, ...]

    def __post_init__(self) -> None:
        fields = np.asarray(self.reduced_field_Td, dtype=float)
        energies = np.asarray(self.mean_energy_eV, dtype=float)
        if (
            fields.ndim != 1
            or energies.ndim != 1
            or fields.size < 2
            or fields.shape != energies.shape
        ):
            raise ModelConfigurationError(
                "Local-field table needs equal one-dimensional axes with at least two points"
            )
        if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(energies)):
            raise ModelConfigurationError("Local-field table values must be finite")
        if np.any(np.diff(fields) <= 0.0):
            raise ModelConfigurationError(
                "Reduced-field table axis must be strictly increasing"
            )
        if np.any(fields < 0.0) or np.any(energies < 0.0):
            raise ModelConfigurationError(
                "Reduced field and mean energy must be non-negative"
            )

    def __call__(self, reduced_field_Td: float) -> float:
        field = float(reduced_field_Td)
        lower = self.reduced_field_Td[0]
        upper = self.reduced_field_Td[-1]
        if not math.isfinite(field) or field < lower or field > upper:
            raise StateDomainError(
                f"Reduced field {field!r} Td is outside table bounds [{lower}, {upper}]"
            )
        return float(np.interp(field, self.reduced_field_Td, self.mean_energy_eV))


@dataclass(frozen=True)
class LocalFieldClosure:
    """Algebraic local-field closure; it never adds electron energy to the ODE."""

    mean_energy_from_field: FieldMeanEnergy
    mode: str = "local_field"
    evolves_energy: bool = False

    def evaluate(
        self,
        *,
        net_heavy_charge_density_m3: float,
        energy_density_J_m3: float | None,
        reduced_field_Td: float | None,
        electron_density_m3: float | None = None,
    ) -> ElectronState:
        if energy_density_J_m3 is not None:
            raise StateDomainError(
                "local_field closure must not receive an electron-energy state"
            )
        density = resolved_electron_density(
            net_heavy_charge_density_m3, electron_density_m3
        )
        if reduced_field_Td is None:
            raise StateDomainError(
                "local_field closure requires reduced_field_Td for every zone and segment"
            )
        field = float(reduced_field_Td)
        mean_energy = float(self.mean_energy_from_field(field))
        if not math.isfinite(mean_energy) or mean_energy < 0.0:
            raise StateDomainError(
                "Local-field mean-energy model returned an invalid value"
            )
        return ElectronState(
            density_m3=density,
            mean_energy_eV=mean_energy,
            temperature_eV=(2.0 / 3.0) * mean_energy,
            energy_density_J_m3=density * ELEMENTARY_CHARGE_C * mean_energy,
            reduced_field_Td=field,
        )


__all__ = [
    "ELEMENTARY_CHARGE_C",
    "ElectronClosure",
    "ElectronEnergyClosure",
    "ElectronState",
    "FieldMeanEnergy",
    "LocalFieldClosure",
    "TabulatedMeanEnergy",
    "quasineutral_electron_density",
    "resolved_electron_density",
]
