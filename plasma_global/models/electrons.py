"""Mutually exclusive electron closures for the compact global model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from plasma_global.models.kinetics import ElectronKineticsResult

from plasma_global.errors import (
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

    mode: ClassVar[Literal["electron_energy"]] = "electron_energy"
    evolves_energy: ClassVar[bool] = True

    def canonical_energy_density(
        self,
        *,
        net_heavy_charge_density_m3: float,
        energy_density_J_m3: float,
        energy_atol_J_m3: float,
        electron_density_m3: float | None = None,
    ) -> float:
        """Validate accepted energy and map only a tolerance-scale cold residue."""

        density = resolved_electron_density(
            net_heavy_charge_density_m3, electron_density_m3
        )
        energy = float(energy_density_J_m3)
        if density == 0.0 and energy > 0.0:
            if energy > 10.0 * energy_atol_J_m3:
                raise StateDomainError(
                    "cold electron energy at zero density exceeds 10*domain_atol"
                )
            energy = 0.0
        self.evaluate(
            net_heavy_charge_density_m3=net_heavy_charge_density_m3,
            energy_density_J_m3=energy,
            reduced_field_Td=None,
            electron_density_m3=density,
        )
        return energy

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
            denominator = density * ELEMENTARY_CHARGE_C
            mean_energy = math.inf if denominator == 0.0 else energy / denominator
            if not math.isfinite(mean_energy):
                raise StateDomainError(
                    "Electron mean energy must be finite at positive density"
                )
        return ElectronState(
            density_m3=density,
            mean_energy_eV=mean_energy,
            temperature_eV=(2.0 / 3.0) * mean_energy,
            energy_density_J_m3=energy,
            reduced_field_Td=reduced_field_Td,
        )


@dataclass(frozen=True)
class LocalFieldClosure:
    """Use the coefficient table's electron state without evolving its energy."""

    mode: ClassVar[Literal["local_field"]] = "local_field"
    evolves_energy: ClassVar[bool] = False

    def evaluate(
        self,
        *,
        electron_density_m3: float,
        reduced_field_Td: float | None,
        kinetics: ElectronKineticsResult,
    ) -> ElectronState:
        """Convert the table's algebraic electron state with no energy ODE."""

        density = resolved_electron_density(0.0, electron_density_m3)
        return ElectronState(
            density_m3=density,
            mean_energy_eV=kinetics.mean_energy_eV,
            temperature_eV=kinetics.electron_temperature_eV,
            energy_density_J_m3=density * ELEMENTARY_CHARGE_C * kinetics.mean_energy_eV,
            reduced_field_Td=(
                reduced_field_Td
                if reduced_field_Td is not None
                else kinetics.effective_field_Td
            ),
        )


ElectronClosure = ElectronEnergyClosure | LocalFieldClosure


__all__ = [
    "ELEMENTARY_CHARGE_C",
    "ElectronClosure",
    "ElectronEnergyClosure",
    "ElectronState",
    "LocalFieldClosure",
    "quasineutral_electron_density",
    "resolved_electron_density",
]
