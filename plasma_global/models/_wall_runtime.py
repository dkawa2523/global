"""Evaluate compiled wall transport and reaction kernels."""

from __future__ import annotations

import math

import numpy as np

from plasma_global.errors import StateDomainError
from plasma_global.models.electrons import ELEMENTARY_CHARGE_C, ElectronState
from plasma_global.models.walls import (
    ELECTRON_MASS_KG,
    CompiledWallBoundary,
    WallEvaluation,
    WallFluxRecord,
)


def _resolved_sheath_energy_eV(
    compiled: CompiledWallBoundary,
    electrons: ElectronState,
    incident_flux_m2_s: np.ndarray,
) -> float:
    boundary = compiled.boundary
    if not compiled.standard_floating_wall:
        return boundary.sheath_energy_eV
    active = incident_flux_m2_s > 0.0
    if np.any(compiled.charges[active] != 1.0):
        raise StateDomainError(
            "Automatic floating-sheath closure requires active singly charged ions; "
            "provide explicit sheath_energy_eV for active multiply charged ions"
        )
    positive_current_flux = float(np.dot(compiled.charges, incident_flux_m2_s))
    electron_thermal_flux = electrons.density_m3 * math.sqrt(
        ELEMENTARY_CHARGE_C
        * electrons.temperature_eV
        / (2.0 * math.pi * ELECTRON_MASS_KG)
    )
    if positive_current_flux == 0.0 and electron_thermal_flux == 0.0:
        return 0.0
    if positive_current_flux <= 0.0 or electron_thermal_flux <= positive_current_flux:
        raise StateDomainError(
            "Automatic floating-sheath current balance has no nonnegative solution; "
            "provide explicit sheath_energy_eV"
        )
    return electrons.temperature_eV * math.log(
        electron_thermal_flux / positive_current_flux
    )


def _loss_frequency(compiled: CompiledWallBoundary) -> float:
    boundary = compiled.boundary
    if (
        boundary.transport_kind == "prescribed_frequency"
        and boundary.prescribed_frequency_s_inv is not None
    ):
        return boundary.prescribed_frequency_s_inv
    if (
        boundary.transport_kind == "ambipolar"
        and boundary.diffusion_coefficient_m2_s is not None
        and boundary.diffusion_length_m is not None
    ):
        return boundary.diffusion_coefficient_m2_s / boundary.diffusion_length_m**2
    raise RuntimeError("compiled wall transport parameters are inconsistent")


def _bohm_h_factor(compiled: CompiledWallBoundary, densities_m3: np.ndarray) -> float:
    closure = compiled.boundary.auto_bohm_h_factor
    if closure is None:
        return compiled.boundary.bohm_factor
    neutral_density = float(
        np.sum(np.maximum(densities_m3[compiled.neutral_indices], 0.0))
    )
    return closure.evaluate(neutral_density)


def _ion_transport(
    compiled: CompiledWallBoundary,
    ion_slot: int,
    ion_density: float,
    densities_m3: np.ndarray,
    electrons: ElectronState,
    area_over_volume: float,
) -> tuple[float, float, float]:
    boundary = compiled.boundary
    if boundary.transport_kind == "bohm":
        charge = float(compiled.charges[ion_slot])
        mass = float(compiled.masses_kg[ion_slot])
        speed = _bohm_h_factor(compiled, densities_m3) * math.sqrt(
            charge * ELEMENTARY_CHARGE_C * electrons.temperature_eV / mass
        )
        flux = ion_density * speed
        return flux, flux * area_over_volume, speed
    frequency = _loss_frequency(compiled)
    rate = ion_density * frequency
    return rate / area_over_volume, rate, frequency / area_over_volume


def _transported_ions(
    compiled: CompiledWallBoundary,
    densities_m3: np.ndarray,
    electrons: ElectronState,
    area_over_volume: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = compiled.ion_indices.size
    fluxes = np.empty(count)
    rates = np.empty(count)
    speeds = np.empty(count)
    for ion_slot, species_index in enumerate(compiled.ion_indices):
        ion_density = max(float(densities_m3[int(species_index)]), 0.0)
        fluxes[ion_slot], rates[ion_slot], speeds[ion_slot] = _ion_transport(
            compiled,
            ion_slot,
            ion_density,
            densities_m3,
            electrons,
            area_over_volume,
        )
    return fluxes, rates, speeds


def _apply_branches(
    compiled: CompiledWallBoundary,
    ion_slot: int,
    incident_rate: float,
    derivative: np.ndarray,
    collect_records: bool,
) -> dict[str, float]:
    branch_rates: dict[str, float] = {}
    for branch in compiled.branches_by_ion[ion_slot]:
        branch_rate = incident_rate * branch.probability
        if collect_records:
            branch_rates[branch.reaction_id] = branch_rate
        if branch.product_indices.size:
            np.add.at(
                derivative,
                branch.product_indices,
                branch.product_yields * branch_rate,
            )
    return branch_rates


def evaluate_wall(
    *,
    compiled: CompiledWallBoundary,
    volume_m3: float,
    species_ids: tuple[str, ...],
    densities_m3: np.ndarray,
    electrons: ElectronState,
    collect_records: bool,
) -> WallEvaluation:
    """Evaluate an indexed wall kernel without rebuilding static mappings."""

    boundary = compiled.boundary
    derivative = np.zeros_like(densities_m3, dtype=float)
    flux_by_species = np.zeros_like(densities_m3, dtype=float)
    area_over_volume = boundary.area_m2 / volume_m3
    if boundary.transport_kind == "off":
        return WallEvaluation(derivative, 0.0, (), flux_by_species, 0.0)

    records: list[WallFluxRecord] = []
    total_energy_loss = 0.0
    ion_fluxes, ion_rates, transport_speeds = _transported_ions(
        compiled, densities_m3, electrons, area_over_volume
    )
    sheath_energy_eV = _resolved_sheath_energy_eV(compiled, electrons, ion_fluxes)
    for ion_slot, species_index_raw in enumerate(compiled.ion_indices):
        species_index = int(species_index_raw)
        incident_flux = float(ion_fluxes[ion_slot])
        incident_rate = float(ion_rates[ion_slot])
        transport_speed = float(transport_speeds[ion_slot])
        flux_by_species[species_index] = incident_flux
        derivative[species_index] -= incident_rate
        branch_rates = _apply_branches(
            compiled, ion_slot, incident_rate, derivative, collect_records
        )
        charge = float(compiled.charges[ion_slot])
        loss_per_ion_eV = charge * 2.0 * electrons.temperature_eV + sheath_energy_eV
        energy_loss = incident_rate * loss_per_ion_eV * ELEMENTARY_CHARGE_C
        total_energy_loss += energy_loss
        if collect_records:
            records.append(
                WallFluxRecord(
                    zone_id=boundary.zone_id,
                    surface_id=boundary.surface_id,
                    transport_kind=boundary.transport_kind,
                    incident_species=species_ids[species_index],
                    incident_flux_m2_s=incident_flux,
                    incident_rate_m3_s=incident_rate,
                    bohm_speed_m_s=transport_speed,
                    electron_energy_loss_J_m3_s=energy_loss,
                    branch_rates_m3_s=branch_rates,
                )
            )
    return WallEvaluation(
        derivative,
        total_energy_loss,
        tuple(records),
        flux_by_species,
        sheath_energy_eV,
    )
