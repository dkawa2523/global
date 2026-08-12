"""Build the flat ODE state from validated physical initial data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from plasma_global.core.domain import InitialState
from plasma_global.core.exceptions import ModelConfigurationError
from plasma_global.models.electrons import ELEMENTARY_CHARGE_C

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


def _validate_initial_zones(model: CompiledGlobalModel, initial: InitialState) -> None:
    expected_zones = set(model.layout.zone_ids)
    supplied_zones = set(initial.densities_m3_by_zone)
    if supplied_zones != expected_zones:
        raise ModelConfigurationError(
            "Initial density zones must be exactly "
            f"{sorted(expected_zones)}, got {sorted(supplied_zones)}"
        )


def _validate_initial_energy(model: CompiledGlobalModel, initial: InitialState) -> None:
    expected_zones = set(model.layout.zone_ids)
    if model.layout.evolves_electron_energy:
        if set(initial.mean_energy_eV_by_zone) != expected_zones:
            raise ModelConfigurationError(
                "electron_energy closure needs initial mean energy for zones "
                f"{sorted(expected_zones)}"
            )
    elif initial.mean_energy_eV_by_zone:
        raise ModelConfigurationError(
            "local_field closure derives mean energy and rejects an initial "
            "energy state"
        )
    if model.layout.evolves_heavy_energy:
        temperature_zones = set(initial.gas_temperature_K_by_zone)
        if temperature_zones and temperature_zones != expected_zones:
            raise ModelConfigurationError(
                "Evolved gas energy initial temperatures must name every zone"
            )
    elif initial.gas_temperature_K_by_zone:
        raise ModelConfigurationError(
            "Fixed gas energy rejects an initial gas-energy state"
        )


def _pack_zone_state(
    model: CompiledGlobalModel,
    initial: InitialState,
    state: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    density_rows: list[np.ndarray] = []
    gas_temperatures: list[float] = []
    first_segment = model.segments[0]
    for zone_id in model.layout.zone_ids:
        values = initial.densities_m3_by_zone[zone_id]
        if set(values) != set(model.species_ids):
            raise ModelConfigurationError(
                f"Initial densities for zone {zone_id!r} must explicitly name "
                "every species "
                f"{list(model.species_ids)}"
            )
        density = np.array(
            [values[species_id] for species_id in model.species_ids], dtype=float
        )
        state[model.layout.density_slices[zone_id]] = density
        density_rows.append(density)
        electron_density = model._electron_density(
            first_segment.start_s,
            zone_id,
            float(model.charges @ density),
            first_segment,
        )
        if model.layout.evolves_electron_energy:
            mean_energy = initial.mean_energy_eV_by_zone[zone_id]
            if electron_density == 0.0 and mean_energy != 0.0:
                raise ModelConfigurationError(
                    f"Zone {zone_id!r} cannot initialize nonzero electron "
                    "energy at zero electron density"
                )
            state[model.layout.electron_energy_indices[zone_id]] = (
                electron_density * ELEMENTARY_CHARGE_C * mean_energy
            )
        if model.layout.evolves_heavy_energy:
            gas_temperatures.append(
                initial.gas_temperature_K_by_zone.get(
                    zone_id, model._zone_by_id[zone_id].gas_temperature_K
                )
            )
    return density_rows, gas_temperatures


def _pack_heavy_energy(
    model: CompiledGlobalModel,
    state: np.ndarray,
    density_rows: list[np.ndarray],
    gas_temperatures: list[float],
) -> None:
    if not model.layout.evolves_heavy_energy:
        return
    closure = model.heavy_energy_closure
    if closure is None:
        raise RuntimeError("Heavy-energy state requires an energy closure")
    energies = closure.energy_J_m3(
        np.asarray(density_rows), np.asarray(gas_temperatures)
    )
    for zone_id, energy in zip(model.layout.zone_ids, energies, strict=True):
        state[model.layout.heavy_energy_indices[zone_id]] = energy


def _pack_surface_state(
    model: CompiledGlobalModel, initial: InitialState, state: np.ndarray
) -> None:
    surface_model = model.surface_model
    if surface_model is None:
        return
    coverage = surface_model.initial_state()
    known_surfaces = {surface.surface_id for surface in surface_model.surfaces}
    unknown_surfaces = set(initial.surface_coverages) - known_surfaces
    if unknown_surfaces:
        raise ModelConfigurationError(
            f"Initial coverages reference unknown surfaces {sorted(unknown_surfaces)}"
        )
    for surface_id, values in initial.surface_coverages.items():
        for species_id, value in values.items():
            local_index = surface_model.layout.state_index.get((surface_id, species_id))
            if local_index is None:
                raise ModelConfigurationError(
                    f"Initial coverage {(surface_id, species_id)!r} is not an "
                    "independent state"
                )
            coverage[local_index] = value
    for surface in surface_model.surfaces:
        free_species = surface_model.layout.free_species_by_surface[surface.surface_id]
        if (
            surface_model.coverage(coverage, surface.surface_id, free_species)
            < -1.0e-12
        ):
            raise ModelConfigurationError(
                f"Initial coverage on surface {surface.surface_id!r} exceeds "
                "site occupancy"
            )
    state[model.layout.surface_coverage_slice] = coverage


def _pack_extension_state(model: CompiledGlobalModel, state: np.ndarray) -> None:
    accumulator = model.extension_accumulator
    if accumulator is None:
        return
    extension_state = np.asarray(accumulator.initial_state(), dtype=float)
    expected = model.layout.extension_slice.stop - model.layout.extension_slice.start
    if extension_state.shape != (expected,) or not np.all(np.isfinite(extension_state)):
        raise ModelConfigurationError(
            "Extension initial state must match its compiled state block"
        )
    state[model.layout.extension_slice] = extension_state


def build_initial_state(
    model: CompiledGlobalModel, initial: InitialState
) -> np.ndarray:
    """Validate and pack every independently owned state block once."""

    _validate_initial_zones(model, initial)
    _validate_initial_energy(model, initial)
    if model.surface_model is None and initial.surface_coverages:
        raise ModelConfigurationError(
            "Initial surface coverages require CompiledSurfaceModel"
        )
    state = np.zeros(model.layout.size, dtype=float)
    density_rows, gas_temperatures = _pack_zone_state(model, initial, state)
    _pack_heavy_energy(model, state, density_rows, gas_temperatures)
    _pack_surface_state(model, initial, state)
    _pack_extension_state(model, state)
    return state


__all__ = ["build_initial_state"]
