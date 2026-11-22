"""Pack physical initial conditions through the compiled state owners."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from plasma_global.core._runtime_state import RuntimeStatePreparer
from plasma_global.core.domain import InitialState
from plasma_global.errors import (
    CouplingConvergenceError,
    IntegrationError,
    ModelConfigurationError,
    ModelDomainError,
)
from plasma_global.models.electrons import ELEMENTARY_CHARGE_C

if TYPE_CHECKING:
    from plasma_global.core._state_blocks import StateBlockLayout
    from plasma_global.core.compiled import CompiledGlobalModel


def _initial_densities(
    model: CompiledGlobalModel, zone_id: str, initial: InitialState
) -> np.ndarray:
    values = initial.densities_m3_by_zone[zone_id]
    if set(values) != set(model.species_ids):
        raise ModelConfigurationError(
            f"Initial densities for zone {zone_id!r} must explicitly name "
            f"every species {list(model.species_ids)}"
        )
    return np.array([values[name] for name in model.species_ids], dtype=float)


def _initial_electron_energy(
    model: CompiledGlobalModel, zone_id: str, initial: InitialState
) -> np.ndarray:
    density = _initial_densities(model, zone_id, initial)
    segment = model.segments[0]
    electron_density = RuntimeStatePreparer(model).electron_density(
        segment.start_s, zone_id, float(model.charges @ density), segment
    )
    mean_energy = initial.mean_energy_eV_by_zone[zone_id]
    if electron_density == 0.0 and mean_energy != 0.0:
        raise ModelConfigurationError(
            f"Zone {zone_id!r} cannot initialize nonzero electron "
            "energy at zero electron density"
        )
    return np.array([electron_density * ELEMENTARY_CHARGE_C * mean_energy])


def _initial_heavy_energy(
    model: CompiledGlobalModel, zone_id: str, initial: InitialState
) -> np.ndarray:
    closure = model.heavy_energy_closure
    if closure is None:
        raise RuntimeError("Heavy-energy state requires an energy closure")
    temperature = initial.gas_temperature_K_by_zone.get(
        zone_id, model._zone_by_id[zone_id].gas_temperature_K
    )
    density = _initial_densities(model, zone_id, initial)
    return closure.energy_J_m3(density.reshape(1, -1), np.array([temperature]))


def _initial_coverage(model: CompiledGlobalModel, initial: InitialState) -> np.ndarray:
    surface_model = model.surface_model
    if surface_model is None:
        raise RuntimeError("Coverage state requires its surface model")
    coverage = surface_model.initial_state()
    known = {surface.surface_id for surface in surface_model.surfaces}
    unknown = set(initial.surface_coverages) - known
    if unknown:
        raise ModelConfigurationError(
            f"Initial coverages reference unknown surfaces {sorted(unknown)}"
        )
    for surface_id, values in initial.surface_coverages.items():
        for species_id, value in values.items():
            index = surface_model.layout.state_index.get((surface_id, species_id))
            if index is None:
                raise ModelConfigurationError(
                    f"Initial coverage {(surface_id, species_id)!r} is not an "
                    "independent state"
                )
            coverage[index] = value
    return coverage


def _initial_inventory(
    model: CompiledGlobalModel, index: int, initial: InitialState
) -> np.ndarray:
    accumulator = model.extension_accumulator
    if accumulator is None:
        raise RuntimeError("Inventory state requires its accumulator")
    return np.asarray(accumulator.initial_state(), dtype=float)[[index]]


def block_initializer(
    model: CompiledGlobalModel, layout: StateBlockLayout
) -> Callable[[InitialState], np.ndarray]:
    if layout.kind == "density":
        return partial(_initial_densities, model, layout.owner_id)
    if layout.kind == "electron_energy":
        return partial(_initial_electron_energy, model, layout.owner_id)
    if layout.kind == "heavy_energy":
        return partial(_initial_heavy_energy, model, layout.owner_id)
    if layout.kind == "coverage":
        return partial(_initial_coverage, model)
    index = layout.indices[0] - model.layout.extension_slice.start
    return partial(_initial_inventory, model, index)


def _validate_initial_fields(model: CompiledGlobalModel, initial: InitialState) -> None:
    zones = set(model.layout.zone_ids)
    if set(initial.densities_m3_by_zone) != zones:
        raise ModelConfigurationError(
            f"Initial density zones must be exactly {sorted(zones)}, "
            f"got {sorted(initial.densities_m3_by_zone)}"
        )
    _validate_initial_energy_fields(model, initial, zones)
    if model.surface_model is None and initial.surface_coverages:
        raise ModelConfigurationError(
            "Initial surface coverages require CompiledSurfaceModel"
        )


def _validate_initial_energy_fields(
    model: CompiledGlobalModel, initial: InitialState, zones: set[str]
) -> None:
    if model.layout.evolves_electron_energy:
        if set(initial.mean_energy_eV_by_zone) != zones:
            raise ModelConfigurationError(
                "electron_energy closure needs initial mean energy for zones "
                f"{sorted(zones)}"
            )
    elif initial.mean_energy_eV_by_zone:
        raise ModelConfigurationError(
            "local_field closure derives mean energy and rejects an "
            "initial energy state"
        )
    if model.layout.evolves_heavy_energy:
        if (
            initial.gas_temperature_K_by_zone
            and set(initial.gas_temperature_K_by_zone) != zones
        ):
            raise ModelConfigurationError(
                "Evolved gas energy initial temperatures must name every zone"
            )
    elif initial.gas_temperature_K_by_zone:
        raise ModelConfigurationError(
            "Fixed gas energy rejects an initial gas-energy state"
        )


def build_initial_state(
    model: CompiledGlobalModel, initial: InitialState
) -> np.ndarray:
    _validate_initial_fields(model, initial)
    state = np.zeros(model.layout.size)
    for block in model.state_blocks:
        values = block.initialize(initial)
        indices = list(block.layout.indices)
        if values.shape != (len(indices),) or not np.all(np.isfinite(values)):
            raise ModelConfigurationError(
                f"Initial {block.layout.kind} state must match its compiled "
                "block and be finite"
            )
        state[indices] = values
    return state


def validate_initial_state(
    model: CompiledGlobalModel, initial: InitialState | np.ndarray
) -> np.ndarray:
    """Check the full initial coupled domain before constructing an integrator."""

    from plasma_global.core._solver_domain import canonicalize_result_state

    try:
        state = (
            build_initial_state(model, initial)
            if isinstance(initial, InitialState)
            else np.array(initial, dtype=float, copy=True)
        )
    except (ModelDomainError, CouplingConvergenceError) as exc:
        raise ModelConfigurationError(
            f"Initial state violates its model domain: {exc}"
        ) from exc
    if state.shape != (model.layout.size,):
        raise ModelConfigurationError(
            f"Initial state has shape {state.shape}, expected {(model.layout.size,)}"
        )
    if not np.all(np.isfinite(state)):
        raise ModelConfigurationError("Initial state must contain only finite values")
    for block in model.state_blocks:
        values = state[list(block.layout.indices)]
        if block.domain.nonnegative and np.any(values < 0.0):
            raise ModelConfigurationError("Initial state must be nonnegative")
    segment = model.segments[0]
    try:
        canonicalize_result_state(
            model,
            state.reshape(1, -1),
            np.zeros(state.size),
            np.array([segment.start_s]),
            segment=segment,
        )
        model.evaluate_derivative(segment.start_s, state, segment)
    except (IntegrationError, ModelDomainError, CouplingConvergenceError) as exc:
        raise ModelConfigurationError(
            f"Initial state violates its model domain: {exc}"
        ) from exc
    return state


__all__ = ["build_initial_state", "validate_initial_state"]
