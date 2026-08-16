"""Runtime conservation diagnostics for compiled simulation models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from plasma_global.core.result import SimulationResult
from plasma_global.postprocess import (
    _derive_observables_and_diagnostics,
    derive_observables_and_diagnostics,
)

_PARTICLE_LEDGER = "particle_ledger_normalized"
_ELECTRON_ENERGY_LEDGER = "electron_energy_ledger_normalized"
_HEAVY_ENERGY_LEDGER = "heavy_energy_ledger_normalized"


def _initial_maxima(model: Any, include_detailed_ledgers: bool) -> dict[str, float]:
    maxima = {"charge_closure_normalized": 0.0}
    if not include_detailed_ledgers:
        return maxima
    maxima[_PARTICLE_LEDGER] = 0.0
    if model.layout.evolves_electron_energy:
        maxima[_ELECTRON_ENERGY_LEDGER] = 0.0
    if model.layout.evolves_heavy_energy:
        maxima[_HEAVY_ENERGY_LEDGER] = 0.0
    return maxima


def _surface_source_vectors(compiled: Any) -> dict[str, np.ndarray]:
    model = compiled.model
    surface_model = model.surface_model
    if surface_model is None:
        return {}

    sources: dict[str, np.ndarray] = {}
    for kernel in surface_model._kernels:
        surface = surface_model.surfaces[kernel.surface_index]
        source = np.zeros(len(model.species_ids), dtype=float)
        zone_index = surface_model._zone_index[surface.zone_id]
        area_over_volume = surface.area_m2 / surface_model.zone_volumes_m3[zone_index]
        for species_index, delta in kernel.gas_delta:
            source[species_index] += area_over_volume * delta
        sources[f"{kernel.reaction.id}@{surface.surface_id}"] = source
    return sources


def _wall_species_source(
    model: Any,
    ledger: Any,
    species_index: dict[str, int],
    boundary_products: dict[str, Any],
) -> np.ndarray:
    source = np.zeros(len(model.species_ids), dtype=float)
    for record in ledger.wall_fluxes:
        source[species_index[record.incident_species]] -= record.incident_rate_m3_s
        for reaction_id, branch_rate in record.branch_rates_m3_s.items():
            for product, yield_per_ion in boundary_products[reaction_id].items():
                source[species_index[product]] += yield_per_ion * branch_rate
    return source


def _surface_species_source(
    model: Any,
    ledger: Any,
    surface_sources: dict[str, np.ndarray],
) -> np.ndarray:
    source = np.zeros(len(model.species_ids), dtype=float)
    for rate_id, rate_m2_s in ledger.surface_rates_m2_s.items():
        source += surface_sources[rate_id] * rate_m2_s
    return source


def _particle_ledger_residual(
    model: Any,
    evaluation: Any,
    ledger: Any,
    zone_id: str,
    species_index: dict[str, int],
    boundary_products: dict[str, Any],
    surface_sources: dict[str, np.ndarray],
) -> float:
    rates = np.asarray(
        [ledger.reaction_rates_m3_s[name] for name in model.reaction_ids]
    )
    reaction_source = model.stoichiometry.T @ rates
    transport_source = np.asarray(
        [ledger.transport_species_source_m3_s[name] for name in model.species_ids]
    )
    wall_source = _wall_species_source(model, ledger, species_index, boundary_products)
    surface_source = _surface_species_source(model, ledger, surface_sources)
    density_rhs = evaluation.derivative[model.layout.density_slices[zone_id]]
    reconstructed = reaction_source + transport_source + wall_source + surface_source
    scale = np.maximum.reduce(
        (
            np.abs(density_rhs),
            np.abs(reaction_source)
            + np.abs(transport_source)
            + np.abs(wall_source)
            + np.abs(surface_source),
            np.full(len(model.species_ids), 1.0e-300),
        )
    )
    return float(np.max(np.abs(density_rhs - reconstructed) / scale))


def _energy_ledger_residual(rhs: float, terms: tuple[float, ...]) -> float:
    scale = max(abs(rhs), sum(abs(value) for value in terms), 1.0e-300)
    return abs(rhs - sum(terms)) / scale


def _electron_energy_residual(
    model: Any, evaluation: Any, ledger: Any, zone_id: str
) -> float:
    terms = (
        ledger.absorbed_power_J_m3_s,
        -ledger.reaction_energy_loss_J_m3_s,
        -ledger.wall_energy_loss_J_m3_s,
        -ledger.elastic_heating_J_m3_s,
        ledger.transport_electron_energy_J_m3_s,
    )
    rhs = float(evaluation.derivative[model.layout.electron_energy_indices[zone_id]])
    return _energy_ledger_residual(rhs, terms)


def _heavy_energy_residual(
    model: Any, evaluation: Any, ledger: Any, zone_id: str
) -> float:
    terms = (
        ledger.gas_power_J_m3_s,
        ledger.gas_reaction_heating_J_m3_s,
        ledger.surface_reaction_heating_J_m3_s,
        ledger.wall_heavy_energy_exchange_J_m3_s,
        ledger.elastic_heating_J_m3_s,
        ledger.wall_species_energy_J_m3_s,
        ledger.surface_species_energy_J_m3_s,
        ledger.transport_heavy_energy_J_m3_s,
    )
    rhs = float(evaluation.derivative[model.layout.heavy_energy_indices[zone_id]])
    return _energy_ledger_residual(rhs, terms)


def _update_detailed_maxima(
    maxima: dict[str, float],
    model: Any,
    evaluation: Any,
    zone_id: str,
    species_index: dict[str, int],
    boundary_products: dict[str, Any],
    surface_sources: dict[str, np.ndarray],
) -> None:
    ledger = evaluation.ledger_by_zone[zone_id]
    particle_residual = _particle_ledger_residual(
        model,
        evaluation,
        ledger,
        zone_id,
        species_index,
        boundary_products,
        surface_sources,
    )
    maxima[_PARTICLE_LEDGER] = max(maxima[_PARTICLE_LEDGER], particle_residual)
    if model.layout.evolves_electron_energy:
        residual = _electron_energy_residual(model, evaluation, ledger, zone_id)
        maxima[_ELECTRON_ENERGY_LEDGER] = max(maxima[_ELECTRON_ENERGY_LEDGER], residual)
    if model.layout.evolves_heavy_energy:
        residual = _heavy_energy_residual(model, evaluation, ledger, zone_id)
        maxima[_HEAVY_ENERGY_LEDGER] = max(maxima[_HEAVY_ENERGY_LEDGER], residual)


class _DetailedLedgerDiagnostics:
    """Accumulate conservation ledgers from evaluations already needed by output."""

    def __init__(self, compiled: Any) -> None:
        self.model = compiled.model
        self.maxima = _initial_maxima(self.model, True)
        del self.maxima["charge_closure_normalized"]
        self.species_index = {
            name: index for index, name in enumerate(self.model.species_ids)
        }
        self.boundary_products = {
            reaction.id: reaction.products
            for reaction in compiled.chemistry_data.boundary_reactions
        }
        self.surface_sources = _surface_source_vectors(compiled)

    def observe(self, _state: np.ndarray, evaluation: Any) -> None:
        for zone in self.model.zones:
            _update_detailed_maxima(
                self.maxima,
                self.model,
                evaluation,
                zone.zone_id,
                self.species_index,
                self.boundary_products,
                self.surface_sources,
            )


def _derive_observables_and_runtime_diagnostics(
    compiled: Any,
    result: SimulationResult,
    names: Iterable[str],
) -> tuple[Mapping[str, np.ndarray], dict[str, float]]:
    """Derive selected output and detailed ledgers in one model replay."""

    detailed = _DetailedLedgerDiagnostics(compiled)
    observables, maxima = _derive_observables_and_diagnostics(
        compiled.model,
        result.time_s,
        result.state,
        names,
        collect_ledger=True,
        evaluation_observer=detailed.observe,
    )
    maxima.update(detailed.maxima)
    return observables, maxima


def runtime_diagnostic_maxima(
    compiled: Any,
    result: SimulationResult,
    *,
    include_detailed_ledgers: bool = True,
) -> dict[str, float]:
    """Evaluate charge closure and optional particle/energy ledger closure."""

    if include_detailed_ledgers:
        return _derive_observables_and_runtime_diagnostics(compiled, result, ())[1]
    return derive_observables_and_diagnostics(
        compiled.model,
        result.time_s,
        result.state,
        (),
    )[1]
