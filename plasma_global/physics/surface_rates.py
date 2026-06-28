from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.physics.types import CompiledSurfaceReaction, SurfaceRateContext, SurfaceRateEvaluation


@dataclass
class _SurfaceRateAccumulator:
    core: Any
    context: SurfaceRateContext
    rate: float = 1.0
    used_surface_species: set[str] = field(default_factory=set)

    def add_gas_power(self, idx: int, power_exp: float) -> None:
        n = max(float(self.context.gas_row[idx]), self.core.system.floor_density)
        self.rate *= n ** power_exp

    def add_surface_power(self, state_idx: int, power_exp: float) -> None:
        theta = max(float(np.clip(self.context.state[state_idx], 0.0, 1.0)), 1.0e-12)
        self.rate *= theta ** power_exp

    def add_surface_reactants(self, reactants: list[tuple[int, float, str]]) -> None:
        for state_idx, nu, species_id in reactants:
            if species_id not in self.used_surface_species:
                self.add_surface_power(state_idx, nu)

    def result(self) -> SurfaceRateEvaluation:
        return SurfaceRateEvaluation(rate_m2_s=float(self.rate))


def surface_coverage_factor(core: Any, cfg: dict[str, Any] | None, surface_id: str, y: np.ndarray) -> tuple[float, set[str]]:
    sys = core.system
    if not cfg:
        return 1.0, set()
    kind = str(cfg.get('kind', 'constant')).lower()
    if kind == 'constant':
        return 1.0, set()
    if kind in {'site_blocking', 'species_power'}:
        site_species = str(cfg.get('site_species') or cfg.get('species'))
        exponent = float(cfg.get('exponent', 1.0))
        idx = sys.state_layout.surface_index[surface_id][site_species]
        theta = float(np.clip(y[idx], 0.0, 1.0))
        return max(theta, 1.0e-12) ** exponent, {site_species}
    if kind == 'logistic_switch':
        species = str(cfg['species'])
        midpoint = float(cfg.get('midpoint', 0.5))
        sharpness = float(cfg.get('sharpness', 12.0))
        low = float(cfg.get('low', 0.0))
        high = float(cfg.get('high', 1.0))
        idx = sys.state_layout.surface_index[surface_id][species]
        theta = float(np.clip(y[idx], 0.0, 1.0))
        logistic = 1.0 / (1.0 + np.exp(-sharpness * (theta - midpoint)))
        factor = low + (high - low) * logistic
        return float(factor), {species}
    raise NotImplementedError(f'Unsupported coverage factor kind: {kind}')


def surface_thermal_prefactor(model: dict[str, Any], surface_temperature_K: float) -> float:
    T = max(float(surface_temperature_K), 1.0)
    beta = float(model.get('beta', 0.0))
    Ea_eV = float(model.get('Ea_eV', model.get('activation_eV', 0.0)))
    factor = (T / 300.0) ** beta * np.exp(-Ea_eV * E_CHARGE / (K_B * T))
    return float(factor)


def surface_energy_factor(model: dict[str, Any], ion_energy_eV: float) -> float:
    threshold = float(model.get('threshold_eV', 0.0))
    exponent = float(model.get('energy_exponent', 1.0))
    ref = float(model.get('reference_energy_eV', max(threshold + 1.0, 100.0)))
    e = max(float(ion_energy_eV), 0.0)
    if e <= threshold:
        return 0.0
    return ((e - threshold) / max(ref - threshold, 1.0e-6)) ** exponent


def evaluate_surface_rate(core: Any, context: SurfaceRateContext) -> SurfaceRateEvaluation:
    acc = _SurfaceRateAccumulator(core, context)
    _apply_common_factors(acc)
    backend = str(context.reaction.rate_model.get('backend', '')).lower()
    if backend in {'sticking', 'eley_rideal'}:
        return _sticking_surface_rate(acc)
    if backend == 'ion_assisted':
        return _ion_assisted_surface_rate(acc)
    if backend == 'desorption':
        return _desorption_surface_rate(acc)
    if backend == 'langmuir_hinshelwood':
        return _langmuir_hinshelwood_surface_rate(acc)
    raise NotImplementedError(f'Unsupported surface rate backend: {backend}')


def _apply_common_factors(acc: _SurfaceRateAccumulator) -> None:
    rxn = acc.context.reaction
    model = rxn.rate_model
    thermal_factor = surface_thermal_prefactor(model, acc.context.surface_temperature_K)
    cov_factor, cov_used = surface_coverage_factor(acc.core, model.get('coverage_factor'), rxn.surface_id, acc.context.state)
    acc.rate *= thermal_factor * cov_factor
    acc.used_surface_species |= cov_used


def _sticking_surface_rate(acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
    rxn = acc.context.reaction
    primary = rxn.gas_reactants[0] if rxn.gas_reactants else None
    if primary is None:
        return SurfaceRateEvaluation(0.0)
    idx, _nu, _sp_id = primary
    if acc.core.system.gas_species[idx].charge > 0:
        _apply_ion_sticking_primary(acc, idx)
    else:
        _apply_neutral_sticking_primary(acc, idx)
    for idx2, nu, _sp in rxn.gas_reactants[1:]:
        acc.add_gas_power(idx2, nu)
    acc.add_surface_reactants(rxn.surface_reactants)
    return acc.result()


def _apply_ion_sticking_primary(acc: _SurfaceRateAccumulator, idx: int) -> None:
    model = acc.context.reaction.rate_model
    n_i = max(float(acc.context.gas_row[idx]), acc.core.system.floor_density)
    frac_i = min(max(n_i / max(acc.context.positive_ion_density_m3, acc.core.system.floor_density), 0.0), 1.0)
    acc.add_gas_power(idx, 1.0)
    acc.rate *= acc.context.ion_flux_m2_s * frac_i / n_i
    acc.rate *= surface_energy_factor(model, acc.context.ion_energy_eV)
    acc.rate *= float(model.get('sticking_value', model.get('yield_value', 1.0)))


def _apply_neutral_sticking_primary(acc: _SurfaceRateAccumulator, idx: int) -> None:
    model = acc.context.reaction.rate_model
    Tg = acc.context.gas_temperature_K
    mass = acc.core.system.gas_masses[idx]
    thermal_speed = np.sqrt(8.0 * K_B * max(Tg, 1.0) / (np.pi * max(mass, 1.0e-30)))
    acc.rate *= 0.25 * thermal_speed * float(model.get('sticking_value', model.get('yield_value', 0.0)))
    acc.add_gas_power(idx, 1.0)


def _ion_assisted_surface_rate(acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
    rxn = acc.context.reaction
    primary = _primary_ion_reactant(acc.core, rxn) or (rxn.gas_reactants[0] if rxn.gas_reactants else None)
    if primary is None:
        return SurfaceRateEvaluation(0.0)
    idx, _nu, _sp_id = primary
    energy_factor = surface_energy_factor(rxn.rate_model, acc.context.ion_energy_eV)
    if energy_factor <= 0.0:
        return SurfaceRateEvaluation(0.0)
    n_i = max(float(acc.context.gas_row[idx]), acc.core.system.floor_density)
    frac_i = min(max(n_i / max(acc.context.positive_ion_density_m3, acc.core.system.floor_density), 0.0), 1.0)
    yield_base = float(rxn.rate_model.get('yield_value', rxn.rate_model.get('yield_at_ref', rxn.rate_model.get('sticking_value', 1.0))))
    acc.rate *= acc.context.ion_flux_m2_s * frac_i * yield_base * energy_factor
    acc.add_gas_power(idx, 1.0)
    acc.rate /= n_i
    acc.add_surface_reactants(rxn.surface_reactants)
    for idx2, nu, _sp in rxn.gas_reactants[1:]:
        acc.add_gas_power(idx2, nu)
    return acc.result()


def _primary_ion_reactant(core: Any, rxn: CompiledSurfaceReaction) -> tuple[int, float, str] | None:
    for item in rxn.gas_reactants:
        if core.system.gas_species[item[0]].charge > 0:
            return item
    return None


def _desorption_surface_rate(acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
    rxn = acc.context.reaction
    acc.rate *= float(rxn.rate_model.get('nu0_s_inv', rxn.rate_model.get('prefactor_s_inv', rxn.rate_model.get('A', 0.0)))) * rxn.site_density_m2
    acc.add_surface_reactants(rxn.surface_reactants)
    return acc.result()


def _langmuir_hinshelwood_surface_rate(acc: _SurfaceRateAccumulator) -> SurfaceRateEvaluation:
    rxn = acc.context.reaction
    acc.rate *= float(rxn.rate_model.get('A_m2_s_inv', rxn.rate_model.get('A', 0.0))) * rxn.site_density_m2
    acc.add_surface_reactants(rxn.surface_reactants)
    for idx, nu, _sp in rxn.gas_reactants:
        acc.add_gas_power(idx, nu)
    return acc.result()
