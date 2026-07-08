from __future__ import annotations

from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.physics.gas_rates import gas_rate_coefficient
from plasma_global.physics.types import CompiledGasReaction, CoupledPlasmaEvaluation, GasReactionTerm


def compile_gas_reactions(system: Any) -> list[CompiledGasReaction]:
    compiled: list[CompiledGasReaction] = []
    for rxn in system.mechanism.gas_reactions:
        if not rxn.enabled:
            continue
        reactant_gas: list[tuple[int, float, str]] = []
        electron_sto = 0.0
        for sp_id, nu in rxn.reactants.items():
            if sp_id == system.mechanism.electron_species_id:
                electron_sto += nu
            elif sp_id in system.gas_species_index:
                reactant_gas.append((system.gas_species_index[sp_id], float(nu), sp_id))
        delta_gas = [
            (idx, float(rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)), sp_id)
            for sp_id, idx in system.gas_species_index.items()
            if abs(rxn.products.get(sp_id, 0.0) - rxn.reactants.get(sp_id, 0.0)) > 0.0
        ]
        compiled.append(
            CompiledGasReaction(
                reaction_id=rxn.reaction_id,
                zones=rxn.zone_filter or system.zone_ids,
                reactant_gas=reactant_gas,
                electron_reactant_stoich=electron_sto,
                delta_gas=delta_gas,
                rate_model=system.mechanism.model(rxn.rate_model_key),
                energy_model=system.mechanism.model(rxn.energy_model_key) if rxn.energy_model_key else None,
            )
        )
    return compiled


def reaction_mass_action(system: Any, rxn: CompiledGasReaction, gas_row: np.ndarray, ne: float) -> float:
    mass_action = 1.0
    for idx, nu, _sp in rxn.reactant_gas:
        density = max(float(gas_row[idx]), 0.0)
        if density <= 0.0 and nu > 0.0:
            return 0.0
        mass_action *= density ** nu
    if rxn.electron_reactant_stoich:
        electron_density = reactive_electron_density(system, gas_row, ne)
        if electron_density <= 0.0 and rxn.electron_reactant_stoich > 0.0:
            return 0.0
        mass_action *= electron_density ** rxn.electron_reactant_stoich
    return float(mass_action)


def reactive_electron_density(system: Any, gas_row: np.ndarray, ne: float) -> float:
    closure = str(getattr(system, 'electron_density_closure', 'quasi_neutral') or 'quasi_neutral').lower()
    if closure == 'prescribed_profile':
        return max(float(ne), 0.0)
    gas_charges = getattr(system, 'gas_charges', None)
    if gas_charges is None:
        return max(float(ne), 0.0)
    return max(float(np.dot(gas_charges, gas_row)), 0.0)


def reaction_rate(system: Any, rxn: CompiledGasReaction, gas_row: np.ndarray, Tg: float, eedf: Any, ne: float, pressure_Pa: float | None = None) -> float:
    k = gas_rate_coefficient(rxn.rate_model, Tg, eedf, pressure_Pa)
    return float(k * reaction_mass_action(system, rxn, gas_row, ne))


def reaction_energy_loss_J_m3_s(rxn: CompiledGasReaction, rate_m3_s: float) -> float:
    if rxn.energy_model and str(rxn.energy_model.get('backend', '')).lower() == 'constant_event_loss':
        return float(rxn.energy_model.get('energy_loss_eV', 0.0)) * E_CHARGE * rate_m3_s
    return 0.0


def gas_reaction_terms(system: Any, gas_reactions: list[CompiledGasReaction], coupled: CoupledPlasmaEvaluation) -> list[GasReactionTerm]:
    terms: list[GasReactionTerm] = []
    for rxn in gas_reactions:
        for zone_id in rxn.zones:
            z = system.zone_index[zone_id]
            ne = coupled.ne_by_zone[zone_id]
            rate = reaction_rate(
                rxn=rxn,
                gas_row=coupled.gas[z],
                Tg=float(coupled.gas_temperature[z]),
                eedf=coupled.eedf_by_zone[zone_id],
                ne=ne,
                pressure_Pa=coupled.pressure_by_zone[zone_id],
                system=system,
            )
            if rate == 0.0:
                continue
            changes = [(idx, float(nu) * rate, sp_id) for idx, nu, sp_id in rxn.delta_gas]
            terms.append(GasReactionTerm(zone_id, rxn.reaction_id, rate, changes, reaction_energy_loss_J_m3_s(rxn, rate)))
    return terms
