from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from plasma_global.observables.fields import observable_id
from plasma_global.physics.gas_reactions import gas_reaction_terms

BudgetValues = dict[tuple[str, str], float]


def reaction_budget_observable_fields(gas_core: Any, coupled: Any) -> dict[str, float]:
    reaction_rates: BudgetValues = {}
    species_source: BudgetValues = {}
    species_loss: BudgetValues = {}
    wall_loss: BudgetValues = {}
    electron_energy_loss: BudgetValues = {}

    for term in gas_reaction_terms(gas_core.system, gas_core.gas_reactions, coupled):
        reaction_rates[(term.zone_id, term.reaction_id)] = float(term.rate_m3_s)
        for _idx, change, species_id in term.species_changes:
            _add_species_change(species_source, species_loss, term.zone_id, species_id, change)
        _add_positive(electron_energy_loss, term.zone_id, term.reaction_id, term.electron_energy_loss_J_m3_s)

    for term in gas_core.ion_wall_loss_terms(coupled):
        _add(wall_loss, term.zone_id, term.species_id, term.loss_m3_s)
        _add_species_change(species_source, species_loss, term.zone_id, term.species_id, -term.loss_m3_s)
        _add_positive(electron_energy_loss, term.zone_id, 'ion_wall', term.electron_energy_loss_J_m3_s)

    out: dict[str, float] = {}
    _extend_budget_fields(out, 'reaction_rate', 'm3_s', reaction_rates.items(), include_zero=False)
    _extend_budget_fields(out, 'wall_loss', 'm3_s', wall_loss.items())
    _extend_budget_fields(out, 'electron_energy_loss', 'J_m3_s', electron_energy_loss.items())
    _extend_budget_fields(out, 'species_source', 'm3_s', species_source.items())
    _extend_budget_fields(out, 'species_loss', 'm3_s', species_loss.items())
    return out


def _add(values: BudgetValues, zone_id: str, item_id: str, value: float) -> None:
    key = (zone_id, item_id)
    values[key] = values.get(key, 0.0) + float(value)


def _add_positive(values: BudgetValues, zone_id: str, item_id: str, value: float) -> None:
    if value > 0.0:
        _add(values, zone_id, item_id, value)


def _add_species_change(
    source: BudgetValues,
    loss: BudgetValues,
    zone_id: str,
    species_id: str,
    value_m3_s: float,
) -> None:
    if value_m3_s >= 0.0:
        _add(source, zone_id, species_id, value_m3_s)
    else:
        _add(loss, zone_id, species_id, abs(value_m3_s))


def _extend_budget_fields(
    out: dict[str, float],
    prefix: str,
    unit: str,
    items: Iterable[tuple[tuple[str, str], float]],
    *,
    include_zero: bool = True,
) -> None:
    for (zone_id, item_id), value in sorted(items):
        if value > 0.0 or (include_zero and value != 0.0):
            out[f'{prefix}_{observable_id(zone_id)}_{observable_id(item_id)}_{unit}'] = float(value)
