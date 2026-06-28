from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any

from plasma_global.observables.fields import observable_id


@dataclass
class ReactionBudget:
    reaction_rates: dict[tuple[str, str], float] = field(default_factory=dict)
    species_source: dict[tuple[str, str], float] = field(default_factory=dict)
    species_loss: dict[tuple[str, str], float] = field(default_factory=dict)
    wall_loss: dict[tuple[str, str], float] = field(default_factory=dict)
    electron_energy_loss: dict[tuple[str, str], float] = field(default_factory=dict)

    def add_species_change(self, zone_id: str, species_id: str, value_m3_s: float) -> None:
        key = (zone_id, species_id)
        if value_m3_s >= 0.0:
            self.species_source[key] = self.species_source.get(key, 0.0) + float(value_m3_s)
        else:
            self.species_loss[key] = self.species_loss.get(key, 0.0) + abs(float(value_m3_s))

    def add_reaction_rate(self, zone_id: str, reaction_id: str, rate_m3_s: float) -> None:
        self.reaction_rates[(zone_id, reaction_id)] = float(rate_m3_s)

    def add_wall_loss(self, zone_id: str, species_id: str, loss_m3_s: float) -> None:
        value = float(loss_m3_s)
        self.wall_loss[(zone_id, species_id)] = self.wall_loss.get((zone_id, species_id), 0.0) + value
        self.add_species_change(zone_id, species_id, -value)

    def add_electron_energy_loss(self, zone_id: str, source_id: str, loss_J_m3_s: float) -> None:
        value = float(loss_J_m3_s)
        if value > 0.0:
            key = (zone_id, source_id)
            self.electron_energy_loss[key] = self.electron_energy_loss.get(key, 0.0) + value


def flatten_reaction_budget(budget: ReactionBudget) -> dict[str, float]:
    out: dict[str, float] = {}
    _extend_budget_fields(out, 'reaction_rate', 'm3_s', budget.reaction_rates.items(), include_zero=False)
    _extend_budget_fields(out, 'wall_loss', 'm3_s', budget.wall_loss.items())
    _extend_budget_fields(out, 'electron_energy_loss', 'J_m3_s', budget.electron_energy_loss.items())
    _extend_budget_fields(out, 'species_source', 'm3_s', budget.species_source.items())
    _extend_budget_fields(out, 'species_loss', 'm3_s', budget.species_loss.items())
    return out


def reaction_source_loss_budget(gas_core: Any, coupled: Any) -> ReactionBudget:
    budget = ReactionBudget()
    for term in gas_core.gas_reaction_terms(coupled):
        budget.add_reaction_rate(term.zone_id, term.reaction_id, term.rate_m3_s)
        for _idx, change, sp_id in term.species_changes:
            budget.add_species_change(term.zone_id, sp_id, change)
        budget.add_electron_energy_loss(term.zone_id, term.reaction_id, term.electron_energy_loss_J_m3_s)
    wall_energy_loss_by_zone: dict[str, float] = {}
    for term in gas_core.ion_wall_loss_terms(coupled):
        budget.add_wall_loss(term.zone_id, term.species_id, term.loss_m3_s)
        wall_energy_loss_by_zone[term.zone_id] = wall_energy_loss_by_zone.get(term.zone_id, 0.0) + term.electron_energy_loss_J_m3_s
    for zone_id, loss in wall_energy_loss_by_zone.items():
        budget.add_electron_energy_loss(zone_id, 'ion_wall', loss)
    return budget


def reaction_budget_observable_fields(gas_core: Any, coupled: Any) -> dict[str, float]:
    return flatten_reaction_budget(reaction_source_loss_budget(gas_core, coupled))


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
