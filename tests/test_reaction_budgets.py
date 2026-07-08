from __future__ import annotations

import pytest

from plasma_global.observables.budgets import reaction_budget_observable_fields
from plasma_global.observables.fields import observable_id
from plasma_global.physics.gas_reactions import gas_reaction_terms
from plasma_global import build_case, load_case_from_yaml
from tests.case_helpers import ZDPLASKIN_CASE


def _budget_key(prefix: str, zone_id: str, item_id: str, unit: str) -> str:
    return f'{prefix}_{observable_id(zone_id)}_{observable_id(item_id)}_{unit}'


def test_gas_terms_and_budget_share_reaction_values() -> None:
    built = build_case(load_case_from_yaml(ZDPLASKIN_CASE))
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))

    terms = gas_reaction_terms(system, system.gas_core.gas_reactions, coupled)
    fields = reaction_budget_observable_fields(system.gas_core, coupled)
    expected_source: dict[tuple[str, str], float] = {}
    expected_loss: dict[tuple[str, str], float] = {}

    assert terms
    for term in terms:
        key = _budget_key('reaction_rate', term.zone_id, term.reaction_id, 'm3_s')
        assert fields[key] == pytest.approx(term.rate_m3_s)
        for _idx, change, species_id in term.species_changes:
            target = expected_source if change >= 0.0 else expected_loss
            key = (term.zone_id, species_id)
            target[key] = target.get(key, 0.0) + abs(change)
    for term in system.gas_core.ion_wall_loss_terms(coupled):
        key = (term.zone_id, term.species_id)
        expected_loss[key] = expected_loss.get(key, 0.0) + term.loss_m3_s

    for (zone_id, species_id), value in expected_source.items():
        key = _budget_key('species_source', zone_id, species_id, 'm3_s')
        assert fields[key] == pytest.approx(value)
    for (zone_id, species_id), value in expected_loss.items():
        key = _budget_key('species_loss', zone_id, species_id, 'm3_s')
        assert fields[key] == pytest.approx(value)


def test_ion_wall_loss_terms_and_budget_share_loss_values() -> None:
    built = build_case(load_case_from_yaml(ZDPLASKIN_CASE))
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))

    terms = system.gas_core.ion_wall_loss_terms(coupled)
    fields = reaction_budget_observable_fields(system.gas_core, coupled)
    expected_wall: dict[tuple[str, str], float] = {}

    assert terms
    for term in terms:
        key = (term.zone_id, term.species_id)
        expected_wall[key] = expected_wall.get(key, 0.0) + term.loss_m3_s

    for (zone_id, species_id), value in expected_wall.items():
        key = _budget_key('wall_loss', zone_id, species_id, 'm3_s')
        assert fields[key] == pytest.approx(value)
