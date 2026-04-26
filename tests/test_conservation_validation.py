from __future__ import annotations

from pathlib import Path

from plasma_global.chemistry.models import MechanismBundle, Reaction, Species
from plasma_global.chemistry.validators import validate_mechanism
from plasma_global.workflows.context import load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_example_mechanism_has_no_conservation_errors() -> None:
    loaded = load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_smoke.yaml')
    report = validate_mechanism(loaded.mechanism)
    errors = [msg for msg in report.messages if msg.level == 'ERROR']
    assert errors == []


def _argon_loss_mechanism(reactions: list[Reaction], rate_models: dict[str, dict]) -> MechanismBundle:
    return MechanismBundle(
        species=[
            Species('e', 'e', 'gas', -1, 0.00054858, {}, state_tags={'electron'}),
            Species('Ar', 'Ar', 'gas', 0, 39.948, {'Ar': 1}),
            Species('Ar_star', 'Ar*', 'gas', 0, 39.948, {'Ar': 1}, state_tags={'metastable'}),
        ],
        gas_reactions=reactions,
        surface_reactions=[],
        rate_models=rate_models,
        cross_sections={},
        aliases={},
    )


def test_first_order_loss_accepts_single_non_electron_reactant() -> None:
    mechanism = _argon_loss_mechanism(
        [
            Reaction(
                reaction_id='ARSTAR_DIFFUSION_LOSS',
                phase='gas',
                equation='Ar_star -> Ar',
                reactants={'Ar_star': 1.0},
                products={'Ar': 1.0},
                rate_model_key='RM_ARSTAR_DIFFUSION_LOSS',
                energy_model_key=None,
                zone_filter=['plasma'],
                surface_filter=[],
                enabled=True,
            )
        ],
        {'RM_ARSTAR_DIFFUSION_LOSS': {'backend': 'first_order_loss', 'rate_s_inv': 2.0e5}},
    )

    errors = [msg.code for msg in validate_mechanism(mechanism).messages if msg.level == 'ERROR']

    assert errors == []


def test_first_order_loss_rejects_electron_impact_form() -> None:
    mechanism = _argon_loss_mechanism(
        [
            Reaction(
                reaction_id='BAD_ARSTAR_ELECTRON_LOSS',
                phase='gas',
                equation='e + Ar_star -> e + Ar',
                reactants={'e': 1.0, 'Ar_star': 1.0},
                products={'e': 1.0, 'Ar': 1.0},
                rate_model_key='RM_BAD_LOSS',
                energy_model_key=None,
                zone_filter=['plasma'],
                surface_filter=[],
                enabled=True,
            )
        ],
        {'RM_BAD_LOSS': {'backend': 'first_order_loss', 'rate_s_inv': 2.0e5}},
    )

    errors = [msg.code for msg in validate_mechanism(mechanism).messages if msg.level == 'ERROR']

    assert 'FIRST_ORDER_LOSS_REACTION_FORM' in errors
