from __future__ import annotations

from pathlib import Path

import pytest

from plasma_global.chemistry.models import MechanismBundle, Reaction, Species
from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.chemistry.provenance import chemistry_provenance_summary
from plasma_global.chemistry.validators import validate_mechanism
from plasma_global.workflows.context import load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_example_mechanism_has_no_conservation_errors() -> None:
    loaded = load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_smoke.yaml')
    report = validate_mechanism(loaded.mechanism)
    errors = [msg for msg in report.messages if msg.level == 'ERROR']
    assert errors == []


def test_example_mechanism_missing_optional_provenance_is_not_error() -> None:
    loaded = load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_smoke.yaml')
    report = validate_mechanism(loaded.mechanism)
    errors = [msg.code for msg in report.messages if msg.level == 'ERROR']

    assert all(not code.startswith('PROVENANCE') for code in errors)


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


def test_reaction_provenance_fixture_loads_and_summarizes(tmp_path: Path) -> None:
    (tmp_path / 'species.csv').write_text(
        '\n'.join(
            [
                'canonical_id,display_name,phase,charge,mass_amu,elements,aliases,state_tags,zones,surfaces',
                'e,e,gas,-1,0.00054858,,,"electron",,',
                'Ar,Ar,gas,0,39.948,Ar:1,,,,',
                'Ar_plus,Ar+,gas,1,39.948,Ar:1,,,,',
            ]
        ),
        encoding='utf-8',
    )
    (tmp_path / 'gas_reactions.csv').write_text(
        '\n'.join(
            [
                'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes,source,reference,version,units,valid_temperature_range,uncertainty,cross_section_id',
                'G_AR_ION,gas,"e + Ar -> Ar_plus + e + e",RM_E_AR_ION,,plasma,,true,test ionization,LXCat,Example reference,2024,m3/s,300-1000 K,20%,xs_ar_ion',
            ]
        ),
        encoding='utf-8',
    )
    (tmp_path / 'surface_reactions.csv').write_text(
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
        encoding='utf-8',
    )
    (tmp_path / 'electron_impact_models.yaml').write_text(
        """
rate_models:
  RM_E_AR_ION:
    backend: electron_impact_xsec
    cross_section_id: xs_ar_ion
    branching_yield: 1.0
    provenance:
      source: model source
      reference: model reference
""",
        encoding='utf-8',
    )
    (tmp_path / 'chemistry_manifest.yaml').write_text(
        """
species_file: species.csv
gas_reactions_file: gas_reactions.csv
surface_reactions_file: surface_reactions.csv
model_files:
  electron_impact: electron_impact_models.yaml
cross_sections_manifest: cross_sections_manifest.yaml
""",
        encoding='utf-8',
    )
    (tmp_path / 'cross_sections_manifest.yaml').write_text(
        """
cross_sections:
  - cross_section_id: xs_ar_ion
    kind: ionization
    target_species: Ar
    threshold_eV: 15.76
    energy_loss_eV: 15.76
    source: surrogate
    metadata:
      provenance:
        source: cross-section source
        reference: cross-section reference
""",
        encoding='utf-8',
    )

    mechanism = load_mechanism_bundle(tmp_path / 'chemistry_manifest.yaml')
    report = validate_mechanism(mechanism)
    summary = chemistry_provenance_summary(mechanism)

    assert [msg for msg in report.messages if msg.level == 'ERROR'] == []
    assert mechanism.gas_reactions[0].provenance['source'] == 'LXCat'
    assert mechanism.gas_reactions[0].provenance['cross_section_id'] == 'xs_ar_ion'
    assert summary['reactions_with_provenance'] == 1
    assert 'reaction_entries' not in summary
    assert summary['rate_models_with_provenance'] == 1
    assert summary['cross_sections_with_provenance'] == 1


def test_chemistry_manifest_rejects_reaction_models_alias(tmp_path: Path) -> None:
    manifest = tmp_path / 'chemistry_manifest.yaml'
    manifest.write_text(
        """
species_file: species.csv
gas_reactions_file: gas_reactions.csv
reaction_models_file: reaction_models.yaml
""",
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='reaction_models_file.*model_files'):
        load_mechanism_bundle(manifest)


@pytest.mark.parametrize('backend', ['arrhenius', 'sputter_yield'])
def test_surface_rate_model_file_rejects_unsupported_backends(tmp_path: Path, backend: str) -> None:
    (tmp_path / 'species.csv').write_text(
        'canonical_id,display_name,phase,charge,mass_amu,elements,aliases,state_tags,zones,surfaces\n',
        encoding='utf-8',
    )
    (tmp_path / 'gas_reactions.csv').write_text(
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
        encoding='utf-8',
    )
    (tmp_path / 'surface_reactions.csv').write_text(
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
        encoding='utf-8',
    )
    (tmp_path / 'surface_rate_models.yaml').write_text(
        f"""
rate_models:
  RM_SURFACE_UNSUPPORTED:
    backend: {backend}
    A: 1.0
""",
        encoding='utf-8',
    )
    (tmp_path / 'chemistry_manifest.yaml').write_text(
        """
species_file: species.csv
gas_reactions_file: gas_reactions.csv
surface_reactions_file: surface_reactions.csv
model_files:
  surface_rate: surface_rate_models.yaml
""",
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='unsupported backends'):
        load_mechanism_bundle(tmp_path / 'chemistry_manifest.yaml')


def test_invalid_rate_model_provenance_warns_without_failing() -> None:
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
        {
            'RM_ARSTAR_DIFFUSION_LOSS': {
                'backend': 'first_order_loss',
                'rate_s_inv': 2.0e5,
                'provenance': 'not-a-mapping',
            }
        },
    )

    report = validate_mechanism(mechanism)
    warnings = [msg.code for msg in report.messages if msg.level == 'WARNING']
    errors = [msg.code for msg in report.messages if msg.level == 'ERROR']

    assert 'PROVENANCE_FORMAT_INVALID' in warnings
    assert errors == []
