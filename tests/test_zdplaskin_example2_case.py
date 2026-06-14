from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from plasma_global.workflows.context import build_case, load_case_from_yaml
from plasma_global.workflows.runner import run_from_yaml


ROOT = Path(__file__).resolve().parents[1]
ZDP_CASE = ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'


def test_zdplaskin_example2_surrogate_loads_extended_argon_mechanism() -> None:
    loaded = load_case_from_yaml(ZDP_CASE)

    assert not any(msg['level'] == 'ERROR' for msg in loaded.validation_messages)
    assert loaded.run_config.case.name == 'zdplaskin_example2_dc_series'
    assert loaded.run_config.physics.eedf_backend == 'rate_table'
    assert loaded.run_config.physics.electrical_backend == 'dc_series_circuit'
    assert loaded.run_config.swarm.closure == 'local_field'
    assert loaded.run_config.swarm.table.file == 'tables/zdplaskin_example2_eovern_rates.h5'
    assert loaded.run_config.physics.enable_gas_temperature is False
    assert loaded.recipe.steps[0].power_ports['dc_series_drive']['mode'] == 'voltage'
    assert loaded.chamber.power_port_by_id['dc_series_drive'].parameters['source_voltage_V'] == 1000.0
    assert loaded.chamber.power_port_by_id['dc_series_drive'].parameters['ballast_resistance_ohm'] == 1.0e5
    assert loaded.resolved_paths.chemistry_manifest.endswith('chemistry_manifest.yaml')

    species = {sp.canonical_id for sp in loaded.mechanism.gas_species}
    assert {'e', 'Ar', 'Ar_star', 'Ar_plus', 'Ar2_plus'} <= species
    assert [sp.canonical_id for sp in loaded.mechanism.gas_state_species] == [
        'Ar',
        'Ar_star',
        'Ar_plus',
        'Ar2_plus',
    ]

    cross_sections = loaded.mechanism.cross_sections
    assert cross_sections['xs_zdp_ar_effective'].metadata['mass_ratio'] == 1.36e-5
    assert cross_sections['xs_zdp_ar_star_ionization'].target_species == 'Ar_star'
    assert 'not a real cross section' in cross_sections['xs_zdp_ar_star_ionization'].metadata['source_warning']

    models = loaded.mechanism.rate_models
    assert models['RM_E_AR_ION_ZDP']['backend'] == 'electron_impact_xsec'
    assert models['RM_E_ARSTAR_DEEXC_ZDP']['cross_section_id'] == 'xs_zdp_ar_star_deexcitation'
    assert models['RM_AR2PLUS_DISS_RECOMB_ZDP']['backend'] == 'te_power_law'
    assert models['RM_AR2PLUS_DISS_RECOMB_ZDP']['Tref_K'] == 300.0
    assert models['EM_E_AR_EXC_ZDP']['backend'] == 'constant_event_loss'

    reaction_ids = {rxn.reaction_id for rxn in loaded.mechanism.gas_reactions}
    assert 'ZDP_ARSTAR_POOLING' in reaction_ids
    assert 'ZDP_ARPLUS_CLUSTERING' in reaction_ids


def test_zdplaskin_example2_run_writes_core_outputs() -> None:
    result = run_from_yaml(ZDP_CASE)
    out = Path(result['output_dir'])

    assert (out / 'summary.yaml').exists()
    assert (out / 'observables.csv').exists()
    assert (out / 'effective_case.yaml').exists()
    assert (out / 'resolved_paths.yaml').exists()


def test_prescribed_electron_profile_overrides_quasineutral_density(tmp_path) -> None:
    profile = tmp_path / 'electron_profile.csv'
    profile.write_text('time_s,electron_density_m3\n0.0,5.0e16\n1.0e-3,6.0e16\n', encoding='utf-8')
    loaded = load_case_from_yaml(ZDP_CASE)
    loaded.run_config.physics.electron_density_closure = 'prescribed_profile'
    loaded.run_config.swarm.prescribed_electron_profile = SimpleNamespace(
        file=str(profile),
        density_column='electron_density_m3',
    )
    built = build_case(loaded)
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.recipe.steps[0])

    assert coupled.ne_by_zone['plasma'] == 5.0e16
    assert system.electron_density_closure == 'prescribed_profile'
