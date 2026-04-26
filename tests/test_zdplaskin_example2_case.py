from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from plasma_global.workflows.runner import run_from_yaml
from plasma_global.workflows.context import build_case, load_case_from_yaml


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


def test_zdplaskin_example2_reaction_budget_reports_key_species() -> None:
    loaded = load_case_from_yaml(ZDP_CASE)
    built = build_case(loaded)
    system = built.system
    y0 = system.initial_state()
    budget = system.reaction_budget(
        0.0,
        y0,
        species_filter=['Ar_star', 'Ar_plus', 'Ar2_plus'],
        max_reactions_per_species=4,
    )

    plasma = budget['zones']['plasma']['species']
    assert plasma['Ar_star']['production_m3_s'] > 0.0
    assert plasma['Ar_plus']['production_m3_s'] > 0.0
    assert plasma['Ar2_plus']['production_m3_s'] > 0.0
    assert any(term['process_id'] == 'ZDP_AR_EXC' for term in plasma['Ar_star']['top_production'])
    assert any(term['process_id'] == 'ZDP_ARPLUS_CLUSTERING' for term in plasma['Ar2_plus']['top_production'])


def test_zdplaskin_example2_energy_surface_and_state_diagnostics_are_available() -> None:
    loaded = load_case_from_yaml(ZDP_CASE)
    built = build_case(loaded)
    system = built.system
    y0 = system.initial_state()

    energy = system.electron_energy_budget(0.0, y0)
    plasma_energy = energy['zones']['plasma']
    assert plasma_energy['terms_W_m3']['absorbed_power_W_m3'] > 0.0
    assert plasma_energy['terms_W_m3']['electron_impact_loss_W_m3'] < 0.0
    assert 'field_table_energy_relaxation_override_W_m3' in plasma_energy['terms_W_m3']
    assert plasma_energy['top_losses']

    surface = system.surface_reaction_budget(0.0, y0)
    assert 'wall' in surface['surfaces']
    assert surface['surfaces']['wall']['area_m2'] > 0.0

    manifest = system.state_manifest()
    labels = {item['label'] for item in manifest['states']}
    assert 'n[plasma,Ar_star]' in labels
    assert 'We[plasma]' in labels
    assert manifest['electron_density']['state_status'] == 'algebraic_not_state_variable'
    catalog = {item['species']: item for item in manifest['species_catalog']}
    assert catalog['e']['state_status'] == 'algebraic_density_from_quasi_neutrality'
    assert catalog['Ar2_plus']['state_status'] == 'solved_gas_density'


def test_zdplaskin_example2_run_writes_diagnostic_yaml_files() -> None:
    result = run_from_yaml(ZDP_CASE)
    out = Path(result['output_dir'])

    assert (out / 'reaction_budget.yaml').exists()
    assert (out / 'electron_energy_budget.yaml').exists()
    assert (out / 'surface_reaction_budget.yaml').exists()
    assert (out / 'state_manifest.yaml').exists()
    assert (out / 'run_provenance.yaml').exists()


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
    manifest = system.state_manifest()
    assert manifest['electron_density']['closure'] == 'prescribed_profile'
    catalog = {item['species']: item for item in manifest['species_catalog']}
    assert catalog['e']['state_status'] == 'prescribed_external_profile_not_state_variable'
