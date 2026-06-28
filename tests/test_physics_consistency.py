from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.config.models import (
    Boltzmann2TermConfig,
    SwarmCacheConfig,
    SwarmConfig,
    SwarmEnergyGridConfig,
    SwarmReducedFieldGridConfig,
)
from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.validator import validate_run_config
from plasma_global.diagnostics.budgets import reaction_source_loss_budget
from plasma_global.eedf.base import EEDFRequest
from plasma_global.eedf.boltzmann_2term import Boltzmann2TermSwarmModel
from plasma_global.electrical.base import SurfaceIED
from plasma_global.numerics.system import GlobalPlasmaSystem
from plasma_global.observables.adapter import compute_observables
from plasma_global.physics.gas_closure import electron_density_from_state_row
from plasma_global.physics.types import SurfaceRateEvaluation
from plasma_global.reactor.surface_models import (
    bohm_h_factor,
    bohm_ion_loss_frequency_s,
    effective_ion_loss_frequency_s,
    ion_loss_enabled,
    ion_loss_family,
)
from plasma_global.workflows.context import build_case, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CASE = ROOT / 'examples' / 'configs' / 'case_smoke.yaml'
BASE_CASE = ROOT / 'examples' / 'configs' / 'base_case.yaml'


def _write_case_with_chamber(tmp_path: Path, chamber: dict) -> Path:
    chamber_path = tmp_path / 'chamber.yaml'
    case_path = tmp_path / 'case.yaml'
    chamber_path.write_text(yaml.safe_dump(chamber, sort_keys=False), encoding='utf-8')
    case_path.write_text(
        yaml.safe_dump(
            {
                'include': str(BASE_CASE),
                'case': {'name': 'wall_loss_test'},
                'files': {
                    'chamber': str(chamber_path),
                    'recipe': str(ROOT / 'examples' / 'configs' / 'recipe_smoke.yaml'),
                    'chemistry': {'manifest': str(ROOT / 'examples' / 'chemistry' / 'chemistry_manifest.yaml')},
                    'output_dir': str(tmp_path / 'out'),
                },
                'outputs': {'plots': {'enabled': False}},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    return case_path


def _example_chamber() -> dict:
    return yaml.safe_load((ROOT / 'examples' / 'configs' / 'chamber.yaml').read_text(encoding='utf-8'))


def test_auto_swarm_closure_uses_electron_energy_state() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))

    for zone_id, eedf in coupled.eedf_by_zone.items():
        assert eedf.transport.lookup_mode == 'mean_energy'
        assert eedf.transport.mean_energy_eV == coupled.mean_e_by_zone[zone_id]


def test_swarm_closure_validation_reports_unrecognized_mode() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.swarm.closure = 'fieldish'
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))

    assert any(msg.code == 'SWARM_CLOSURE_UNRECOGNIZED' for msg in report.messages)


def test_local_field_closure_validation_is_error_free() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.swarm.closure = 'local_field'
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))

    assert not any(msg.level == 'ERROR' for msg in report.messages)


def test_recipe_boundary_selects_next_step() -> None:
    fake_system = SimpleNamespace(
        recipe=SimpleNamespace(
            steps=[
                SimpleNamespace(step_id='first', t_start_s=0.0, t_end_s=1.0),
                SimpleNamespace(step_id='second', t_start_s=1.0, t_end_s=2.0),
            ]
        )
    )

    assert GlobalPlasmaSystem.current_step(fake_system, 1.0).step_id == 'second'


def test_global_electron_energy_observable_is_volume_weighted() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y = system.initial_state()

    for zone_id, mean_e in {'source': 3.0, 'process': 30.0}.items():
        gas = system.gas_core.gas_row(y, zone_id)
        ne = electron_density_from_state_row(system, gas)
        y[system.state_layout.electron_energy_index[zone_id]] = mean_e * ne * E_CHARGE

    rec = compute_observables(system, np.array([0.0]), y.reshape(-1, 1))[0]

    assert np.isclose(rec['mean_electron_energy_eV'], 23.25)


def test_ion_loss_area_uses_active_surface_model_flags() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system

    assert np.isclose(system.zone_ion_loss_area['source'], 0.18)
    assert np.isclose(system.zone_ion_loss_area['process'], 0.0314 + 0.45)
    assert system.zone_ion_loss_family['source'] == 'bohm'
    assert system.zone_ion_loss_family['process'] == 'bohm'
    assert np.isclose(system.zone_ion_loss_h_factor['source'], 1.0)


def test_zone_wall_temperature_is_area_weighted() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system

    assert system.zone_wall_temperature['source'] == pytest.approx(360.0)
    assert system.zone_wall_temperature['process'] == pytest.approx(329.34773577066886)


def test_ion_loss_model_flag_can_disable_surface() -> None:
    assert ion_loss_enabled({'ion_loss': 'bohm'}) is True
    with pytest.raises(ValueError, match='Unknown ion_loss mode'):
        ion_loss_family({'ion_loss': 'legacy_bohm'})
    assert ion_loss_family({'ion_loss': 'prescribed_loss_frequency'}) == 'prescribed_loss_frequency'
    assert ion_loss_family({'ion_loss': 'ambipolar_diffusion'}) == 'ambipolar_diffusion'
    assert ion_loss_enabled({'ion_loss': 'off'}) is False
    assert ion_loss_enabled({}) is True


def test_ion_loss_helper_frequencies_are_finite_and_compatible() -> None:
    bohm_frequency = bohm_ion_loss_frequency_s(
        area_m2=0.1,
        volume_m3=1.0e-3,
        h_factor=0.5,
        mean_energy_eV=3.0,
        ion_mass_kg=6.63e-26,
    )
    prescribed = effective_ion_loss_frequency_s(
        {'ion_loss': 'prescribed_loss_frequency', 'frequency_s': 3230.0},
        volume_m3=1.0e-3,
        area_m2=1.0e-2,
    )
    diffusion_rate = effective_ion_loss_frequency_s(
        {'ion_loss': 'ambipolar_diffusion', 'diffusion_coefficient_m2_s': 0.01, 'diffusion_length_m': 0.1},
        volume_m3=1.0e-3,
        area_m2=1.0e-2,
    )

    assert np.isfinite(bohm_frequency)
    assert bohm_frequency > 0.0
    assert prescribed == pytest.approx(3230.0)
    assert diffusion_rate == pytest.approx(1.0)


def test_bohm_h_factor_can_be_auto_or_explicit() -> None:
    explicit = bohm_h_factor(
        {'ion_loss': 'bohm', 'h_factor': 0.42},
        pressure_Pa=10.0,
        gas_temperature_K=300.0,
        characteristic_length_m=0.1,
    )
    auto = bohm_h_factor(
        {'ion_loss': 'bohm', 'h_factor': 'auto'},
        pressure_Pa=10.0,
        gas_temperature_K=300.0,
        characteristic_length_m=0.1,
    )

    assert np.isclose(explicit, 0.42)
    assert 0.02 <= auto <= 1.0


def test_observables_keep_compact_zone_and_surface_columns() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]

    for zone_id in system.zone_ids:
        assert np.isfinite(rec[f'ne_{zone_id}_m3'])
        assert np.isfinite(rec[f'mean_energy_{zone_id}_eV'])
        assert np.isfinite(rec[f'pabs_{zone_id}_W'])
        assert np.isfinite(rec[f'EoverN_{zone_id}_Td'])
    surface_id = system.surface_ids[0]
    assert rec[f'ion_flux_{surface_id}_m2_s'] >= 0.0
    assert not any(key.startswith(('ion_loss_', 'ion_wall_', 'ambipolar_loss_')) for key in rec)
    assert not any(key.startswith(('reaction_rate_', 'species_source_', 'species_loss_')) for key in rec)


def test_prescribed_ion_loss_frequency_validates_and_reports_observable(tmp_path: Path) -> None:
    chamber = _example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['frequency_s'] = 3230.0

    case_path = _write_case_with_chamber(tmp_path, chamber)
    run_config = load_run_config(case_path)
    report = validate_run_config(run_config, resolve_run_paths(run_config, case_path))
    assert not any(msg.level == 'ERROR' for msg in report.messages)

    built = build_case(load_case_from_yaml(case_path))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]

    assert system.zone_ion_loss_family['source'] == 'prescribed_loss_frequency'
    assert system.zone_effective_ion_loss_frequency_s['source'] == pytest.approx(3230.0)
    assert rec['ion_flux_source_wall_m2_s'] >= 0.0
    assert 'ion_wall_loss_frequency_source_s' not in rec


def test_surface_ion_flux_uses_zone_wall_loss_when_no_ied(tmp_path: Path) -> None:
    chamber = _example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['frequency_s'] = 3230.0

    case_path = _write_case_with_chamber(tmp_path, chamber)
    built = build_case(load_case_from_yaml(case_path))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))
    gas_row = system.gas_core.gas_row(y0, 'source')
    expected = system.gas_core.ion_wall_loss_flux_m2_s('source', gas_row, coupled.mean_e_by_zone['source'])

    assert rec['ion_flux_source_wall_m2_s'] == pytest.approx(expected)
    assert rec['ion_flux_source_wall_m2_s'] >= 0.0


def test_surface_ion_flux_prefers_explicit_ied() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    step = system.current_step(0.0)
    coupled = system.electrical_adapter.evaluate(0.0, y0, step)
    coupled.power.surface_ied['source_wall'] = SurfaceIED(ion_flux_m2_s=123.0, mean_ion_energy_eV=10.0)
    gas_row = system.gas_core.gas_row(y0, 'source')

    assert system.surface_core.surface_ion_flux_m2_s('source_wall', 'source', gas_row, coupled) == pytest.approx(123.0)


def test_surface_ied_contract_is_compact() -> None:
    assert set(SurfaceIED.__dataclass_fields__) == {'ion_flux_m2_s', 'mean_ion_energy_eV'}


def test_surface_rate_evaluation_has_no_diagnostics_bus() -> None:
    assert set(SurfaceRateEvaluation.__dataclass_fields__) == {'rate_m2_s'}


def test_prescribed_ion_loss_frequency_requires_rate_or_diffusion_data(tmp_path: Path) -> None:
    chamber = _example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models'].pop('frequency_s', None)
            surface['models'].pop('diffusion_coefficient_m2_s', None)

    case_path = _write_case_with_chamber(tmp_path, chamber)
    with pytest.raises(ValueError, match='ION_LOSS_FREQUENCY_CONFIG_INVALID'):
        load_case_from_yaml(case_path)


def test_observables_do_not_flatten_power_port_diagnostics() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['total_absorbed_power_W'] > 0.0
    assert not any(key.startswith('port_') for key in rec)


def test_observables_include_model_budget_and_electrical_waveform_columns() -> None:
    built = build_case(load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['electrical_dc_series_drive_source_voltage_V'] == pytest.approx(1000.0)
    assert rec['electrical_dc_series_drive_current_A'] != 0.0
    assert rec['rate_table_lookup_clipped_plasma'] in {0, 1}
    assert system.run_config.outputs.diagnostics.budgets is True
    assert any(key.startswith('reaction_rate_plasma_') for key in rec)
    assert any(key.startswith('species_source_plasma_') for key in rec)
    assert any(key.startswith('species_loss_plasma_') for key in rec)


def test_gas_terms_and_budget_share_reaction_values() -> None:
    built = build_case(load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'))
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))

    terms = system.gas_core.gas_reaction_terms(coupled)
    budget = reaction_source_loss_budget(system.gas_core, coupled)
    expected_source: dict[tuple[str, str], float] = {}
    expected_loss: dict[tuple[str, str], float] = {}

    assert terms
    for term in terms:
        assert budget.reaction_rates[(term.zone_id, term.reaction_id)] == pytest.approx(term.rate_m3_s)
        for _idx, change, species_id in term.species_changes:
            target = expected_source if change >= 0.0 else expected_loss
            key = (term.zone_id, species_id)
            target[key] = target.get(key, 0.0) + abs(change)
    for term in system.gas_core.ion_wall_loss_terms(coupled):
        key = (term.zone_id, term.species_id)
        expected_loss[key] = expected_loss.get(key, 0.0) + term.loss_m3_s

    for key, value in expected_source.items():
        assert budget.species_source[key] == pytest.approx(value)
    for key, value in expected_loss.items():
        assert budget.species_loss[key] == pytest.approx(value)


def test_ion_wall_loss_terms_and_budget_share_loss_values() -> None:
    built = build_case(load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'))
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))

    terms = system.gas_core.ion_wall_loss_terms(coupled)
    budget = reaction_source_loss_budget(system.gas_core, coupled)
    expected_wall: dict[tuple[str, str], float] = {}

    assert terms
    for term in terms:
        key = (term.zone_id, term.species_id)
        expected_wall[key] = expected_wall.get(key, 0.0) + term.loss_m3_s

    for key, value in expected_wall.items():
        assert budget.wall_loss[key] == pytest.approx(value)


def test_case_validation_rejects_mixed_ion_loss_families(tmp_path: Path) -> None:
    chamber = {
        'chamber_id': 'mixed_ion_loss_test',
        'zones': [
            {
                'zone_id': 'plasma',
                'description': '',
                'volume_m3': 1.0e-3,
                'pressure_Pa': 10.0,
                'gas_temperature_K': 300.0,
                'role': 'process',
            }
        ],
        'edges': [],
        'surfaces': [
            {
                'surface_id': 'wall_bohm',
                'zone_id': 'plasma',
                'kind': 'wall',
                'area_m2': 1.0e-2,
                'material': 'test',
                'temperature_K': 300.0,
                'site_density_m2': 1.0e18,
                'initial_coverages': {},
                'models': {'ion_loss': 'bohm'},
            },
            {
                'surface_id': 'wall_diffusion',
                'zone_id': 'plasma',
                'kind': 'wall',
                'area_m2': 1.0e-2,
                'material': 'test',
                'temperature_K': 300.0,
                'site_density_m2': 1.0e18,
                'initial_coverages': {},
                'models': {'ion_loss': 'ambipolar_diffusion', 'diffusion_coefficient_m2_s': 0.01, 'diffusion_length_m': 0.1},
            },
        ],
        'gas_inlets': [
            {
                'inlet_id': 'initial_argon_seed',
                'zone_id': 'plasma',
                'flow_sccm': {'Ar': 1.0e-30},
                'temperature_K': 300.0,
            }
        ],
        'pumps': [],
        'power_ports': [
            {
                'port_id': 'circuit_surrogate_power',
                'kind': 'direct_power',
                'zone_id': 'plasma',
                'coupling_target': 'plasma',
                'parameters': {'control_mode': 'absorbed_power'},
            }
        ],
    }
    chamber_path = tmp_path / 'chamber_mixed.yaml'
    case_path = tmp_path / 'case.yaml'
    chamber_path.write_text(yaml.safe_dump(chamber, sort_keys=False), encoding='utf-8')
    case_path.write_text(
        yaml.safe_dump(
            {
                'case': {'name': 'mixed_ion_loss_test'},
                'files': {
                    'chamber': str(chamber_path),
                    'recipe': str(ROOT / 'examples' / 'configs' / 'recipe_zdplaskin_example2.yaml'),
                    'chemistry': {
                        'manifest': str(ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml')
                    },
                    'output_dir': str(tmp_path / 'out'),
                },
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='ION_LOSS_MODE_MIXED_IN_ZONE'):
        load_case_from_yaml(case_path)


def test_boltzmann_2term_requires_momentum_transfer_cross_section() -> None:
    class IonizationOnlyCrossSection:
        target_species = 'Ar'
        kind = 'ionization'
        threshold_eV = 15.76
        energy_loss_eV = 15.76

        def sigma_interp(self, energy_grid):
            return np.full_like(energy_grid, 1.0e-20, dtype=float)

    model = Boltzmann2TermSwarmModel()
    swarm = SwarmConfig(
        closure='mean_energy',
        mixture_key_species=['Ar'],
        cache=SwarmCacheConfig(max_entries=1, fraction_decimals=3),
        boltzmann_2term=Boltzmann2TermConfig(
            energy_grid=SwarmEnergyGridConfig(min_eV=0.1, max_eV=5.0, n=12),
            reduced_field_grid_Td=SwarmReducedFieldGridConfig(min=1.0, max=10.0, n=3),
            max_shape_iterations=2,
            max_field_iterations=2,
        ),
    )
    model.prepare(
        mechanism=SimpleNamespace(cross_sections={'xs_ar_ion': IonizationOnlyCrossSection()}, species_by_id={}),
        chamber=SimpleNamespace(),
        run_config=SimpleNamespace(),
        resolved_paths=SimpleNamespace(),
        swarm_config=swarm,
    )

    with pytest.raises(ValueError, match="momentum-transfer cross section.*'Ar'"):
        model.evaluate(
            EEDFRequest(
                time_s=0.0,
                zone_id='plasma',
                composition={'Ar': 1.0e20},
                electron_density_m3=1.0e16,
                mean_energy_eV=3.0,
                reduced_field_Td=5.0,
                gas_temperature_K=300.0,
                pressure_Pa=10.0,
            )
        )
