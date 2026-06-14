from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.validator import validate_run_config
from plasma_global.numerics.system import GlobalPlasmaSystem
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
        assert eedf.transport['lookup_mode'] == 'mean_energy'
        assert eedf.transport['mean_energy_eV'] == coupled.mean_e_by_zone[zone_id]


def test_swarm_closure_validation_reports_unrecognized_mode() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.swarm.closure = 'fieldish'
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))

    assert any(msg.code == 'SWARM_CLOSURE_UNRECOGNIZED' for msg in report.messages)


def test_local_field_closure_validation_warns_about_reduced_model() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.swarm.closure = 'local_field'
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))

    assert any(msg.code == 'LOCAL_FIELD_CLOSURE_REDUCED_MODEL' for msg in report.messages)


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
        ne = system.gas_core.electron_density_from_state_row(gas)
        y[system.state_layout.electron_energy_index[zone_id]] = mean_e * ne * E_CHARGE

    rec = system.compute_observables(np.array([0.0]), y.reshape(-1, 1))[0]

    assert np.isclose(rec['mean_electron_energy_eV'], 23.25)


def test_ion_loss_area_uses_active_surface_model_flags() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system

    assert np.isclose(system.zone_ion_loss_area['source'], 0.18)
    assert np.isclose(system.zone_ion_loss_area['process'], 0.0314 + 0.45)
    assert system.zone_ion_loss_family['source'] == 'bohm'
    assert system.zone_ion_loss_family['process'] == 'bohm'
    assert np.isclose(system.zone_ion_loss_h_factor['source'], 1.0)


def test_ion_loss_model_flag_can_disable_surface() -> None:
    assert ion_loss_enabled({'ion_loss': 'Bohm_like'}) is True
    assert ion_loss_family({'ion_loss': 'bohm_edge_loss'}) == 'bohm'
    assert ion_loss_family({'ion_loss': 'bohm_global_loss'}) == 'bohm'
    assert ion_loss_family({'ion_loss': 'prescribed_loss_frequency'}) == 'prescribed_loss_frequency'
    assert ion_loss_family({'ion_loss': 'loss_frequency'}) == 'prescribed_loss_frequency'
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
        {'ion_loss': 'prescribed_loss_frequency', 'loss_rate_s': 3230.0},
        volume_m3=1.0e-3,
        area_m2=1.0e-2,
    )
    legacy_ambipolar = effective_ion_loss_frequency_s(
        {'ion_loss': 'ambipolar_diffusion', 'loss_rate_s': 123.0},
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
    assert legacy_ambipolar == pytest.approx(123.0)
    assert diffusion_rate == pytest.approx(1.0)


def test_bohm_h_factor_can_be_auto_or_explicit() -> None:
    explicit = bohm_h_factor(
        {'ion_loss': 'bohm_edge_loss', 'h_factor': 0.42},
        pressure_Pa=10.0,
        gas_temperature_K=300.0,
        characteristic_length_m=0.1,
    )
    auto = bohm_h_factor(
        {'ion_loss': 'bohm_global_loss'},
        pressure_Pa=10.0,
        gas_temperature_K=300.0,
        characteristic_length_m=0.1,
    )

    assert np.isclose(explicit, 0.42)
    assert 0.02 <= auto <= 1.0


def test_observables_report_ion_loss_diagnostics() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    rec = system.compute_observables(np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['ion_loss_family_source'] == 'bohm'
    assert rec['ion_loss_area_source_m2'] == system.zone_ion_loss_area['source']
    assert rec['ion_loss_h_factor_source'] == system.zone_ion_loss_h_factor['source']
    assert rec['ion_loss_characteristic_length_source_m'] == system.zone_ion_loss_characteristic_length_m['source']
    assert rec['ambipolar_loss_rate_source_s'] == 0.0
    for zone_id in system.zone_ids:
        assert np.isfinite(rec[f'ion_wall_loss_frequency_{zone_id}_s'])
        assert np.isfinite(rec[f'ion_wall_loss_source_{zone_id}_m3_s'])
        assert np.isfinite(rec[f'ion_wall_flux_{zone_id}_m2_s'])
        assert rec[f'ion_wall_loss_frequency_{zone_id}_s'] >= 0.0
        assert rec[f'ion_wall_loss_source_{zone_id}_m3_s'] >= 0.0
        assert rec[f'ion_wall_flux_{zone_id}_m2_s'] >= 0.0


def test_prescribed_ion_loss_frequency_validates_and_reports_observable(tmp_path: Path) -> None:
    chamber = _example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['loss_rate_s'] = 3230.0

    case_path = _write_case_with_chamber(tmp_path, chamber)
    run_config = load_run_config(case_path)
    report = validate_run_config(run_config, resolve_run_paths(run_config, case_path))
    assert not any(msg.level == 'ERROR' for msg in report.messages)

    built = build_case(load_case_from_yaml(case_path))
    system = built.system
    y0 = system.initial_state()
    rec = system.compute_observables(np.array([0.0]), y0.reshape(-1, 1))[0]

    assert system.zone_ion_loss_family['source'] == 'prescribed_loss_frequency'
    assert system.zone_effective_ion_loss_frequency_s['source'] == pytest.approx(3230.0)
    assert rec['ion_wall_loss_frequency_source_s'] == pytest.approx(3230.0)


def test_surface_ion_flux_uses_zone_wall_loss_when_no_ied(tmp_path: Path) -> None:
    chamber = _example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['loss_rate_s'] = 3230.0

    case_path = _write_case_with_chamber(tmp_path, chamber)
    built = build_case(load_case_from_yaml(case_path))
    system = built.system
    y0 = system.initial_state()
    rec = system.compute_observables(np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['ion_flux_source_wall_m2_s'] == pytest.approx(rec['ion_wall_flux_source_m2_s'])
    assert rec['ion_flux_source_wall_m2_s'] >= 0.0


def test_surface_ion_flux_prefers_explicit_ied() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    step = system.current_step(0.0)
    coupled = system.electrical_adapter.evaluate(0.0, y0, step)
    coupled.power.metadata.setdefault('surface_ied', {})['source_wall'] = {'ion_flux_m2_s': 123.0}
    gas_row = system.gas_core.gas_row(y0, 'source')

    assert system.surface_core.surface_ion_flux_m2_s('source_wall', 'source', gas_row, coupled) == pytest.approx(123.0)


def test_prescribed_ion_loss_frequency_requires_rate_or_diffusion_data(tmp_path: Path) -> None:
    chamber = _example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models'].pop('loss_rate_s', None)
            surface['models'].pop('ambipolar_loss_rate_s', None)
            surface['models'].pop('diffusion_coefficient_m2_s', None)
            surface['models'].pop('ambipolar_diffusion_coefficient_m2_s', None)

    case_path = _write_case_with_chamber(tmp_path, chamber)
    with pytest.raises(ValueError, match='ION_LOSS_FREQUENCY_CONFIG_INVALID'):
        load_case_from_yaml(case_path)


def test_observables_flatten_numeric_power_port_diagnostics() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    rec = system.compute_observables(np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['port_source_rf_absorbed_power_W'] > 0.0
    assert rec['port_source_rf_frequency_Hz'] == pytest.approx(13.56e6)
    assert rec['port_wafer_bias_absorbed_power_W'] >= 0.0
    assert rec['port_wafer_bias_self_bias_V'] <= 0.0


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
                'models': {'ion_loss': 'bohm_edge_loss'},
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
                'models': {'ion_loss': 'ambipolar_diffusion', 'ambipolar_loss_rate_s': 1000.0},
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
