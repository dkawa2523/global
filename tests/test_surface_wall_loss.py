from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.validator import validate_run_config
from plasma_global.observables.adapter import compute_observables
from plasma_global.reactor.surface_models import (
    bohm_h_factor,
    bohm_ion_loss_frequency_s,
    effective_ion_loss_frequency_s,
    ion_loss_enabled,
    ion_loss_family,
)
from plasma_global import build_case, load_case_from_yaml
from tests.case_helpers import ROOT, SMOKE_CASE, example_chamber, write_smoke_case_with_chamber


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
        ion_loss_family({'ion_loss': 'unknown_bohm'})
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


def test_prescribed_ion_loss_frequency_validates_and_reports_observable(tmp_path: Path) -> None:
    chamber = example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['frequency_s'] = 3230.0

    case_path = write_smoke_case_with_chamber(tmp_path, chamber)
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


def test_prescribed_ion_loss_frequency_requires_rate_or_diffusion_data(tmp_path: Path) -> None:
    chamber = example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models'].pop('frequency_s', None)
            surface['models'].pop('diffusion_coefficient_m2_s', None)

    case_path = write_smoke_case_with_chamber(tmp_path, chamber)
    with pytest.raises(ValueError, match='ION_LOSS_FREQUENCY_CONFIG_INVALID'):
        load_case_from_yaml(case_path)


def test_multiple_effective_frequency_surfaces_are_warnings(tmp_path: Path) -> None:
    chamber = example_chamber()
    for surface in chamber['surfaces']:
        if surface['zone_id'] == 'process':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['frequency_s'] = 100.0

    case_path = write_smoke_case_with_chamber(tmp_path, chamber)
    loaded = load_case_from_yaml(case_path)

    assert any(
        msg['level'] == 'WARNING' and msg['code'] == 'ION_LOSS_FREQUENCY_MULTIPLE_SURFACES'
        for msg in loaded.validation_messages
    )


def test_case_validation_rejects_unknown_ion_loss_mode(tmp_path: Path) -> None:
    chamber = example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'unknown_bohm'

    case_path = write_smoke_case_with_chamber(tmp_path, chamber)
    with pytest.raises(ValueError, match='ION_LOSS_MODEL_UNRECOGNIZED'):
        load_case_from_yaml(case_path)


def test_case_validation_rejects_invalid_bohm_h_factor(tmp_path: Path) -> None:
    chamber = example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'bohm'
            surface['models']['h_factor'] = 'invalid'

    case_path = write_smoke_case_with_chamber(tmp_path, chamber)
    with pytest.raises(ValueError, match='BOHM_H_FACTOR_INVALID'):
        load_case_from_yaml(case_path)


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
