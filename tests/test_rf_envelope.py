from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from plasma_global.electrical.base import PowerRequest
from plasma_global.electrical.rf_envelope import RFEnvelopeBackend, validate_rf_envelope_port
from plasma_global.workflows.context import ELECTRICAL_REGISTRY, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def _rf_chamber() -> SimpleNamespace:
    zone = SimpleNamespace(zone_id='plasma', pressure_Pa=100.0, gas_temperature_K=300.0)
    source = SimpleNamespace(
        port_id='hf_source',
        kind='hf_source',
        zone_id='plasma',
        coupling_target='plasma',
        parameters={},
    )
    bias = SimpleNamespace(
        port_id='lf_bias',
        kind='lf_bias',
        zone_id='plasma',
        coupling_target='electrode',
        parameters={},
    )
    return SimpleNamespace(
        zones=[zone],
        zone_by_id={'plasma': zone},
        power_port_by_id={'hf_source': source, 'lf_bias': bias},
    )


def test_rf_envelope_combines_hf_power_and_lf_bias() -> None:
    chamber = _rf_chamber()
    step = SimpleNamespace(
        power_ports={
            'hf_source': {
                'role': 'hf_source',
                'zone_id': 'plasma',
                'frequency_Hz': 13.56e6,
                'value_W': 200.0,
                'coupling_efficiency': 0.5,
                'base_reduced_field_Td': 20.0,
                'reduced_field_per_sqrt_W_Td': 2.0,
            },
            'lf_bias': {
                'role': 'lf_bias',
                'zone_id': 'plasma',
                'frequency_Hz': 400.0e3,
                'voltage_rms_V': 100.0,
                'effective_impedance_ohm': 50.0,
                'coupling_efficiency': 0.2,
                'self_bias_fraction': 0.5,
                'plasma_potential_from_bias_fraction': 0.1,
            },
        }
    )
    backend = RFEnvelopeBackend()
    backend.prepare(
        chamber=chamber,
        recipe=SimpleNamespace(steps=[step]),
        run_config=SimpleNamespace(),
        resolved_paths=SimpleNamespace(),
    )

    result = backend.evaluate(PowerRequest(time_s=0.0, state_vector=None, recipe_step=step, chamber=chamber))

    assert result.port_power_W['hf_source'] == pytest.approx(100.0)
    assert result.port_power_W['lf_bias'] == pytest.approx(40.0)
    assert result.absorbed_power_W_by_zone['plasma'] == pytest.approx(140.0)
    assert result.self_bias_V < 0.0
    assert result.zone_reduced_field_Td['plasma'] == pytest.approx(40.0)


def test_rf_envelope_pulsed_square_turns_power_off() -> None:
    chamber = _rf_chamber()
    step = SimpleNamespace(
        power_ports={
            'hf_source': {
                'frequency_Hz': 13.56e6,
                'value_W': 100.0,
                'waveform': 'pulsed_square',
                'duty_cycle': 0.25,
                'repetition_Hz': 1000.0,
            }
        }
    )
    backend = RFEnvelopeBackend()
    backend.prepare(
        chamber=chamber,
        recipe=SimpleNamespace(steps=[step]),
        run_config=SimpleNamespace(),
        resolved_paths=SimpleNamespace(),
    )

    result = backend.evaluate(PowerRequest(time_s=5.0e-4, state_vector=None, recipe_step=step, chamber=chamber))

    assert result.port_power_W['hf_source'] == 0.0


def test_rf_envelope_backend_is_registered() -> None:
    details = ELECTRICAL_REGISTRY.details()
    assert 'HF/LF' in details['rf_envelope']['description']


def test_rf_envelope_calibration_example_validates_without_hints() -> None:
    loaded = load_case_from_yaml(ROOT / 'examples' / 'configs' / 'case_rf_envelope_calibration.yaml')
    assert loaded.run_config.physics.electrical_backend == 'rf_envelope'
    assert not any(msg['level'] == 'ERROR' for msg in loaded.validation_messages)


def test_rf_envelope_validation_rejects_invalid_calibration_values() -> None:
    with pytest.raises(ValueError, match='coupling_efficiency'):
        validate_rf_envelope_port(
            {
                'role': 'hf_source',
                'frequency_Hz': 13.56e6,
                'value_W': 100.0,
                'coupling_efficiency': 1.2,
            }
        )


def test_rf_envelope_validation_reports_missing_impedance_for_voltage_drive(tmp_path: Path) -> None:
    chamber_path = tmp_path / 'chamber.yaml'
    recipe_path = tmp_path / 'recipe.yaml'
    case_path = tmp_path / 'case.yaml'
    chamber_path.write_text(
        yaml.safe_dump(
            {
                'chamber_id': 'rf_envelope_validation_test',
                'zones': [
                    {
                        'zone_id': 'plasma',
                        'description': '',
                        'volume_m3': 1.0e-3,
                        'pressure_Pa': 100.0,
                        'gas_temperature_K': 300.0,
                    }
                ],
                'edges': [],
                'surfaces': [],
                'gas_inlets': [],
                'pumps': [],
                'power_ports': [
                    {
                        'port_id': 'lf_bias',
                        'kind': 'lf_bias',
                        'zone_id': 'plasma',
                        'coupling_target': 'electrode',
                        'parameters': {},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    recipe_path.write_text(
        yaml.safe_dump(
            {
                'recipe_id': 'rf_envelope_validation_test',
                'steps': [
                    {
                        'step_id': 'main',
                        't_start_s': 0.0,
                        't_end_s': 1.0e-6,
                        'power_ports': {
                            'lf_bias': {
                                'frequency_Hz': 400.0e3,
                                'voltage_rms_V': 100.0,
                            }
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    case_path.write_text(
        yaml.safe_dump(
            {
                'case': {'name': 'rf_envelope_validation_test'},
                'files': {
                    'chamber': str(chamber_path),
                    'recipe': str(recipe_path),
                    'chemistry': {
                        'manifest': str(ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml')
                    },
                    'output_dir': str(tmp_path / 'out'),
                },
                'physics': {'electrical_backend': 'rf_envelope'},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='RF_ENVELOPE_CONFIG_INVALID'):
        load_case_from_yaml(case_path)
