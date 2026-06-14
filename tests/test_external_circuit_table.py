from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from plasma_global.electrical.base import PowerRequest
from plasma_global.electrical.external_table import ExternalCircuitTableBackend, read_circuit_table
from plasma_global.workflows.context import ELECTRICAL_REGISTRY, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def _single_zone_chamber() -> SimpleNamespace:
    zone = SimpleNamespace(zone_id='plasma', pressure_Pa=100.0, gas_temperature_K=300.0)
    port = SimpleNamespace(
        port_id='circuit_waveform',
        kind='external_circuit_table',
        zone_id='plasma',
        coupling_target='plasma',
        parameters={},
    )
    return SimpleNamespace(
        zones=[zone],
        zone_by_id={'plasma': zone},
        power_port_by_id={'circuit_waveform': port},
    )


def test_external_circuit_table_backend_interpolates_power_and_field(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text(
        'time_s,absorbed_power_W,voltage_V,current_A,reduced_field_Td\n'
        '0.0,10.0,100.0,0.1,40.0\n'
        '1e-6,20.0,120.0,0.2,60.0\n',
        encoding='utf-8',
    )
    chamber = _single_zone_chamber()
    step = SimpleNamespace(
        power_ports={
            'circuit_waveform': {
                'zone_id': 'plasma',
                'file': str(csv_path),
            }
        }
    )
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    result = backend.evaluate(
        PowerRequest(
            time_s=5.0e-7,
            state_vector=None,
            recipe_step=step,
            chamber=chamber,
            metadata={},
        )
    )

    detail = result.metadata['port_details']['circuit_waveform']
    assert result.port_power_W['circuit_waveform'] == pytest.approx(15.0)
    assert result.absorbed_power_W_by_zone['plasma'] == pytest.approx(15.0)
    assert result.metadata['zone_reduced_field_Td']['plasma'] == pytest.approx(50.0)
    assert detail['backend'] == 'external_circuit_table'
    assert detail['voltage_V'] == pytest.approx(110.0)
    assert detail['current_A'] == pytest.approx(0.15)


def test_external_circuit_table_can_use_voltage_current_product(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,voltage_V,current_A\n0.0,100.0,0.1\n1e-6,200.0,0.2\n', encoding='utf-8')
    table = read_circuit_table(csv_path)
    assert {'voltage_V', 'current_A'} <= table.names

    chamber = _single_zone_chamber()
    step = SimpleNamespace(power_ports={'circuit_waveform': {'file': str(csv_path)}})
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    result = backend.evaluate(
        PowerRequest(time_s=1.0e-6, state_vector=None, recipe_step=step, chamber=chamber, metadata={})
    )

    assert result.port_power_W['circuit_waveform'] == pytest.approx(40.0)


def test_external_circuit_table_backend_is_registered() -> None:
    details = ELECTRICAL_REGISTRY.details()
    assert 'CSV' in details['external_circuit_table']['description']


def test_external_circuit_table_validation_reports_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / 'bad_circuit.csv'
    csv_path.write_text('time_s,source_voltage_V\n0.0,100.0\n', encoding='utf-8')
    chamber_path = tmp_path / 'chamber.yaml'
    recipe_path = tmp_path / 'recipe.yaml'
    case_path = tmp_path / 'case.yaml'
    chamber_path.write_text(
        yaml.safe_dump(
            {
                'chamber_id': 'external_table_validation_test',
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
                        'port_id': 'circuit_waveform',
                        'kind': 'external_circuit_table',
                        'zone_id': 'plasma',
                        'coupling_target': 'plasma',
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
                'recipe_id': 'external_table_validation_test',
                'steps': [
                    {
                        'step_id': 'main',
                        't_start_s': 0.0,
                        't_end_s': 1.0e-6,
                        'power_ports': {'circuit_waveform': {'file_key': 'circuit_result_csv'}},
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
                'case': {'name': 'external_table_validation_test'},
                'files': {
                    'chamber': str(chamber_path),
                    'recipe': str(recipe_path),
                    'chemistry': {
                        'manifest': str(ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml')
                    },
                    'output_dir': str(tmp_path / 'out'),
                    'external_inputs': {'circuit_result_csv': str(csv_path)},
                },
                'physics': {'electrical_backend': 'external_circuit_table'},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='EXTERNAL_CIRCUIT_TABLE_CONFIG_INVALID'):
        load_case_from_yaml(case_path)
