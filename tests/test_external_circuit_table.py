from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from plasma_global.electrical.base import PowerRequest
from plasma_global.electrical.external_table import ExternalCircuitTableBackend, read_circuit_table, validate_circuit_table_columns
from plasma_global.workflows.runner import run_from_yaml
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
    source = result.metadata['circuit_interface']['table_sources']['circuit_waveform']
    assert result.port_power_W['circuit_waveform'] == pytest.approx(15.0)
    assert result.absorbed_power_W_by_zone['plasma'] == pytest.approx(15.0)
    assert result.metadata['zone_reduced_field_Td']['plasma'] == pytest.approx(50.0)
    assert detail['backend'] == 'external_circuit_table'
    assert detail['voltage_V'] == pytest.approx(110.0)
    assert detail['current_A'] == pytest.approx(0.15)
    assert source['power_column'] == 'absorbed_power_W'
    assert source['voltage_column'] == 'voltage_V'
    assert source['current_column'] == 'current_A'
    assert source['reduced_field_column'] == 'reduced_field_Td'
    assert source['power_source'] == 'column'


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
    source = result.metadata['circuit_interface']['table_sources']['circuit_waveform']
    assert source['power_column'] is None
    assert source['power_source'] == 'voltage_current_product'


def test_external_circuit_table_derives_reduced_field_from_runtime_density(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,absorbed_power_W,voltage_V\n0.0,10.0,100.0\n', encoding='utf-8')
    chamber = _single_zone_chamber()
    step = SimpleNamespace(power_ports={'circuit_waveform': {'file': str(csv_path), 'gap_m': 0.1}})
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    result = backend.evaluate(
        PowerRequest(
            time_s=0.0,
            state_vector=None,
            recipe_step=step,
            chamber=chamber,
            metadata={'zone_total_density_m3': {'plasma': 2.0e20}},
        )
    )

    source = result.metadata['circuit_interface']['table_sources']['circuit_waveform']
    assert result.metadata['zone_reduced_field_Td']['plasma'] == pytest.approx(5.0e3)
    assert source['reduced_field_source'] == 'voltage_gap_runtime_density'


def test_external_circuit_table_config_density_takes_precedence_for_reduced_field(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,absorbed_power_W,voltage_V\n0.0,10.0,100.0\n', encoding='utf-8')
    chamber = _single_zone_chamber()
    step = SimpleNamespace(
        power_ports={
            'circuit_waveform': {
                'file': str(csv_path),
                'gap_m': 0.1,
                'total_density_m3': 1.0e20,
            }
        }
    )
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    result = backend.evaluate(
        PowerRequest(
            time_s=0.0,
            state_vector=None,
            recipe_step=step,
            chamber=chamber,
            metadata={'zone_total_density_m3': {'plasma': 2.0e20}},
        )
    )

    source = result.metadata['circuit_interface']['table_sources']['circuit_waveform']
    assert result.metadata['zone_reduced_field_Td']['plasma'] == pytest.approx(1.0e4)
    assert source['reduced_field_source'] == 'voltage_gap_config_density'


def test_external_circuit_table_requires_time_column(tmp_path: Path) -> None:
    csv_path = tmp_path / 'missing_time.csv'
    csv_path.write_text('voltage_V,current_A\n100.0,0.1\n', encoding='utf-8')

    with pytest.raises(ValueError, match='time_s column'):
        read_circuit_table(csv_path)


def test_external_circuit_table_validation_lists_available_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / 'bad_circuit.csv'
    csv_path.write_text('time_s,source_voltage_V\n0.0,100.0\n', encoding='utf-8')
    table = read_circuit_table(csv_path)

    with pytest.raises(ValueError, match='Available columns: .*source_voltage_V'):
        validate_circuit_table_columns(table, {})


def test_external_circuit_table_validation_reports_missing_override_column(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,absorbed_power_W\n0.0,10.0\n', encoding='utf-8')
    table = read_circuit_table(csv_path)

    with pytest.raises(ValueError, match="configured by voltage_column.*Available columns"):
        validate_circuit_table_columns(table, {'voltage_column': 'missing_voltage_V'})


def test_external_circuit_table_hold_error_rejects_out_of_range_time(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,absorbed_power_W\n0.0,10.0\n1e-6,20.0\n', encoding='utf-8')
    chamber = _single_zone_chamber()
    step = SimpleNamespace(power_ports={'circuit_waveform': {'file': str(csv_path), 'hold': 'error'}})
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    with pytest.raises(ValueError, match='outside external circuit table range'):
        backend.evaluate(PowerRequest(time_s=2.0e-6, state_vector=None, recipe_step=step, chamber=chamber, metadata={}))


def test_external_circuit_table_default_out_of_range_clamps_to_edge(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,absorbed_power_W\n0.0,10.0\n1e-6,20.0\n', encoding='utf-8')
    chamber = _single_zone_chamber()
    step = SimpleNamespace(power_ports={'circuit_waveform': {'file': str(csv_path)}})
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    result = backend.evaluate(PowerRequest(time_s=2.0e-6, state_vector=None, recipe_step=step, chamber=chamber, metadata={}))

    assert result.port_power_W['circuit_waveform'] == pytest.approx(20.0)


def test_external_circuit_table_zero_order_hold_uses_previous_sample(tmp_path: Path) -> None:
    csv_path = tmp_path / 'circuit.csv'
    csv_path.write_text('time_s,absorbed_power_W\n0.0,10.0\n1e-6,20.0\n', encoding='utf-8')
    chamber = _single_zone_chamber()
    step = SimpleNamespace(power_ports={'circuit_waveform': {'file': str(csv_path), 'interpolation': 'zoh'}})
    backend = ExternalCircuitTableBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    result = backend.evaluate(PowerRequest(time_s=5.0e-7, state_vector=None, recipe_step=step, chamber=chamber, metadata={}))

    assert result.port_power_W['circuit_waveform'] == pytest.approx(10.0)


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


def test_external_circuit_table_file_key_run_summary_includes_sources(tmp_path: Path) -> None:
    case_path = tmp_path / 'case_external_table.yaml'
    case_path.write_text(
        yaml.safe_dump(
            {
                'include': str(ROOT / 'examples' / 'configs' / 'case_zdplaskin_example2.yaml'),
                'files': {
                    'chamber': str(ROOT / 'examples' / 'configs' / 'chamber_zdplaskin_example2.yaml'),
                    'recipe': str(ROOT / 'examples' / 'configs' / 'recipe_zdplaskin_example2.yaml'),
                    'chemistry': {
                        'manifest': str(ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml')
                    },
                    'output_dir': str(tmp_path / 'out'),
                    'external_inputs': {
                        'circuit_result_csv': str(ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv')
                    },
                },
                'physics': {'electrical_backend': 'external_circuit_table'},
                'outputs': {'plots': {'enabled': False}},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    result = run_from_yaml(case_path)

    summary = result['summary']
    interface = summary['electrical_coupling']['circuit_interface']
    source = interface['table_sources']['dc_series_drive']
    assert interface['kind'] == 'external_table_one_way'
    assert source['power_column'] == 'absorbed_power_W'
    assert source['voltage_column'] == 'voltage_V'
    assert source['current_column'] == 'current_A'
    assert source['power_source'] == 'column'
    assert source['time_start_s'] <= summary['t_start_s'] + 1.0e-18
    assert source['time_end_s'] > source['time_start_s']
