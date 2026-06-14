from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from plasma_global.electrical.base import PowerRequest
from plasma_global.electrical.circuit_models import DCSeriesCircuitConfig, DCSeriesCircuitModel, PlasmaLoadState
from plasma_global.electrical.dc_series import DCSeriesCircuitBackend
from plasma_global.workflows.context import ELECTRICAL_REGISTRY, load_case_from_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_dc_series_model_voltage_collapses_as_density_increases() -> None:
    model = DCSeriesCircuitModel()
    cfg = DCSeriesCircuitConfig(
        source_voltage_V=1000.0,
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
        electron_mobility_m2_V_s=0.02,
    )
    weak = PlasmaLoadState(
        electron_density_m3=1.0e12,
        mean_energy_eV=3.0,
        pressure_Pa=1.0e4,
        gas_temperature_K=300.0,
    )
    dense = PlasmaLoadState(
        electron_density_m3=1.0e17,
        mean_energy_eV=3.0,
        pressure_Pa=1.0e4,
        gas_temperature_K=300.0,
    )

    weak_solution = model.solve(cfg, weak)
    dense_solution = model.solve(cfg, dense)

    assert weak_solution.gap_voltage_V > dense_solution.gap_voltage_V
    assert dense_solution.current_A > weak_solution.current_A
    assert dense_solution.absorbed_power_W > 0.0


def test_dc_series_model_can_use_transport_table_mobility() -> None:
    model = DCSeriesCircuitModel()
    fixed_cfg = DCSeriesCircuitConfig(
        source_voltage_V=1000.0,
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
        electron_mobility_m2_V_s=0.1,
    )
    table_cfg = DCSeriesCircuitConfig(
        source_voltage_V=1000.0,
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
        electron_mobility_m2_V_s=0.1,
        mobility_source='table',
    )
    load = PlasmaLoadState(
        electron_density_m3=1.0e17,
        mean_energy_eV=3.0,
        pressure_Pa=1.0e4,
        gas_temperature_K=300.0,
        electron_mobility_m2_V_s=1.0,
    )

    fixed_solution = model.solve(fixed_cfg, load)
    table_solution = model.solve(table_cfg, load)

    assert fixed_solution.electron_mobility_m2_V_s == pytest.approx(0.1)
    assert table_solution.electron_mobility_m2_V_s == pytest.approx(1.0)
    assert table_solution.gap_voltage_V < fixed_solution.gap_voltage_V


def test_dc_series_model_supports_conductance_multiplier() -> None:
    model = DCSeriesCircuitModel()
    base_cfg = DCSeriesCircuitConfig(
        source_voltage_V=1000.0,
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
        electron_mobility_m2_V_s=0.1,
    )
    boosted_cfg = DCSeriesCircuitConfig(
        source_voltage_V=1000.0,
        ballast_resistance_ohm=1.0e5,
        gap_m=4.0e-3,
        electrode_area_m2=5.0e-5,
        electron_mobility_m2_V_s=0.1,
        conductance_multiplier=3.0,
    )
    load = PlasmaLoadState(
        electron_density_m3=1.0e17,
        mean_energy_eV=3.0,
        pressure_Pa=1.0e4,
        gas_temperature_K=300.0,
    )

    base_solution = model.solve(base_cfg, load)
    boosted_solution = model.solve(boosted_cfg, load)

    assert boosted_solution.plasma_resistance_ohm == pytest.approx(base_solution.plasma_resistance_ohm / 3.0)
    assert boosted_solution.gap_voltage_V < base_solution.gap_voltage_V


def test_dc_series_backend_exposes_required_circuit_metadata() -> None:
    zone = SimpleNamespace(zone_id='plasma', pressure_Pa=100.0, gas_temperature_K=300.0)
    port = SimpleNamespace(
        port_id='dc_drive',
        kind='dc_series_circuit',
        zone_id='plasma',
        coupling_target='plasma',
        parameters={},
    )
    chamber = SimpleNamespace(
        zones=[zone],
        zone_by_id={'plasma': zone},
        power_port_by_id={'dc_drive': port},
    )
    step = SimpleNamespace(
        power_ports={
            'dc_drive': {
                'zone_id': 'plasma',
                'source_voltage_V': 500.0,
                'ballast_resistance_ohm': 1.0e5,
                'gap_m': 4.0e-3,
                'electrode_area_m2': 1.0e-4,
                'electron_mobility_m2_V_s': 0.05,
                'voltage': {
                    'waveform': 'pulsed_square',
                    'high_V': 500.0,
                    'low_V': 0.0,
                    'duty_cycle': 0.5,
                    'frequency_Hz': 1000.0,
                },
            }
        }
    )
    backend = DCSeriesCircuitBackend()
    backend.prepare(chamber=chamber, recipe=SimpleNamespace(steps=[step]), run_config=SimpleNamespace())

    on_result = backend.evaluate(
        PowerRequest(
            time_s=1.0e-4,
            state_vector=None,
            recipe_step=step,
            chamber=chamber,
            metadata={
                'zone_electron_density_m3': {'plasma': 1.0e16},
                'zone_mean_energy_eV': {'plasma': 3.0},
                'zone_pressure_Pa': {'plasma': 100.0},
                'zone_gas_temperature_K': {'plasma': 300.0},
                'zone_electron_mobility_m2_V_s': {'plasma': 0.2},
            },
        )
    )
    off_result = backend.evaluate(
        PowerRequest(
            time_s=6.0e-4,
            state_vector=None,
            recipe_step=step,
            chamber=chamber,
            metadata={'zone_electron_density_m3': {'plasma': 1.0e16}},
        )
    )

    detail = on_result.metadata['port_details']['dc_drive']
    assert detail['backend'] == 'dc_series_circuit'
    assert detail['gap_voltage_V'] > 0.0
    assert detail['current_A'] > 0.0
    assert detail['electron_mobility_m2_V_s'] == pytest.approx(0.05)
    assert on_result.metadata['circuit_interface']['external_circuit_ready'] is True
    assert off_result.port_power_W['dc_drive'] == 0.0


def test_dc_series_backend_is_registered() -> None:
    details = ELECTRICAL_REGISTRY.details()
    assert 'ballast-resistor' in details['dc_series_circuit']['description']


def test_dc_series_case_validation_reports_missing_required_parameters(tmp_path: Path) -> None:
    chamber = {
        'chamber_id': 'dc_missing_parameter_test',
        'zones': [
            {
                'zone_id': 'plasma',
                'description': '',
                'volume_m3': 1.0e-3,
                'pressure_Pa': 100.0,
                'gas_temperature_K': 300.0,
                'role': 'process',
            }
        ],
        'edges': [],
        'surfaces': [],
        'gas_inlets': [],
        'pumps': [],
        'power_ports': [
            {
                'port_id': 'dc_drive',
                'kind': 'dc_series_circuit',
                'zone_id': 'plasma',
                'coupling_target': 'plasma',
                'parameters': {},
            }
        ],
    }
    recipe = {
        'recipe_id': 'dc_missing_parameter_test',
        'steps': [
            {
                'step_id': 'pulse',
                't_start_s': 0.0,
                't_end_s': 1.0e-6,
                'power_ports': {
                    'dc_drive': {
                        'source_voltage_V': 1000.0,
                        'ballast_resistance_ohm': 1.0e5,
                        'gap_m': 4.0e-3,
                    }
                },
            }
        ],
    }
    chamber_path = tmp_path / 'chamber.yaml'
    recipe_path = tmp_path / 'recipe.yaml'
    case_path = tmp_path / 'case.yaml'
    chamber_path.write_text(yaml.safe_dump(chamber, sort_keys=False), encoding='utf-8')
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding='utf-8')
    case_path.write_text(
        yaml.safe_dump(
            {
                'case': {'name': 'dc_missing_parameter_test'},
                'files': {
                    'chamber': str(chamber_path),
                    'recipe': str(recipe_path),
                    'chemistry': {
                        'manifest': str(ROOT / 'examples' / 'chemistry_zdplaskin_example2' / 'chemistry_manifest.yaml')
                    },
                    'output_dir': str(tmp_path / 'out'),
                },
                'physics': {'electrical_backend': 'dc_series_circuit'},
            },
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='DC_SERIES_CIRCUIT_CONFIG_INVALID'):
        load_case_from_yaml(case_path)
