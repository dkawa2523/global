from __future__ import annotations

from pathlib import Path

import pytest

from plasma_global.reactor.io import load_chamber_config
from plasma_global.reactor.models import (
    ChamberConfig,
    Edge,
    Inlet,
    PowerPort,
    Pump,
    RecipeConfig,
    RecipeStep,
    Surface,
    Zone,
)
from plasma_global.reactor.validation import validate_reactor_config


def _valid_chamber() -> ChamberConfig:
    return ChamberConfig(
        chamber_id='validation_test',
        description='',
        zones=[
            Zone(
                zone_id='plasma',
                description='',
                volume_m3=1.0e-3,
                pressure_Pa=10.0,
                gas_temperature_K=300.0,
            )
        ],
        edges=[],
        surfaces=[
            Surface(
                surface_id='wafer',
                zone_id='plasma',
                kind='wall',
                area_m2=1.0e-2,
                material='Si',
                temperature_K=300.0,
                site_density_m2=1.0e19,
            )
        ],
        gas_inlets=[Inlet(inlet_id='feed', zone_id='plasma', flow_sccm={'Ar': 10.0}, temperature_K=300.0)],
        pumps=[Pump(pump_id='pump', zone_id='plasma', speed_m3_s=1.0e-3)],
        power_ports=[
            PowerPort(
                port_id='source',
                kind='direct_power',
                zone_id='plasma',
                coupling_target='plasma',
            )
        ],
    )


def test_reactor_validation_accepts_consistent_chamber_and_recipe() -> None:
    chamber = _valid_chamber()
    recipe = RecipeConfig(
        recipe_id='validation_test',
        description='',
        steps=[
            RecipeStep(
                step_id='main',
                t_start_s=0.0,
                t_end_s=1.0e-6,
                gas_inlets={'feed': {'Ar': 10.0}},
                power_ports={'source': {'value_W': 100.0}},
                surface_overrides={'wafer': {'temperature_K': 310.0}},
            )
        ],
    )

    assert validate_reactor_config(chamber, recipe) == []


def test_reactor_validation_reports_structural_errors() -> None:
    chamber = _valid_chamber()
    chamber.zones[0].volume_m3 = -1.0
    chamber.edges.append(Edge(edge_id='bad_edge', from_zone='plasma', to_zone='missing', conductance_m3_s=1.0))
    recipe = RecipeConfig(
        recipe_id='validation_test',
        description='',
        steps=[
            RecipeStep(
                step_id='bad',
                t_start_s=1.0,
                t_end_s=0.5,
                gas_inlets={'missing_feed': {'Ar': 1.0}},
                power_ports={'missing_source': {'value_W': 10.0}},
                surface_overrides={'missing_surface': {'temperature_K': 310.0}},
            )
        ],
    )

    codes = {issue.code for issue in validate_reactor_config(chamber, recipe)}

    assert {
        'ZONE_VOLUME_INVALID',
        'UNKNOWN_EDGE_ZONE',
        'RECIPE_STEP_TIME_INVALID',
        'UNKNOWN_STEP_INLET',
        'UNKNOWN_STEP_POWER_PORT',
        'UNKNOWN_STEP_SURFACE',
    } <= codes


def test_reactor_validation_reports_recipe_step_gaps() -> None:
    chamber = _valid_chamber()
    recipe = RecipeConfig(
        recipe_id='validation_test',
        description='',
        steps=[
            RecipeStep(step_id='first', t_start_s=0.0, t_end_s=1.0),
            RecipeStep(step_id='second', t_start_s=1.5, t_end_s=2.0),
        ],
    )

    codes = {issue.code for issue in validate_reactor_config(chamber, recipe)}

    assert 'RECIPE_STEP_GAP' in codes


def test_chamber_loader_rejects_unsupported_pump_target_pressure(tmp_path: Path) -> None:
    path = tmp_path / 'chamber.yaml'
    path.write_text(
        """
chamber_id: unsupported_pump_target
zones:
  - zone_id: plasma
    description: ''
    volume_m3: 1.0e-3
    pressure_Pa: 10.0
    gas_temperature_K: 300.0
edges: []
surfaces: []
gas_inlets: []
pumps:
  - pump_id: pump
    zone_id: plasma
    speed_m3_s: 1.0e-3
    target_pressure_Pa: 10.0
power_ports: []
""",
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='Unsupported pump keys: target_pressure_Pa'):
        load_chamber_config(path)
