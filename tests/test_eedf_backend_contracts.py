from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_global.config.loader import load_run_config, resolve_run_paths
from plasma_global.config.models import (
    Boltzmann2TermConfig,
    SwarmCacheConfig,
    SwarmConfig,
    SwarmEnergyGridConfig,
    SwarmReducedFieldGridConfig,
)
from plasma_global.config.validator import validate_run_config
from plasma_global.eedf.base import EEDFRequest
from plasma_global.eedf.boltzmann_2term import Boltzmann2TermSwarmModel
from plasma_global import build_case, load_case_from_yaml
from tests.case_helpers import ARGON_LXCAT_CASE, SMOKE_CASE


def test_auto_swarm_closure_uses_electron_energy_state() -> None:
    built = build_case(load_case_from_yaml(ARGON_LXCAT_CASE))
    system = built.system
    y0 = system.initial_state()
    coupled = system.electrical_adapter.evaluate(0.0, y0, system.current_step(0.0))

    for zone_id, eedf in coupled.eedf_by_zone.items():
        assert eedf.transport.lookup_mode == 'mean_energy'
        assert eedf.transport.mean_energy_eV == coupled.mean_e_by_zone[zone_id]


def test_swarm_closure_validation_reports_unrecognized_mode() -> None:
    run_config = load_run_config(ARGON_LXCAT_CASE)
    run_config.swarm.closure = 'fieldish'
    report = validate_run_config(run_config, resolve_run_paths(run_config, ARGON_LXCAT_CASE))

    assert any(msg.code == 'SWARM_CLOSURE_UNRECOGNIZED' for msg in report.messages)


def test_local_field_closure_validation_is_error_free() -> None:
    run_config = load_run_config(ARGON_LXCAT_CASE)
    run_config.swarm.closure = 'local_field'
    report = validate_run_config(run_config, resolve_run_paths(run_config, ARGON_LXCAT_CASE))

    assert not any(msg.level == 'ERROR' for msg in report.messages)


def test_swarm_model_validation_reports_unrecognized_model() -> None:
    run_config = load_run_config(ARGON_LXCAT_CASE)
    run_config.swarm.model_name = 'mystery_swarm'
    report = validate_run_config(run_config, resolve_run_paths(run_config, ARGON_LXCAT_CASE))

    message = next(msg.message for msg in report.messages if msg.code == 'SWARM_MODEL_UNRECOGNIZED')
    assert 'boltzmann_2term' in message
    assert 'table' in message
    assert 'maxwell' not in message


def test_swarm_model_validation_rejects_non_swarm_model_name() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.physics.eedf_backend = 'swarm'
    run_config.swarm.model_name = 'maxwell'
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))

    assert any(msg.code == 'SWARM_MODEL_UNRECOGNIZED' for msg in report.messages)


def test_table_swarm_model_validation_requires_table_file() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.physics.eedf_backend = 'swarm'
    run_config.swarm.model_name = 'table'
    run_config.swarm.table.file = None
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))

    assert any(msg.code == 'SWARM_TABLE_FILE_MISSING' for msg in report.messages)


def test_backend_name_validation_reports_unknown_entries() -> None:
    run_config = load_run_config(SMOKE_CASE)
    run_config.physics.eedf_backend = 'mystery_eedf'
    run_config.physics.integrator = 'mystery_integrator'
    report = validate_run_config(run_config, resolve_run_paths(run_config, SMOKE_CASE))
    codes = {msg.code for msg in report.messages}

    assert 'EEDF_BACKEND_UNRECOGNIZED' in codes
    assert 'INTEGRATOR_UNSUPPORTED' in codes


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
