from __future__ import annotations

import numpy as np
import pytest

from plasma_global.chemistry.models import E_CHARGE
from plasma_global import build_case, load_case_from_yaml
from plasma_global.observables.adapter import compute_observables
from plasma_global.physics.gas_closure import electron_density_from_state_row
from tests.case_helpers import SMOKE_CASE, ZDPLASKIN_CASE


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


def test_observables_do_not_flatten_power_port_diagnostics() -> None:
    built = build_case(load_case_from_yaml(SMOKE_CASE))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['total_absorbed_power_W'] > 0.0
    assert not any(key.startswith('port_') for key in rec)


def test_observables_include_model_budget_and_electrical_waveform_columns() -> None:
    built = build_case(load_case_from_yaml(ZDPLASKIN_CASE))
    system = built.system
    y0 = system.initial_state()
    rec = compute_observables(system, np.array([0.0]), y0.reshape(-1, 1))[0]

    assert rec['electrical_dc_series_drive_source_voltage_V'] == pytest.approx(1000.0)
    assert rec['electrical_dc_series_drive_current_A'] != 0.0
    assert rec['table_lookup_clipped_plasma'] in {0, 1}
    assert system.run_config.outputs.budgets.enabled is True
    assert any(key.startswith('reaction_rate_plasma_') for key in rec)
    assert any(key.startswith('species_source_plasma_') for key in rec)
    assert any(key.startswith('species_loss_plasma_') for key in rec)
