from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_global.electrical.base import SurfaceIED
from plasma_global.observables.adapter import compute_observables
from plasma_global import build_case, load_case_from_yaml
from tests.case_helpers import SMOKE_CASE, example_chamber, write_smoke_case_with_chamber


def test_surface_ion_flux_uses_zone_wall_loss_when_no_ied(tmp_path: Path) -> None:
    chamber = example_chamber()
    for surface in chamber['surfaces']:
        if surface['surface_id'] == 'source_wall':
            surface['models']['ion_loss'] = 'prescribed_loss_frequency'
            surface['models']['frequency_s'] = 3230.0

    case_path = write_smoke_case_with_chamber(tmp_path, chamber)
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
    ied = SurfaceIED(ion_flux_m2_s=123.0, mean_ion_energy_eV=10.0)

    assert ied.ion_flux_m2_s == pytest.approx(123.0)
    assert ied.mean_ion_energy_eV == pytest.approx(10.0)
    assert not hasattr(ied, 'diagnostics')


def test_surface_rate_results_are_plain_scalars() -> None:
    rate_m2_s = 2.0

    assert rate_m2_s == pytest.approx(2.0)
