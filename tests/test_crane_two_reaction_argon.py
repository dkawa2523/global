from __future__ import annotations

from pathlib import Path

import pytest

from plasma_global.workflows.context import build_case, load_case_from_yaml
from tools.external_benchmarks.crane_two_reaction_argon import build_report


ROOT = Path(__file__).resolve().parents[1]
CRANE_CASE = ROOT / 'examples' / 'configs' / 'case_crane_two_reaction_argon.yaml'
CRANE_REFERENCE = ROOT / 'examples' / 'external' / 'crane_two_reaction_argon_reference.yaml'


def test_crane_two_reaction_argon_loads_explicit_initial_densities() -> None:
    loaded = load_case_from_yaml(CRANE_CASE)
    built = build_case(loaded)
    system = built.system
    y0 = system.initial_state()
    labels = system.state_labels()
    state = {label: y0[idx] for idx, label in enumerate(labels)}

    assert loaded.run_config.case.name == 'crane_two_reaction_argon'
    assert loaded.run_config.physics.electrical_backend == 'direct_power'
    assert loaded.run_config.physics.eedf_backend == 'maxwell'
    assert state['n[plasma,Ar]'] == pytest.approx(2.5e25)
    assert state['n[plasma,Ar_plus]'] == pytest.approx(1.0e6)
    assert state['We[plasma]'] == pytest.approx(3.0 * 1.0e6 * 1.602176634e-19)


def test_crane_two_reaction_argon_matches_committed_crane_output() -> None:
    report = build_report(CRANE_CASE, CRANE_REFERENCE)
    comparison = report['comparison']

    assert comparison['final_electron_density_m3_relative_error'] < 5.0e-4
    assert comparison['final_Ar_plus_density_m3_relative_error'] < 5.0e-4
    assert comparison['final_Ar_density_m3_relative_error'] < 1.0e-8
