from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plasma_global import build_case, load_case_from_yaml
from plasma_global.workflows.runner import run_from_yaml
from tests.case_helpers import write_case_with_output_dir


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
    assert 'film_thickness' not in built.state_layout.slices
    assert not any(label.startswith('film[') for label in labels)
    assert system.surface_core.enabled is False
    assert system.surface_core.surface_reactions == []


def test_crane_two_reaction_argon_matches_committed_crane_output(tmp_path: Path) -> None:
    result = run_from_yaml(write_case_with_output_dir(tmp_path, CRANE_CASE))
    labels = result['system'].state_labels()
    final_state = {
        label: result['solution'].y[idx, -1]
        for idx, label in enumerate(labels)
    }
    reference = yaml.safe_load(CRANE_REFERENCE.read_text(encoding='utf-8'))['converted_output_final']

    assert final_state['n[plasma,Ar_plus]'] == pytest.approx(float(reference['Ar_plus_density_m3']), rel=5.0e-4)
    assert final_state['n[plasma,Ar]'] == pytest.approx(float(reference['Ar_density_m3']), rel=1.0e-8)
