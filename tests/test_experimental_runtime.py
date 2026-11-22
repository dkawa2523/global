from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plasma_global import load_case
from plasma_global.build import compile_case, simulate_case
from plasma_global.input.load import case_from_mapping

MINIMAL = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"
SMOKE = (
    Path(__file__).parents[1] / "plasma_global" / "examples" / "cases" / "smoke.yaml"
)


def test_smoke_film_and_inventory_are_independent_runtime_states() -> None:
    case = load_case(SMOKE)
    compiled = compile_case(case)
    labels = compiled.model.layout.labels
    state = compiled.model.initial_state(compiled.initial_state)

    assert labels[-4:] == (
        "film[wafer]",
        "film[grounded_wall]",
        "film[source_wall]",
        "inventory[wafer,F_reservoir]",
    )
    assert "coverage[wafer,wafer:poly*]" not in labels
    np.testing.assert_array_equal(
        state[compiled.model.layout.extension_slice], np.zeros(4)
    )

    state[labels.index("n[process,CF3]")] = 1.0e19
    state[labels.index("n[process,F]")] = 1.0e19
    evaluated = compiled.model.evaluate(
        compiled.segments[0].start_s, state, compiled.segments[0]
    )
    assert evaluated.derivative[labels.index("film[wafer]")] > 0.0
    assert evaluated.derivative[labels.index("inventory[wafer,F_reservoir]")] > 0.0

    result = simulate_case(case)
    assert result.status.success
    assert result.state_labels[-4:] == labels[-4:]
    assert np.isfinite(result.state[:, -4:]).all()


def test_unmapped_surface_reaction_does_not_guess_inventory_yield() -> None:
    source = load_case(SMOKE)
    data = source.model_dump(mode="python")
    data["experimental"]["wall_inventory"]["events"] = []
    compiled = compile_case(
        case_from_mapping(data, base_directory=source.base_directory)
    )
    labels = compiled.model.layout.labels
    state = compiled.model.initial_state(compiled.initial_state)
    state[labels.index("n[process,F]")] = 1.0e19

    evaluated = compiled.model.evaluate(
        compiled.segments[0].start_s, state, compiled.segments[0]
    )

    assert evaluated.derivative[
        labels.index("inventory[wafer,F_reservoir]")
    ] == pytest.approx(0.0)
