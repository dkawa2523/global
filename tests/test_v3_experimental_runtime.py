from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global import load_case
from plasma_global.build import compile_case, simulate_case
from plasma_global.errors import ModelDomainError
from plasma_global.input.schema import CaseSpec

MINIMAL = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"
SMOKE = Path(__file__).parents[1] / "examples" / "v3" / "cases" / "smoke.yaml"


def _case_with_generic_extensions(tmp_path: Path) -> CaseSpec:
    base_case = load_case(MINIMAL)
    source_manifest = base_case.chemistry.manifest
    manifest = yaml.safe_load(source_manifest.read_text(encoding="utf-8"))
    for key in (
        "species",
        "gas_reactions",
        "boundary_reactions",
        "surface_reactions",
        "rate_models",
        "cross_sections",
    ):
        if manifest.get(key):
            manifest[key] = str((source_manifest.parent / manifest[key]).resolve())
    manifest["experimental"] = {
        "state_variables": {
            "zone_marker": {
                "scope": "zone",
                "initial": 2.0,
                "lower_bound": 0.0,
            },
            "hidden_zone": {
                "scope": "zone",
                "initial": 4.0,
                "lower_bound": 0.0,
            },
            "wall_charge": {
                "scope": "surface",
                "surfaces": ["wall"],
                "initial": -1.0,
                "lower_bound": None,
            },
        },
        "processes": {
            "zone_source": {
                "kind": "source",
                "target": "zone_marker",
                "value": 3.0,
            },
            "zone_relaxation": {
                "kind": "relaxation",
                "target": "hidden_zone",
                "tau_s": 2.0,
                "equilibrium": 10.0,
            },
            "wall_charge_from_flux": {
                "kind": "ion_flux_source",
                "target": "wall_charge",
                "coefficient": 1.0e-18,
                "surfaces": ["wall"],
            },
        },
    }
    target_manifest = tmp_path / "chemistry.yaml"
    target_manifest.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    data = base_case.model_dump(mode="python")
    data["chemistry"]["manifest"] = target_manifest
    data["experimental"] = {"extensions": {}}
    return CaseSpec.model_validate(data)


def test_generic_extension_state_processes_reach_rhs_and_result(
    tmp_path: Path,
) -> None:
    case = _case_with_generic_extensions(tmp_path)
    compiled = compile_case(case)
    labels = compiled.model.layout.labels
    state = compiled.model.initial_state(compiled.initial_state)
    evaluated = compiled.model.evaluate(
        compiled.segments[0].start_s, state, compiled.segments[0]
    )

    assert labels[-3:] == (
        "extra[zone_marker,plasma]",
        "extra[hidden_zone,plasma]",
        "extra[wall_charge,wall]",
    )
    np.testing.assert_allclose(state[-3:], [2.0, 4.0, -1.0])
    assert evaluated.derivative[-3] == pytest.approx(3.0)
    assert evaluated.derivative[-2] == pytest.approx(3.0)
    assert evaluated.derivative[-1] > 0.0

    result = simulate_case(case)
    assert result.status.success
    assert result.state_labels[-3:] == labels[-3:]
    assert np.isfinite(result.state[:, -3:]).all()


def test_generic_extension_domain_violation_is_not_projected(tmp_path: Path) -> None:
    compiled = compile_case(_case_with_generic_extensions(tmp_path))
    state = compiled.model.initial_state(compiled.initial_state)
    state[compiled.model.layout.labels.index("extra[zone_marker,plasma]")] = -1.0

    with pytest.raises(ModelDomainError, match="below its declared domain"):
        compiled.model.evaluate(
            compiled.segments[0].start_s, state, compiled.segments[0]
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
    compiled = compile_case(CaseSpec.model_validate(data))
    labels = compiled.model.layout.labels
    state = compiled.model.initial_state(compiled.initial_state)
    state[labels.index("n[process,F]")] = 1.0e19

    evaluated = compiled.model.evaluate(
        compiled.segments[0].start_s, state, compiled.segments[0]
    )

    assert evaluated.derivative[
        labels.index("inventory[wafer,F_reservoir]")
    ] == pytest.approx(0.0)
