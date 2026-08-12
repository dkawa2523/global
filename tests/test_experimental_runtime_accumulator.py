from __future__ import annotations

import numpy as np
import pytest

from plasma_global.experimental.accumulators import (
    GenericStateAccumulator,
    ProcessContext,
    SurfaceEventRate,
)
from plasma_global.experimental.runtime import (
    ExperimentalRuntimeAccumulator,
    compile_generic_state_accumulator,
)


def _generic_specification() -> dict[str, object]:
    return {
        "state_variables": {
            "global_marker": {
                "scope": "global",
                "initial": 1.0,
                "lower_bound": None,
                "upper_bound": 5.0,
            },
            "zone_marker": {
                "scope": "zone",
                "zones": ["downstream", "plasma"],
                "initial": 2.0,
                "lower_bound": 0.0,
            },
            "surface_marker": {
                "scope": "surface",
                "surfaces": ["wall_b", "wall_a"],
                "initial": -1.0,
                "lower_bound": None,
            },
        },
        "processes": {
            "global_relaxation": {
                "kind": "relaxation",
                "target": "global_marker",
                "equilibrium": 5.0,
                "tau_s": 2.0,
            },
            "selected_zone_source": {
                "kind": "source",
                "target": "zone_marker",
                "zones": ["plasma"],
                "value": 3.0,
            },
            "selected_surface_driver": {
                "kind": "ion_flux_source",
                "target": "surface_marker",
                "surfaces": ["wall_b"],
                "coefficient": 0.5,
            },
        },
    }


def _compiled_generic() -> GenericStateAccumulator:
    compiled = compile_generic_state_accumulator(
        _generic_specification(),
        zone_ids=("plasma", "downstream"),
        surface_ids=("wall_a", "wall_b"),
    )
    assert compiled is not None
    return compiled


def test_compile_generic_state_preserves_declaration_and_owner_order() -> None:
    compiled = _compiled_generic()

    assert tuple(state.state_id for state in compiled.states) == (
        "global_marker",
        "zone_marker",
        "surface_marker",
    )
    assert tuple(tuple(state.owners) for state in compiled.states) == (
        ("global",),
        ("downstream", "plasma"),
        ("wall_b", "wall_a"),
    )
    assert compiled.labels == (
        "global_marker[global]",
        "zone_marker[downstream]",
        "zone_marker[plasma]",
        "surface_marker[wall_b]",
        "surface_marker[wall_a]",
    )
    np.testing.assert_array_equal(compiled.initial_state(), [1.0, 2.0, 2.0, -1.0, -1.0])
    derivative = compiled.rhs(
        compiled.initial_state(),
        ProcessContext({"ion_flux_m2_s": {"wall_b": 10.0}}),
    )
    np.testing.assert_array_equal(derivative, [2.0, 0.0, 3.0, 5.0, 0.0])
    with pytest.raises(KeyError, match="ion_flux_m2_s"):
        compiled.rhs(compiled.initial_state())


def test_runtime_layout_and_rhs_combine_film_inventory_and_generic_state() -> None:
    generic = _compiled_generic()
    events = (
        SurfaceEventRate(
            event_id="deposit_b",
            zone_id="plasma",
            surface_id="wall_b",
            rate_m2_s=0.0,
            area_m2=4.0,
            site_density_m2=10.0,
            inventory_particles_per_event={"C": 2.0},
            film_layers_per_event=1.0,
        ),
        SurfaceEventRate(
            event_id="inventory_a",
            zone_id="plasma",
            surface_id="wall_a",
            rate_m2_s=0.0,
            area_m2=2.0,
            site_density_m2=10.0,
            inventory_particles_per_event={"A": 4.0},
        ),
    )
    runtime = ExperimentalRuntimeAccumulator(
        generic=generic,
        film_surfaces=("wall_b", "wall_a"),
        initial_inventory_by_surface={
            "wall_a": {"B": 2.0, "A": 1.0},
            "wall_b": {"C": 3.0},
        },
        surface_event_templates=events,
        monolayer_thickness_m=0.5,
    )

    assert runtime.labels == (
        "film[wall_b]",
        "film[wall_a]",
        "inventory[wall_a,B]",
        "inventory[wall_a,A]",
        "inventory[wall_b,C]",
        "extra[global_marker,global]",
        "extra[zone_marker,downstream]",
        "extra[zone_marker,plasma]",
        "extra[surface_marker,wall_b]",
        "extra[surface_marker,wall_a]",
    )
    assert runtime.lower_bounds == (0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0, 0.0, None, None)
    expected_initial = [0.0, 0.0, 2.0, 1.0, 3.0, 1.0, 2.0, 2.0, -1.0, -1.0]
    initial = runtime.initial_state()
    np.testing.assert_array_equal(initial, expected_initial)
    initial[0] = 99.0
    np.testing.assert_array_equal(runtime.initial_state(), expected_initial)

    derivative = runtime.rhs(
        runtime.initial_state(),
        drivers={"ion_flux_m2_s": {"wall_b": 10.0}},
        surface_rates_m2_s={"deposit_b": 20.0, "inventory_a": 3.0},
    )

    np.testing.assert_array_equal(
        derivative,
        [1.0, 0.0, 0.0, 24.0, 160.0, 2.0, 0.0, 3.0, 5.0, 0.0],
    )
    assert tuple(event.rate_m2_s for event in runtime.surface_event_templates) == (
        0.0,
        0.0,
    )
    invalid = runtime.initial_state()
    invalid[0] = -1.0
    with pytest.raises(
        ValueError, match="film thickness and wall inventory must be nonnegative"
    ):
        runtime.rhs(invalid)


def test_runtime_normalizes_numpy_scalar_inventory_and_event_rates() -> None:
    runtime = ExperimentalRuntimeAccumulator(
        film_surfaces=("wall",),
        initial_inventory_by_surface={"wall": {"A": np.float32(1.25)}},
        surface_event_templates=(
            SurfaceEventRate(
                event_id="deposit",
                zone_id="plasma",
                surface_id="wall",
                rate_m2_s=0.0,
                area_m2=1.7,
                site_density_m2=3.1,
                inventory_particles_per_event={"A": 0.7},
                film_layers_per_event=0.3,
            ),
        ),
    )

    numpy_rate = runtime.rhs(
        runtime.initial_state(),
        surface_rates_m2_s={"deposit": np.float32(4.78827991)},
    )
    python_rate = runtime.rhs(
        runtime.initial_state(),
        surface_rates_m2_s={"deposit": float(np.float32(4.78827991))},
    )

    assert type(runtime.initial_inventory_by_surface["wall"]["A"]) is float
    np.testing.assert_array_equal(numpy_rate, python_rate)


@pytest.mark.parametrize(
    ("specification", "message"),
    [
        pytest.param(
            {"processes": {"source": {"kind": "source"}}},
            "experimental processes require state_variables",
            id="process-without-state",
        ),
        pytest.param(
            {
                "state_variables": {"known": {"scope": "global"}},
                "processes": {
                    "source": {"kind": "source", "target": "missing", "value": 1.0}
                },
            },
            "targets unknown state 'missing'",
            id="unknown-target",
        ),
        pytest.param(
            {
                "state_variables": {"known": {"scope": "global"}},
                "processes": {
                    "source": {
                        "kind": "ion_flux_source",
                        "target": "known",
                        "coefficient": 1.0,
                    }
                },
            },
            "ion_flux_source target must use surface scope",
            id="driver-requires-surface",
        ),
    ],
)
def test_compile_generic_state_rejects_invalid_declarations(
    specification: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_generic_state_accumulator(
            specification,
            zone_ids=("plasma",),
            surface_ids=("wall",),
        )


@pytest.mark.parametrize(
    ("film_layers", "inventory", "message"),
    [
        pytest.param(1.0, {}, "undeclared film state 'wall'", id="film"),
        pytest.param(
            0.0,
            {"A": 1.0},
            r"undeclared inventory state \('wall', 'A'\)",
            id="inventory",
        ),
    ],
)
def test_runtime_rejects_nonzero_surface_output_without_declared_state(
    film_layers: float, inventory: dict[str, float], message: str
) -> None:
    runtime = ExperimentalRuntimeAccumulator(
        generic=_compiled_generic(),
        surface_event_templates=(
            SurfaceEventRate(
                event_id="event",
                zone_id="plasma",
                surface_id="wall",
                rate_m2_s=0.0,
                area_m2=1.0,
                site_density_m2=1.0,
                inventory_particles_per_event=inventory,
                film_layers_per_event=film_layers,
            ),
        ),
    )

    with pytest.raises(ValueError, match=message):
        runtime.rhs(
            runtime.initial_state(),
            drivers={"ion_flux_m2_s": {"wall_b": 0.0}},
            surface_rates_m2_s={"event": 1.0},
        )
