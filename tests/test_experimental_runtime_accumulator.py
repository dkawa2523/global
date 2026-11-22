from __future__ import annotations

import numpy as np
import pytest

from plasma_global.experimental.accumulators import (
    SurfaceEventRate,
)
from plasma_global.experimental.runtime import (
    ExperimentalRuntimeAccumulator,
)


def test_runtime_layout_and_rhs_combine_film_inventory() -> None:
    events = (
        SurfaceEventRate(
            reaction_id="deposit_b",
            zone_id="plasma",
            surface_id="wall_b",
            area_m2=4.0,
            site_density_m2=10.0,
            inventory_particles_per_event={"C": 2.0},
            film_layers_per_event=1.0,
        ),
        SurfaceEventRate(
            reaction_id="inventory_a",
            zone_id="plasma",
            surface_id="wall_a",
            area_m2=2.0,
            site_density_m2=10.0,
            inventory_particles_per_event={"A": 4.0},
        ),
    )
    runtime = ExperimentalRuntimeAccumulator(
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
    )
    expected_initial = [0.0, 0.0, 2.0, 1.0, 3.0]
    initial = runtime.initial_state()
    np.testing.assert_array_equal(initial, expected_initial)
    initial[0] = 99.0
    np.testing.assert_array_equal(runtime.initial_state(), expected_initial)

    derivative = runtime.rhs(
        runtime.initial_state(),
        surface_rates_m2_s={
            ("deposit_b", "wall_b"): 20.0,
            ("inventory_a", "wall_a"): 3.0,
        },
    )

    np.testing.assert_array_equal(
        derivative,
        [1.0, 0.0, 0.0, 24.0, 160.0],
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
                reaction_id="deposit",
                zone_id="plasma",
                surface_id="wall",
                area_m2=1.7,
                site_density_m2=3.1,
                inventory_particles_per_event={"A": 0.7},
                film_layers_per_event=0.3,
            ),
        ),
    )

    numpy_rate = runtime.rhs(
        runtime.initial_state(),
        surface_rates_m2_s={("deposit", "wall"): np.float32(4.78827991)},
    )
    python_rate = runtime.rhs(
        runtime.initial_state(),
        surface_rates_m2_s={("deposit", "wall"): float(np.float32(4.78827991))},
    )

    assert type(runtime.initial_inventory_by_surface["wall"]["A"]) is float
    np.testing.assert_array_equal(numpy_rate, python_rate)


def test_runtime_surface_plan_preserves_order_without_runtime_dtos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = (
        SurfaceEventRate(
            reaction_id="a",
            zone_id="plasma",
            surface_id="wall",
            area_m2=1.7,
            site_density_m2=3.1,
            inventory_particles_per_event={"A": 0.7},
            film_layers_per_event=0.3,
        ),
        SurfaceEventRate(
            reaction_id="b",
            zone_id="plasma",
            surface_id="wall",
            area_m2=2.3,
            site_density_m2=7.9,
            inventory_particles_per_event={"A": -0.2},
            film_layers_per_event=-0.11,
        ),
        SurfaceEventRate(
            reaction_id="c",
            zone_id="plasma",
            surface_id="wall",
            area_m2=0.13,
            site_density_m2=2.2,
            inventory_particles_per_event={"A": 4.2},
            film_layers_per_event=0.07,
        ),
    )
    runtime = ExperimentalRuntimeAccumulator(
        film_surfaces=("wall",),
        initial_inventory_by_surface={"wall": {"A": 0.0}},
        surface_event_templates=events,
        monolayer_thickness_m=3.0e-10,
    )

    def reject_runtime_dto(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("surface DTOs must not be rebuilt by the RHS")

    monkeypatch.setattr(SurfaceEventRate, "__post_init__", reject_runtime_dto)
    derivative = runtime.rhs(
        runtime.initial_state(),
        surface_rates_m2_s={
            ("a", "wall"): np.float32(0.11),
            ("b", "wall"): 1.3,
            ("c", "wall"): 9.7,
        },
    )

    assert tuple(float(value).hex() for value in derivative) == (
        "0x1.8d61a5ce6a3cbp-34",
        "0x1.350ff971844d0p+2",
    )


def test_runtime_surface_plan_returns_before_zero_rate_accumulation() -> None:
    class RejectMultiplication(float):
        def __mul__(self, _other: object) -> float:
            raise AssertionError("zero surface rates must return before accumulation")

    runtime = ExperimentalRuntimeAccumulator(
        film_surfaces=("wall",),
        surface_event_templates=(
            SurfaceEventRate(
                reaction_id="event",
                zone_id="plasma",
                surface_id="wall",
                area_m2=RejectMultiplication(1.0),
                site_density_m2=1.0,
                film_layers_per_event=1.0,
            ),
        ),
    )

    np.testing.assert_array_equal(runtime.rhs(runtime.initial_state()), [0.0])


@pytest.mark.parametrize("rate", [-1.0, np.nan])
def test_runtime_surface_plan_still_validates_dynamic_rates(rate: float) -> None:
    runtime = ExperimentalRuntimeAccumulator(
        film_surfaces=("wall",),
        surface_event_templates=(
            SurfaceEventRate(
                reaction_id="event",
                zone_id="plasma",
                surface_id="wall",
                area_m2=1.0,
                site_density_m2=1.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError, match="surface event rate must be finite and nonnegative"
    ):
        runtime.rhs(
            runtime.initial_state(), surface_rates_m2_s={("event", "wall"): rate}
        )


def test_runtime_surface_plan_preserves_deferred_monolayer_validation() -> None:
    runtime = ExperimentalRuntimeAccumulator(
        film_surfaces=("wall",),
        surface_event_templates=(
            SurfaceEventRate(
                reaction_id="event",
                zone_id="plasma",
                surface_id="wall",
                area_m2=1.0,
                site_density_m2=1.0,
            ),
        ),
        monolayer_thickness_m=0.0,
    )

    with pytest.raises(
        ValueError, match="monolayer_thickness_m must be finite and positive"
    ):
        runtime.rhs(runtime.initial_state())


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
        film_surfaces=("unused",),
        surface_event_templates=(
            SurfaceEventRate(
                reaction_id="event",
                zone_id="plasma",
                surface_id="wall",
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
            surface_rates_m2_s={("event", "wall"): 1.0},
        )
