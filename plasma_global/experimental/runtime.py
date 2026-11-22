"""Composition facade for opt-in experimental runtime states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from plasma_global.experimental._surface_inventory import (
    SurfaceExecutionPlan,
    add_surface_rates,
    build_runtime_layout,
    compile_surface_execution_plan,
    normalize_film_surfaces,
    normalize_inventory,
    validated_runtime_state,
)
from plasma_global.experimental.accumulators import (
    SurfaceEventRate,
)


@dataclass(frozen=True, slots=True)
class ExperimentalRuntimeAccumulator:
    """One ODE block for film thickness and wall particle inventory.

    Film thickness and wall inventory are extensive bookkeeping states, not
    surface site coverages. Surface chemistry remains the sole owner of the
    coverage/site balance.
    """

    film_surfaces: Sequence[str] = ()
    initial_inventory_by_surface: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    surface_event_templates: Sequence[SurfaceEventRate] = ()
    monolayer_thickness_m: float = 3.0e-10
    labels: tuple[str, ...] = field(init=False)
    _initial: np.ndarray = field(init=False, repr=False)
    _surface_plan: SurfaceExecutionPlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        films = normalize_film_surfaces(self.film_surfaces)
        inventory = normalize_inventory(self.initial_inventory_by_surface)
        templates = tuple(self.surface_event_templates)
        event_keys = tuple(event.event_key for event in templates)
        if len(set(event_keys)) != len(event_keys):
            raise ValueError("surface-event reaction/surface pairs must be unique")
        layout = build_runtime_layout(films, inventory)
        surface_plan = compile_surface_execution_plan(
            templates,
            film_index=layout.film_index,
            inventory_index=layout.inventory_index,
            monolayer_thickness_m=self.monolayer_thickness_m,
        )
        object.__setattr__(self, "film_surfaces", films)
        object.__setattr__(
            self, "initial_inventory_by_surface", MappingProxyType(inventory)
        )
        object.__setattr__(self, "surface_event_templates", templates)
        object.__setattr__(self, "labels", layout.labels)
        object.__setattr__(self, "_initial", layout.initial)
        object.__setattr__(self, "_surface_plan", surface_plan)

    def initial_state(self) -> np.ndarray:
        return self._initial.copy()

    def rhs(
        self,
        values: np.ndarray,
        *,
        surface_rates_m2_s: Mapping[tuple[str, str], float] | None = None,
    ) -> np.ndarray:
        state = validated_runtime_state(values, self._initial)
        derivative = np.zeros_like(state)
        add_surface_rates(derivative, self._surface_plan, surface_rates_m2_s)
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError(
                "experimental runtime accumulator returned a non-finite rate"
            )
        return derivative


__all__ = ["ExperimentalRuntimeAccumulator"]
