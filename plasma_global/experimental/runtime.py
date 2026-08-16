"""Composition facade for opt-in experimental runtime states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global.experimental._runtime_declarations import compile_generic_state
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
    GenericStateAccumulator,
    ProcessContext,
    SurfaceEventRate,
)


def compile_generic_state_accumulator(
    specification: Mapping[str, Any],
    *,
    zone_ids: Sequence[str],
    surface_ids: Sequence[str],
) -> GenericStateAccumulator | None:
    """Compile strict chemistry-declared state/process mappings.

    The canonical experimental mapping uses ``state_variables`` and
    ``processes`` dictionaries. Supported processes deliberately match the
    previous feature set: constant source, linear relaxation, and a surface
    ion-flux-driven source.
    """

    return compile_generic_state(
        specification,
        zone_ids=zone_ids,
        surface_ids=surface_ids,
    )


@dataclass(frozen=True, slots=True)
class ExperimentalRuntimeAccumulator:
    """One independent ODE block for film, inventory, and generic state.

    Film thickness and wall inventory are extensive bookkeeping states, not
    surface site coverages. Surface chemistry remains the sole owner of the
    coverage/site balance.
    """

    generic: GenericStateAccumulator | None = None
    film_surfaces: Sequence[str] = ()
    initial_inventory_by_surface: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    surface_event_templates: Sequence[SurfaceEventRate] = ()
    monolayer_thickness_m: float = 3.0e-10
    labels: tuple[str, ...] = field(init=False)
    lower_bounds: tuple[float | None, ...] = field(init=False)
    upper_bounds: tuple[float | None, ...] = field(init=False)
    _initial: np.ndarray = field(init=False, repr=False)
    _generic_slice: slice = field(init=False, repr=False)
    _surface_plan: SurfaceExecutionPlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        films = normalize_film_surfaces(self.film_surfaces)
        inventory = normalize_inventory(self.initial_inventory_by_surface)
        templates = tuple(self.surface_event_templates)
        event_ids = tuple(event.event_id for event in templates)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("surface-event template IDs must be unique")
        layout = build_runtime_layout(films, inventory, self.generic)
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
        object.__setattr__(self, "lower_bounds", layout.lower_bounds)
        object.__setattr__(self, "upper_bounds", layout.upper_bounds)
        object.__setattr__(self, "_initial", layout.initial)
        object.__setattr__(self, "_generic_slice", layout.generic_slice)
        object.__setattr__(self, "_surface_plan", surface_plan)

    def initial_state(self) -> np.ndarray:
        return self._initial.copy()

    def rhs(
        self,
        values: np.ndarray,
        *,
        drivers: Mapping[str, Mapping[str, float]] | None = None,
        surface_rates_m2_s: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        physical_stop = self._generic_slice.start
        state = validated_runtime_state(values, self._initial, physical_stop)

        derivative = np.zeros_like(state)
        if self.generic is not None:
            derivative[self._generic_slice] = self.generic.rhs(
                state[self._generic_slice], ProcessContext(drivers or {})
            )

        add_surface_rates(derivative, self._surface_plan, surface_rates_m2_s)
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError(
                "experimental runtime accumulator returned a non-finite rate"
            )
        return derivative


__all__ = [
    "ExperimentalRuntimeAccumulator",
    "compile_generic_state_accumulator",
]
