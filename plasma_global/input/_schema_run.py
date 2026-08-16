"""Solver, output, and opt-in experimental run controls."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Literal, Self

from pydantic import Field, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
    _publish_schema_types,
)


def _validate_save_times(save_at: list[float] | None) -> None:
    if save_at is None:
        return
    if not save_at:
        raise ValueError("solver.save_at_s must not be empty")
    if any(right <= left for left, right in pairwise(save_at)):
        raise ValueError("solver.save_at_s must be strictly increasing")


def _validate_step_order(first_step: float | None, max_step: float | None) -> None:
    if first_step is not None and max_step is not None and first_step > max_step:
        raise ValueError("solver.first_step_s must not exceed solver.max_step_s")


class SolverConfig(StrictModel):
    method: Literal["BDF"] = "BDF"
    rtol: float = Field(
        default=1.0e-6,
        ge=100.0 * math.ulp(1.0),
        lt=1.0,
    )
    atol: PositiveFloat = Field(
        default=1.0e-14,
        description="Dimensionless absolute tolerance applied after state scaling.",
    )
    first_step_s: PositiveFloat | None = None
    max_step_s: PositiveFloat | None = None
    sample_interval_s: PositiveFloat | None = None
    save_at_s: list[NonNegativeFloat] | None = None

    @model_validator(mode="after")
    def one_sampling_mode(self) -> Self:
        if self.sample_interval_s is not None and self.save_at_s is not None:
            raise ValueError(
                "set at most one of solver.sample_interval_s or solver.save_at_s"
            )
        _validate_save_times(self.save_at_s)
        _validate_step_order(self.first_step_s, self.max_step_s)
        return self


class OutputConfig(StrictModel):
    observables: list[str] = Field(default_factory=list)
    summary_series: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_series_selection(self) -> Self:
        for field_name, values in (
            ("observables", self.observables),
            ("summary_series", self.summary_series),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"output.{field_name} must not contain empty names")
            if len(set(values)) != len(values):
                raise ValueError(f"output.{field_name} must not contain duplicates")
        return self


class ExperimentalFilmConfig(StrictModel):
    """Presence enables chemistry-declared film state."""

    monolayer_thickness_m: PositiveFloat = 3.0e-10


class ExperimentalInventoryEventConfig(StrictModel):
    reaction_id: Identifier
    surface_id: Identifier
    inventory_particles_per_event: dict[Identifier, float]

    @model_validator(mode="after")
    def nonzero_named_yields(self) -> Self:
        if not self.inventory_particles_per_event:
            raise ValueError("inventory event yield mapping must not be empty")
        if any(value == 0.0 for value in self.inventory_particles_per_event.values()):
            raise ValueError("inventory event yields must be explicitly nonzero")
        return self


class ExperimentalWallInventoryConfig(StrictModel):
    """Presence enables chemistry-declared wall inventories."""

    initial_by_surface: dict[str, dict[str, NonNegativeFloat]] = Field(
        default_factory=dict
    )
    events: list[ExperimentalInventoryEventConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_events(self) -> Self:
        pairs = [(item.reaction_id, item.surface_id) for item in self.events]
        if len(set(pairs)) != len(pairs):
            raise ValueError("wall inventory reaction/surface events must be unique")
        return self


class ExperimentalExtensionsConfig(StrictModel):
    """Presence enables experimental chemistry extension state and terms."""


class ExperimentalStopWhenQuasiSteadyConfig(StrictModel):
    """Legacy-named nonterminal quasi-steady observation, not a root solve."""

    relative_rhs_norm_s_inv: NonNegativeFloat = 1.0e-3
    min_time_s: NonNegativeFloat = 0.0


class ExperimentalConfig(StrictModel):
    film: ExperimentalFilmConfig | None = None
    wall_inventory: ExperimentalWallInventoryConfig | None = None
    extensions: ExperimentalExtensionsConfig | None = None
    stop_when_quasi_steady: ExperimentalStopWhenQuasiSteadyConfig | None = None

    @model_validator(mode="after")
    def enables_at_least_one_feature(self) -> Self:
        if (
            self.film is None
            and self.wall_inventory is None
            and self.extensions is None
            and self.stop_when_quasi_steady is None
        ):
            raise ValueError("experimental must enable at least one explicit feature")
        return self


_publish_schema_types(globals())
