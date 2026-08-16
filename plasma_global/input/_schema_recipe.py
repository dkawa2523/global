"""Recipe timing and non-power command input models."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
    _publish_schema_types,
    _require_unique,
)
from plasma_global.input._schema_power import PowerPortCommand


class GasInletCommand(StrictModel):
    flow_sccm: dict[str, NonNegativeFloat]


class SurfaceCommand(StrictModel):
    temperature_K: PositiveFloat | None = None


class RecipeCommands(StrictModel):
    gas_inlets: dict[str, GasInletCommand] = Field(default_factory=dict)
    power_ports: dict[str, PowerPortCommand] = Field(default_factory=dict)
    surfaces: dict[str, SurfaceCommand] = Field(default_factory=dict)


class RecipeStepConfig(StrictModel):
    step_id: Identifier
    duration_s: PositiveFloat
    commands: RecipeCommands = Field(default_factory=RecipeCommands)


class RecipeConfig(StrictModel):
    recipe_id: Identifier
    description: str = ""
    start_time_s: NonNegativeFloat = 0.0
    steps: list[RecipeStepConfig]

    @model_validator(mode="after")
    def valid_steps(self) -> Self:
        if not self.steps:
            raise ValueError("recipe.steps must contain at least one step")
        _require_unique(self.steps, "step_id", "step_id")
        return self

    @property
    def duration_s(self) -> float:
        return sum(step.duration_s for step in self.steps)

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s

    def step_bounds(self) -> list[tuple[float, float]]:
        start = self.start_time_s
        bounds: list[tuple[float, float]] = []
        for step in self.steps:
            end = start + step.duration_s
            bounds.append((start, end))
            start = end
        return bounds


_publish_schema_types(globals())
