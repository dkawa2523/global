"""Non-power commands in public run steps."""

from __future__ import annotations

from plasma_global.input._schema_base import (
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
)


class GasInletCommand(StrictModel):
    flow_sccm: dict[str, NonNegativeFloat]


class SurfaceCommand(StrictModel):
    temperature_K: PositiveFloat | None = None
