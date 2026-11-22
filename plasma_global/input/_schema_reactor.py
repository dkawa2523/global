"""Wall transport choices shared by public surface declarations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from plasma_global.input._schema_base import (
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
)


class BohmWallTransport(StrictModel):
    kind: Literal["bohm"]
    characteristic_length_m: PositiveFloat | None = None
    h_factor: Literal["auto"] | Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    min_h_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 0.02
    max_h_factor: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    ion_neutral_cross_section_m2: PositiveFloat = 1.0e-18

    @model_validator(mode="after")
    def ordered_h_factors(self) -> Self:
        if self.max_h_factor < self.min_h_factor:
            raise ValueError("max_h_factor must be at least min_h_factor")
        return self


class PrescribedFrequencyWallTransport(StrictModel):
    kind: Literal["prescribed_frequency"]
    frequency_s_inv: NonNegativeFloat


class AmbipolarWallTransport(StrictModel):
    kind: Literal["ambipolar"]
    diffusion_coefficient_m2_s: PositiveFloat
    diffusion_length_m: PositiveFloat
    electron_temperature_reference_eV: PositiveFloat | None = None


class OffWallTransport(StrictModel):
    kind: Literal["off"]


WallTransport = Annotated[
    BohmWallTransport
    | PrescribedFrequencyWallTransport
    | AmbipolarWallTransport
    | OffWallTransport,
    Field(discriminator="kind"),
]
