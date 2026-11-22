"""Shared quasineutral and gas-energy choices in the public declaration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from plasma_global.input._schema_base import NonNegativeFloat, StrictModel


class QuasiNeutralElectronDensity(StrictModel):
    kind: Literal["quasineutral"]


class FixedGasEnergy(StrictModel):
    kind: Literal["fixed"]


class EvolvedGasEnergy(StrictModel):
    kind: Literal["evolved"]
    wall_energy_relaxation_s_inv_by_zone: dict[str, NonNegativeFloat] = Field(
        default_factory=dict
    )


GasEnergyModel = Annotated[
    FixedGasEnergy | EvolvedGasEnergy,
    Field(discriminator="kind"),
]
