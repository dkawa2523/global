"""Electron, closure, gas-energy, and surface model selections."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    InputPath,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StrictModel,
    _publish_schema_types,
)


class MaxwellianElectronModel(StrictModel):
    kind: Literal["maxwellian"]


class TableElectronModel(StrictModel):
    kind: Literal["table"]
    file: InputPath
    lookup: Literal["mean_energy", "local_field"]
    bounds_policy: Literal["error", "clip"] = "error"
    mobility_reference_neutral_density_m3: PositiveFloat | None = None


class CacheConfig(StrictModel):
    max_entries: PositiveInt = Field(
        default=12,
        description="Maximum exact full-target mole-fraction vectors retained.",
    )


class EnergyGridConfig(StrictModel):
    min_eV: PositiveFloat = 1.0e-3
    max_eV: PositiveFloat = 160.0
    n: Annotated[int, Field(ge=32)] = 360

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.max_eV <= self.min_eV:
            raise ValueError("energy_grid.max_eV must exceed min_eV")
        return self


class ReducedFieldGridConfig(StrictModel):
    # The approximate closure only publishes states whose power-balance root
    # is bracketed. These defaults cover the validated argon examples.
    min_Td: PositiveFloat = 1.0
    max_Td: PositiveFloat = 100.0
    n: Annotated[int, Field(ge=2)] = 48

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.max_Td <= self.min_Td:
            raise ValueError("reduced_field_grid.max_Td must exceed min_Td")
        return self


class ApproximateTwoTermElectronModel(StrictModel):
    kind: Literal["experimental.approximate_two_term"]
    cache: CacheConfig = Field(default_factory=CacheConfig)
    energy_grid: EnergyGridConfig = Field(default_factory=EnergyGridConfig)
    reduced_field_grid: ReducedFieldGridConfig = Field(
        default_factory=ReducedFieldGridConfig
    )
    max_shape_iterations: PositiveInt = 48


ElectronModel = Annotated[
    MaxwellianElectronModel | TableElectronModel | ApproximateTwoTermElectronModel,
    Field(discriminator="kind"),
]


class ElectronEnergyClosure(StrictModel):
    kind: Literal["electron_energy"]


class LocalFieldClosure(StrictModel):
    kind: Literal["local_field"]


ElectronClosure = Annotated[
    ElectronEnergyClosure | LocalFieldClosure,
    Field(discriminator="kind"),
]


class QuasiNeutralElectronDensity(StrictModel):
    kind: Literal["quasineutral"]


class PrescribedElectronDensity(StrictModel):
    kind: Literal["experimental.prescribed_profile"]
    file: InputPath
    zone_columns: dict[str, Identifier] = Field(default_factory=dict)
    interpolation: Literal["linear", "previous"] = "linear"
    hold: Literal["edge", "error"] = "error"


ElectronDensityModel = Annotated[
    QuasiNeutralElectronDensity | PrescribedElectronDensity,
    Field(discriminator="kind"),
]


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


class SurfaceKineticsConfig(StrictModel):
    """Presence enables surface-coverage state and surface reactions."""


class ModelsConfig(StrictModel):
    electrons: ElectronModel
    electron_closure: ElectronClosure
    electron_density: ElectronDensityModel = Field(
        default_factory=lambda: QuasiNeutralElectronDensity(kind="quasineutral")
    )
    gas_energy: GasEnergyModel = Field(
        default_factory=lambda: FixedGasEnergy(kind="fixed")
    )
    surface_kinetics: SurfaceKineticsConfig | None = None


_publish_schema_types(globals())
