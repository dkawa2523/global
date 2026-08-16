"""Reactor topology, boundary, and flow input models."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from plasma_global.input._schema_base import (
    Identifier,
    NonNegativeFloat,
    PositiveFloat,
    StrictModel,
    _publish_schema_types,
    _require_unique,
)
from plasma_global.input._schema_power import (
    ExperimentalCCPModel,
    PowerPortConfig,
)


def _validate_initial_densities(
    densities: dict[str, float], seeds: dict[str, float]
) -> None:
    if seeds:
        raise ValueError(
            "initial_seed_densities_m3 is only valid with initial_mole_fractions"
        )
    if not densities or not any(value > 0.0 for value in densities.values()):
        raise ValueError("initial_densities_m3 must contain a positive density")


def _validate_initial_fractions(fractions: dict[str, float] | None) -> None:
    if not fractions or not any(value > 0.0 for value in fractions.values()):
        raise ValueError("initial_mole_fractions must contain a positive fraction")
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"initial_mole_fractions must sum to one, got {total:g}")


class ZoneConfig(StrictModel):
    zone_id: Identifier
    description: str = ""
    volume_m3: PositiveFloat
    pressure_Pa: PositiveFloat
    gas_temperature_K: PositiveFloat
    initial_densities_m3: dict[str, NonNegativeFloat] | None = None
    initial_mole_fractions: dict[str, NonNegativeFloat] | None = None
    initial_seed_densities_m3: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    initial_mean_energy_eV: PositiveFloat | None = None

    @model_validator(mode="after")
    def has_initial_composition(self) -> Self:
        densities = self.initial_densities_m3
        fractions = self.initial_mole_fractions
        if (densities is None) == (fractions is None):
            raise ValueError(
                "set exactly one of initial_densities_m3 or initial_mole_fractions"
            )
        if densities is not None:
            _validate_initial_densities(densities, self.initial_seed_densities_m3)
        else:
            _validate_initial_fractions(fractions)
        return self


class EdgeConfig(StrictModel):
    edge_id: Identifier
    from_zone: Identifier
    to_zone: Identifier
    conductance_m3_s: NonNegativeFloat

    @model_validator(mode="after")
    def distinct_zones(self) -> Self:
        if self.from_zone == self.to_zone:
            raise ValueError("an edge must connect two different zones")
        return self


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
    diffusion_length_m: PositiveFloat | None = None


class OffWallTransport(StrictModel):
    kind: Literal["off"]


WallTransport = Annotated[
    BohmWallTransport
    | PrescribedFrequencyWallTransport
    | AmbipolarWallTransport
    | OffWallTransport,
    Field(discriminator="kind"),
]


class SurfaceConfig(StrictModel):
    surface_id: Identifier
    zone_id: Identifier
    area_m2: PositiveFloat
    temperature_K: PositiveFloat
    ion_impact_energy_eV: NonNegativeFloat = 0.0
    site_density_m2: NonNegativeFloat = 0.0
    initial_coverages: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    wall_transport: WallTransport


class GasInletConfig(StrictModel):
    inlet_id: Identifier
    zone_id: Identifier
    flow_sccm: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    temperature_K: PositiveFloat


class PumpConfig(StrictModel):
    pump_id: Identifier
    zone_id: Identifier
    speed_m3_s: NonNegativeFloat


def _power_coupling_target_error(
    port: PowerPortConfig,
    surface_zone_by_id: dict[str, str],
) -> str | None:
    """Return the one reactor-level contract for an optional power target."""

    target = port.coupling_target
    if not target:
        return None
    if not isinstance(port.model, ExperimentalCCPModel):
        return (
            f"power port {port.port_id!r} model {port.model.kind!r} "
            "does not support coupling_target"
        )
    target_zone = surface_zone_by_id.get(target)
    if target_zone is None:
        return (
            f"experimental CCP power port {port.port_id!r} coupling_target "
            f"{target!r} must reference a reactor surface"
        )
    if target_zone != port.zone_id:
        return (
            f"experimental CCP power port {port.port_id!r} coupling_target "
            f"{target!r} belongs to zone {target_zone!r}, not port zone "
            f"{port.zone_id!r}"
        )
    return None


class ReactorConfig(StrictModel):
    chamber_id: Identifier
    description: str = ""
    zones: list[ZoneConfig]
    edges: list[EdgeConfig] = Field(default_factory=list)
    surfaces: list[SurfaceConfig] = Field(default_factory=list)
    gas_inlets: list[GasInletConfig] = Field(default_factory=list)
    pumps: list[PumpConfig] = Field(default_factory=list)
    power_ports: list[PowerPortConfig] = Field(default_factory=list)

    def _validate_unique_ids(self) -> None:
        for items, attribute, label in (
            (self.zones, "zone_id", "zone_id"),
            (self.edges, "edge_id", "edge_id"),
            (self.surfaces, "surface_id", "surface_id"),
            (self.gas_inlets, "inlet_id", "inlet_id"),
            (self.pumps, "pump_id", "pump_id"),
            (self.power_ports, "port_id", "port_id"),
        ):
            _require_unique(items, attribute, label)

    def _validate_zone_references(self, zone_ids: set[str]) -> None:
        for edge in self.edges:
            for field_name, zone_id in (
                ("from_zone", edge.from_zone),
                ("to_zone", edge.to_zone),
            ):
                if zone_id not in zone_ids:
                    raise ValueError(
                        f"edge {edge.edge_id!r} {field_name} references unknown "
                        f"zone {zone_id!r}"
                    )
        for label, entities, id_field in (
            ("surface", self.surfaces, "surface_id"),
            ("gas inlet", self.gas_inlets, "inlet_id"),
            ("pump", self.pumps, "pump_id"),
            ("power port", self.power_ports, "port_id"),
        ):
            for entity in entities:
                if entity.zone_id not in zone_ids:
                    raise ValueError(
                        f"{label} {getattr(entity, id_field)!r} "
                        f"references unknown zone {entity.zone_id!r}"
                    )

    def _validate_coupling_targets(self) -> None:
        surface_zone_by_id = {
            surface.surface_id: surface.zone_id for surface in self.surfaces
        }
        for port in self.power_ports:
            if error := _power_coupling_target_error(port, surface_zone_by_id):
                raise ValueError(error)

    @model_validator(mode="after")
    def valid_topology(self) -> Self:
        if not self.zones:
            raise ValueError("reactor.zones must contain at least one zone")
        self._validate_unique_ids()
        zone_ids = {zone.zone_id for zone in self.zones}
        self._validate_zone_references(zone_ids)
        self._validate_coupling_targets()
        return self


_publish_schema_types(globals())
