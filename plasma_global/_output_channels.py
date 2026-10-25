"""Reaction frequencies on explicit volume and surface measures."""

from __future__ import annotations

from dataclasses import dataclass

from plasma_global.core.compiled import CompiledGlobalModel, ModelEvaluation


@dataclass(frozen=True, slots=True)
class EventChannel:
    domain: str
    region: str
    zone: str
    reaction: str
    measure: float

    @property
    def rate_name(self) -> str:
        unit = "m3_s" if self.domain == "gas" else "m2_s"
        return f"{self.domain}_reaction_rate_{unit}[{self.region},{self.reaction}]"

    @property
    def frequency_name(self) -> str:
        return f"{self.domain}_event_rate_s_inv[{self.region},{self.reaction}]"

    @property
    def count_name(self) -> str:
        return f"{self.domain}_event_count[{self.region},{self.reaction}]"

    def rate(self, evaluated: ModelEvaluation) -> float:
        ledger = evaluated.ledger_by_zone[self.zone]
        if self.domain == "gas":
            return ledger.reaction_rates_m3_s[self.reaction]
        if self.domain == "surface":
            return ledger.surface_rates_m2_s[f"{self.reaction}@{self.region}"]
        # Boundary branch rates are volume-normalized, not incident fluxes.
        return sum(
            record.branch_rates_m3_s.get(self.reaction, 0.0)
            for record in ledger.wall_fluxes
            if record.surface_id == self.region
        )


def _gas_channels(model: CompiledGlobalModel) -> list[EventChannel]:
    return [
        EventChannel("gas", zone.zone_id, zone.zone_id, reaction, zone.volume_m3)
        for zone in model.zones
        for index, reaction in enumerate(model.reaction_ids)
        if not model.reaction_zones[index]
        or zone.zone_id in model.reaction_zones[index]
    ]


def _boundary_channels(model: CompiledGlobalModel) -> list[EventChannel]:
    return [
        EventChannel(
            "boundary",
            str(wall.surface_id),
            wall.zone_id,
            reaction.reaction_id,
            wall.area_m2,
        )
        for wall in model.wall_boundaries
        if wall.surface_id is not None
        for reaction in wall.reactions
    ]


def _surface_channels(model: CompiledGlobalModel) -> list[EventChannel]:
    if model.surface_model is None:
        return []
    return [
        EventChannel(
            "surface", surface.surface_id, surface.zone_id, reaction.id, surface.area_m2
        )
        for surface in model.surface_model.surfaces
        for reaction in model.surface_model.chemistry.surface_reactions
        if (not reaction.zones or surface.zone_id in reaction.zones)
        and (not reaction.surfaces or surface.surface_id in reaction.surfaces)
    ]


def event_channels(model: CompiledGlobalModel) -> tuple[EventChannel, ...]:
    return tuple(
        _gas_channels(model) + _boundary_channels(model) + _surface_channels(model)
    )


def channel_rates(
    model: CompiledGlobalModel,
    evaluated: ModelEvaluation,
    channels: tuple[EventChannel, ...],
) -> dict[str, float]:
    volumes = {zone.zone_id: zone.volume_m3 for zone in model.zones}
    values: dict[str, float] = {}
    for channel in channels:
        rate = channel.rate(evaluated)
        if channel.domain == "boundary":
            frequency = rate * volumes[channel.zone]
            rate = frequency / channel.measure
        else:
            frequency = rate * channel.measure
        values[channel.rate_name] = rate
        values[channel.frequency_name] = frequency
    return values
