from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from plasma_global.reactor.surface_models import (
    bohm_h_factor,
    effective_ion_loss_frequency_s,
    ion_loss_enabled,
    ion_loss_family,
    ion_loss_uses_effective_frequency,
)


@dataclass(frozen=True)
class ZoneIonLossProperties:
    family: dict[str, str]
    area_m2: dict[str, float]
    h_factor: dict[str, float]
    effective_frequency_s: dict[str, float]


def zone_ion_loss_family(chamber: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for zone in chamber.zones:
        families = {
            ion_loss_family(surface.models)
            for surface in chamber.surfaces_by_zone.get(zone.zone_id, [])
            if ion_loss_enabled(surface.models)
        }
        families.discard('disabled')
        if not families:
            out[zone.zone_id] = 'disabled'
        elif len(families) == 1:
            out[zone.zone_id] = next(iter(families))
        else:
            raise ValueError(
                f'Zone {zone.zone_id} mixes ion-loss model families {sorted(families)}; '
                'choose either Bohm-like or one effective-frequency wall-loss family to avoid double counting.'
            )
    return out


def zone_ion_loss_area(chamber: Any) -> dict[str, float]:
    return {
        z.zone_id: sum(
            s.area_m2
            for s in chamber.surfaces_by_zone.get(z.zone_id, [])
            if ion_loss_enabled(s.models)
        )
        for z in chamber.zones
    }


def zone_bohm_characteristic_length_m(chamber: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for zone in chamber.zones:
        surfaces = [
            surface for surface in chamber.surfaces_by_zone.get(zone.zone_id, [])
            if ion_loss_enabled(surface.models) and ion_loss_family(surface.models) == 'bohm'
        ]
        area = sum(surface.area_m2 for surface in surfaces)
        if area <= 0.0:
            out[zone.zone_id] = 0.0
            continue
        default_length = zone.volume_m3 / max(area, 1.0e-30)
        weighted = 0.0
        for surface in surfaces:
            char_length = float(surface.models.get('characteristic_length_m') or default_length)
            weighted += surface.area_m2 * max(char_length, 1.0e-30)
        out[zone.zone_id] = weighted / area
    return out


def zone_bohm_h_factor(chamber: Any, characteristic_length_m: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for zone in chamber.zones:
        surfaces = [
            surface for surface in chamber.surfaces_by_zone.get(zone.zone_id, [])
            if ion_loss_enabled(surface.models) and ion_loss_family(surface.models) == 'bohm'
        ]
        area = sum(surface.area_m2 for surface in surfaces)
        if area <= 0.0:
            out[zone.zone_id] = 0.0
            continue
        weighted = 0.0
        for surface in surfaces:
            char_length = float(surface.models.get('characteristic_length_m') or characteristic_length_m[zone.zone_id])
            h = bohm_h_factor(
                surface.models,
                pressure_Pa=zone.pressure_Pa,
                gas_temperature_K=zone.gas_temperature_K,
                characteristic_length_m=char_length,
            )
            weighted += surface.area_m2 * h
        out[zone.zone_id] = weighted / area
    return out


def zone_effective_ion_loss_frequency_s(chamber: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    for zone in chamber.zones:
        surfaces = [
            surface for surface in chamber.surfaces_by_zone.get(zone.zone_id, [])
            if ion_loss_enabled(surface.models) and ion_loss_uses_effective_frequency(ion_loss_family(surface.models))
        ]
        area = sum(surface.area_m2 for surface in surfaces)
        if area <= 0.0:
            out[zone.zone_id] = 0.0
            continue
        weighted = 0.0
        for surface in surfaces:
            rate = effective_ion_loss_frequency_s(surface.models, volume_m3=zone.volume_m3, area_m2=surface.area_m2)
            weighted += surface.area_m2 * rate
        out[zone.zone_id] = weighted / area
    return out


def zone_ion_loss_properties(chamber: Any) -> ZoneIonLossProperties:
    characteristic_length = zone_bohm_characteristic_length_m(chamber)
    return ZoneIonLossProperties(
        family=zone_ion_loss_family(chamber),
        area_m2=zone_ion_loss_area(chamber),
        h_factor=zone_bohm_h_factor(chamber, characteristic_length),
        effective_frequency_s=zone_effective_ion_loss_frequency_s(chamber),
    )
