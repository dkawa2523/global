from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plasma_global.reactor.surface_models import (
    bohm_h_factor,
    effective_ion_loss_frequency_s,
    ion_loss_enabled,
    ion_loss_family,
    ion_loss_uses_effective_frequency,
)


class IonLossConfigError(ValueError):
    def __init__(self, code: str, message: str, entity_id: str = '') -> None:
        super().__init__(message)
        self.code = code
        self.entity_id = entity_id


@dataclass(frozen=True)
class ZoneIonLossProperties:
    family: dict[str, str]
    area_m2: dict[str, float]
    h_factor: dict[str, float]
    effective_frequency_s: dict[str, float]


@dataclass(frozen=True)
class _ZoneIonLossSummary:
    family: str
    area_m2: float
    bohm_characteristic_length_m: float
    h_factor: float
    effective_frequency_s: float
    effective_frequency_surface_count: int


def _surface_id(surface: Any) -> str:
    return str(getattr(surface, 'surface_id', 'unknown_surface'))


def _active_ion_loss_surfaces(chamber: Any, zone_id: str) -> list[Any]:
    surfaces = []
    for surface in chamber.surfaces_by_zone.get(zone_id, []):
        try:
            enabled = ion_loss_enabled(surface.models)
        except ValueError as exc:
            raise IonLossConfigError('ION_LOSS_MODEL_UNRECOGNIZED', str(exc), _surface_id(surface)) from exc
        if enabled:
            surfaces.append(surface)
    return surfaces


def _surface_area_m2(surfaces: list[Any]) -> float:
    return sum(surface.area_m2 for surface in surfaces)


def _zone_ion_loss_summary(zone: Any, surfaces: list[Any]) -> _ZoneIonLossSummary:
    if not surfaces:
        return _ZoneIonLossSummary('disabled', 0.0, 0.0, 0.0, 0.0, 0)

    families = {ion_loss_family(surface.models) for surface in surfaces}
    families.discard('disabled')
    if len(families) > 1:
        raise IonLossConfigError(
            'ION_LOSS_MODE_MIXED_IN_ZONE',
            f'Zone {zone.zone_id} mixes ion-loss model families {sorted(families)}; '
            'choose either Bohm-like or one effective-frequency wall-loss family to avoid double counting.',
            str(zone.zone_id),
        )
    family = next(iter(families), 'disabled')
    area = _surface_area_m2(surfaces)
    if family == 'bohm':
        characteristic_length = _bohm_characteristic_length_m(zone, surfaces)
        h_factor = _bohm_h_factor(zone, surfaces, characteristic_length)
        return _ZoneIonLossSummary(family, area, characteristic_length, h_factor, 0.0, 0)
    if ion_loss_uses_effective_frequency(family):
        rate = _effective_frequency_s(zone, surfaces)
        return _ZoneIonLossSummary(family, area, 0.0, 0.0, rate, len(surfaces))
    return _ZoneIonLossSummary('disabled', area, 0.0, 0.0, 0.0, 0)


def _bohm_characteristic_length_m(zone: Any, surfaces: list[Any]) -> float:
    area = _surface_area_m2(surfaces)
    if area <= 0.0:
        return 0.0
    default_length = zone.volume_m3 / max(area, 1.0e-30)
    weighted = 0.0
    for surface in surfaces:
        try:
            char_length = float(surface.models.get('characteristic_length_m') or default_length)
        except Exception as exc:
            raise IonLossConfigError(
                'BOHM_H_FACTOR_INVALID',
                f'Bohm ion loss characteristic_length_m for {_surface_id(surface)} is invalid: {exc}',
                _surface_id(surface),
            ) from exc
        weighted += surface.area_m2 * max(char_length, 1.0e-30)
    return weighted / area


def _bohm_h_factor(zone: Any, surfaces: list[Any], characteristic_length_m: float) -> float:
    area = _surface_area_m2(surfaces)
    if area <= 0.0:
        return 0.0
    weighted = 0.0
    for surface in surfaces:
        try:
            char_length = float(surface.models.get('characteristic_length_m') or characteristic_length_m)
            h = bohm_h_factor(
                surface.models,
                pressure_Pa=zone.pressure_Pa,
                gas_temperature_K=zone.gas_temperature_K,
                characteristic_length_m=char_length,
            )
        except Exception as exc:
            raise IonLossConfigError(
                'BOHM_H_FACTOR_INVALID',
                f'Bohm ion loss h_factor for {_surface_id(surface)} is invalid: {exc}',
                _surface_id(surface),
            ) from exc
        weighted += surface.area_m2 * h
    return weighted / area


def _effective_frequency_s(zone: Any, surfaces: list[Any]) -> float:
    area = _surface_area_m2(surfaces)
    if area <= 0.0:
        return 0.0
    weighted = 0.0
    for surface in surfaces:
        try:
            rate = effective_ion_loss_frequency_s(surface.models, volume_m3=zone.volume_m3, area_m2=surface.area_m2)
        except Exception as exc:
            family = ion_loss_family(surface.models)
            code = 'AMBIPOLAR_LOSS_CONFIG_INVALID' if family == 'ambipolar_diffusion' else 'ION_LOSS_FREQUENCY_CONFIG_INVALID'
            raise IonLossConfigError(
                code,
                f'effective-frequency ion loss for {_surface_id(surface)} is invalid: {exc}',
                _surface_id(surface),
            ) from exc
        weighted += surface.area_m2 * rate
    return weighted / area


def _zone_ion_loss_summaries(chamber: Any) -> dict[str, _ZoneIonLossSummary]:
    return {
        zone.zone_id: _zone_ion_loss_summary(zone, _active_ion_loss_surfaces(chamber, zone.zone_id))
        for zone in chamber.zones
    }


def zone_effective_frequency_surface_counts(chamber: Any) -> dict[str, int]:
    return {
        zone_id: summary.effective_frequency_surface_count
        for zone_id, summary in _zone_ion_loss_summaries(chamber).items()
    }


def zone_ion_loss_properties(chamber: Any) -> ZoneIonLossProperties:
    summaries = _zone_ion_loss_summaries(chamber)
    return ZoneIonLossProperties(
        family={zone_id: summary.family for zone_id, summary in summaries.items()},
        area_m2={zone_id: summary.area_m2 for zone_id, summary in summaries.items()},
        h_factor={zone_id: summary.h_factor for zone_id, summary in summaries.items()},
        effective_frequency_s={zone_id: summary.effective_frequency_s for zone_id, summary in summaries.items()},
    )
