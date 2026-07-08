"""Structural validation for chamber and recipe objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from plasma_global.validation import ValidationIssue


_IdRule = tuple[str, str, str, str]
_ValueRule = tuple[str, str, str, str, str, Callable[[float], bool]]
_MapValueRule = tuple[str, str, str, str, str, str, Callable[[float], bool]]
_ReferenceRule = tuple[str, str, str, str, str, str]


def _positive(value: float) -> bool:
    return float(value) > 0.0


def _nonnegative(value: float) -> bool:
    return float(value) >= 0.0


ID_RULES: tuple[_IdRule, ...] = (
    ('zones', 'zone_id', 'DUPLICATE_ZONE_ID', 'zone'),
    ('edges', 'edge_id', 'DUPLICATE_EDGE_ID', 'edge'),
    ('surfaces', 'surface_id', 'DUPLICATE_SURFACE_ID', 'surface'),
    ('gas_inlets', 'inlet_id', 'DUPLICATE_INLET_ID', 'gas inlet'),
    ('pumps', 'pump_id', 'DUPLICATE_PUMP_ID', 'pump'),
    ('power_ports', 'port_id', 'DUPLICATE_POWER_PORT_ID', 'power port'),
)

VALUE_RULES: tuple[_ValueRule, ...] = (
    ('zones', 'volume_m3', 'zone_id', 'ZONE_VOLUME_INVALID', 'zone volume_m3 must be positive', _positive),
    ('zones', 'pressure_Pa', 'zone_id', 'ZONE_PRESSURE_INVALID', 'zone pressure_Pa must be non-negative', _nonnegative),
    ('zones', 'gas_temperature_K', 'zone_id', 'ZONE_TEMPERATURE_INVALID', 'zone gas_temperature_K must be positive', _positive),
    ('edges', 'conductance_m3_s', 'edge_id', 'EDGE_CONDUCTANCE_INVALID', 'edge conductance_m3_s must be non-negative', _nonnegative),
    ('surfaces', 'area_m2', 'surface_id', 'SURFACE_AREA_INVALID', 'surface area_m2 must be positive', _positive),
    ('surfaces', 'temperature_K', 'surface_id', 'SURFACE_TEMPERATURE_INVALID', 'surface temperature_K must be positive', _positive),
    ('surfaces', 'site_density_m2', 'surface_id', 'SURFACE_SITE_DENSITY_INVALID', 'surface site_density_m2 must be positive', _positive),
    ('gas_inlets', 'temperature_K', 'inlet_id', 'INLET_TEMPERATURE_INVALID', 'inlet temperature_K must be positive', _positive),
    ('pumps', 'speed_m3_s', 'pump_id', 'PUMP_SPEED_INVALID', 'pump speed_m3_s must be non-negative', _nonnegative),
)

OPTIONAL_VALUE_RULES: tuple[_ValueRule, ...] = ()

MAP_VALUE_RULES: tuple[_MapValueRule, ...] = (
    ('zones', 'initial_densities_m3', 'zone_id', 'INITIAL_DENSITY_INVALID', 'initial density must be non-negative', ':', _nonnegative),
    ('surfaces', 'initial_coverages', 'surface_id', 'SURFACE_COVERAGE_INVALID', 'initial coverage must be non-negative', ':', _nonnegative),
    ('surfaces', 'initial_inventory', 'surface_id', 'SURFACE_INVENTORY_INVALID', 'initial inventory must be non-negative', ':', _nonnegative),
    ('gas_inlets', 'flow_sccm', 'inlet_id', 'INLET_FLOW_INVALID', 'inlet flow_sccm must be non-negative', ':', _nonnegative),
)

REFERENCE_RULES: tuple[_ReferenceRule, ...] = (
    ('edges', 'from_zone', 'edge_id', 'UNKNOWN_EDGE_ZONE', 'edge {item_id} from_zone is unknown', 'zone'),
    ('edges', 'to_zone', 'edge_id', 'UNKNOWN_EDGE_ZONE', 'edge {item_id} to_zone is unknown', 'zone'),
    ('surfaces', 'zone_id', 'surface_id', 'UNKNOWN_SURFACE_ZONE', 'surface {item_id} zone_id is unknown', 'zone'),
    ('gas_inlets', 'zone_id', 'inlet_id', 'UNKNOWN_INLET_ZONE', 'inlet {item_id} zone_id is unknown', 'zone'),
    ('pumps', 'zone_id', 'pump_id', 'UNKNOWN_PUMP_ZONE', 'pump {item_id} zone_id is unknown', 'zone'),
    ('power_ports', 'zone_id', 'port_id', 'UNKNOWN_POWER_PORT_ZONE', 'power port {item_id} zone_id is unknown', 'zone'),
)


def validate_reactor_config(chamber: Any, recipe: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    _validate_ids(chamber, issues)
    _validate_chamber_values(chamber, issues)
    _validate_references(chamber, issues)
    _validate_recipe(recipe, chamber, issues)
    return issues


def _items(owner: Any, attr: str) -> Iterable[Any]:
    return getattr(owner, attr, []) or []


def _item_id(item: Any, attr: str) -> str:
    return str(getattr(item, attr, ''))


def _validate_ids(chamber: Any, issues: list[ValidationIssue]) -> None:
    for collection, id_attr, code, label in ID_RULES:
        _check_unique((_item_id(item, id_attr) for item in _items(chamber, collection)), code, label, issues)


def _check_unique(ids: Iterable[str], code: str, label: str, issues: list[ValidationIssue]) -> None:
    seen: set[str] = set()
    for item_id in ids:
        if not item_id:
            issues.append(ValidationIssue('ERROR', f'{code}_MISSING', f'{label} id is missing.'))
            continue
        if item_id in seen:
            issues.append(ValidationIssue('ERROR', code, f'Duplicate {label} id {item_id!r}.', item_id))
        seen.add(item_id)


def _validate_chamber_values(chamber: Any, issues: list[ValidationIssue]) -> None:
    if not getattr(chamber, 'zones', []):
        issues.append(ValidationIssue('ERROR', 'NO_ZONES', 'Chamber must define at least one zone.'))
    for rule in VALUE_RULES:
        _check_value_rule(chamber, rule, issues)
    for rule in OPTIONAL_VALUE_RULES:
        _check_value_rule(chamber, rule, issues, skip_none=True)
    for rule in MAP_VALUE_RULES:
        _check_map_value_rule(chamber, rule, issues)


def _check_value_rule(
    chamber: Any,
    rule: _ValueRule,
    issues: list[ValidationIssue],
    *,
    skip_none: bool = False,
) -> None:
    collection, value_attr, id_attr, code, message, predicate = rule
    for item in _items(chamber, collection):
        value = getattr(item, value_attr, None)
        if value is None and skip_none:
            continue
        if not _passes(value, predicate):
            issues.append(ValidationIssue('ERROR', code, message, _item_id(item, id_attr)))


def _check_map_value_rule(chamber: Any, rule: _MapValueRule, issues: list[ValidationIssue]) -> None:
    collection, map_attr, id_attr, code, message, separator, predicate = rule
    for item in _items(chamber, collection):
        owner_id = _item_id(item, id_attr)
        for key, value in (getattr(item, map_attr, {}) or {}).items():
            if not _passes(value, predicate):
                issues.append(ValidationIssue('ERROR', code, message, f'{owner_id}{separator}{key}'))


def _passes(value: Any, predicate: Callable[[float], bool]) -> bool:
    try:
        return predicate(float(value))
    except (TypeError, ValueError):
        return False


def _validate_references(chamber: Any, issues: list[ValidationIssue]) -> None:
    known = {'zone': {zone.zone_id for zone in _items(chamber, 'zones')}}
    for collection, ref_attr, id_attr, code, message, known_key in REFERENCE_RULES:
        for item in _items(chamber, collection):
            value = str(getattr(item, ref_attr, ''))
            item_id = _item_id(item, id_attr)
            if value not in known[known_key]:
                issues.append(ValidationIssue('ERROR', code, message.format(item_id=item_id), item_id))


def _validate_recipe(recipe: Any, chamber: Any, issues: list[ValidationIssue]) -> None:
    steps = list(getattr(recipe, 'steps', []) or [])
    if not steps:
        issues.append(ValidationIssue('ERROR', 'NO_RECIPE_STEPS', 'Recipe must define at least one step.'))
        return

    _check_unique((_item_id(step, 'step_id') for step in steps), 'DUPLICATE_RECIPE_STEP_ID', 'recipe step', issues)
    _validate_step_times(steps, issues)
    _validate_step_references(steps, chamber, issues)


def _validate_step_times(steps: list[Any], issues: list[ValidationIssue]) -> None:
    previous_end = None
    for step in steps:
        step_id = _item_id(step, 'step_id')
        start = float(getattr(step, 't_start_s', 0.0))
        end = float(getattr(step, 't_end_s', 0.0))
        if end <= start:
            issues.append(
                ValidationIssue(
                    'ERROR',
                    'RECIPE_STEP_TIME_INVALID',
                    'recipe step t_end_s must be greater than t_start_s',
                    step_id,
                )
            )
        tolerance = max(1.0e-30, 1.0e-12 * max(abs(start), abs(previous_end or 0.0), 1.0))
        if previous_end is not None and start < previous_end - tolerance:
            issues.append(
                ValidationIssue(
                    'ERROR',
                    'RECIPE_STEP_OVERLAP',
                    'recipe steps must be non-overlapping and ordered by time',
                    step_id,
                )
            )
        if previous_end is not None and start > previous_end + tolerance:
            issues.append(
                ValidationIssue(
                    'ERROR',
                    'RECIPE_STEP_GAP',
                    'recipe steps must be contiguous; insert an explicit step for holds or idle periods',
                    step_id,
                )
            )
        previous_end = end


def _validate_step_references(steps: list[Any], chamber: Any, issues: list[ValidationIssue]) -> None:
    reference_rules = (
        ('gas_inlets', {inlet.inlet_id for inlet in _items(chamber, 'gas_inlets')}, 'UNKNOWN_STEP_INLET', 'inlet'),
        ('power_ports', {port.port_id for port in _items(chamber, 'power_ports')}, 'UNKNOWN_STEP_POWER_PORT', 'power port'),
        ('surface_overrides', {surface.surface_id for surface in _items(chamber, 'surfaces')}, 'UNKNOWN_STEP_SURFACE', 'surface'),
    )
    for step in steps:
        step_id = _item_id(step, 'step_id')
        for attr, known_ids, code, label in reference_rules:
            for target_id in getattr(step, attr, {}) or {}:
                if target_id not in known_ids:
                    issues.append(
                        ValidationIssue(
                            'ERROR',
                            code,
                            f'recipe step {step_id} references unknown {label}',
                            target_id,
                        )
                    )


__all__ = ['validate_reactor_config']
