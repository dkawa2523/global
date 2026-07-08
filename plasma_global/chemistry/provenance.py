from __future__ import annotations

from typing import Any


PROVENANCE_FIELDS = (
    'source',
    'reference',
    'version',
    'units',
    'valid_temperature_range',
    'valid_pressure_range',
    'uncertainty',
    'notes',
    'cross_section_id',
)

RANGE_PROVENANCE_FIELDS = {'valid_temperature_range', 'valid_pressure_range'}


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _plain_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain_value(v) for v in value]
    if isinstance(value, list):
        return [_plain_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain_value(v) for k, v in value.items() if not _is_empty(v)}
    return value


def provenance_from_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, Any] = {}
    nested = mapping.get('provenance')
    if isinstance(nested, dict):
        out.update(_plain_value(nested))
    elif not _is_empty(nested):
        out['_invalid_provenance'] = nested
    if '_invalid_provenance' in mapping and not _is_empty(mapping.get('_invalid_provenance')):
        out['_invalid_provenance'] = mapping['_invalid_provenance']
    for key in PROVENANCE_FIELDS:
        if key in mapping and not _is_empty(mapping.get(key)):
            out[key] = _plain_value(mapping[key])
    if 'notes' not in out and not _is_empty(mapping.get('note')):
        out['notes'] = _plain_value(mapping['note'])
    return {str(k): v for k, v in out.items() if not _is_empty(v)}


def validate_provenance(report: Any, raw: Any, entity_kind: str, entity_id: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        report.add(
            'WARNING',
            'PROVENANCE_FORMAT_INVALID',
            f'{entity_kind} {entity_id} provenance should be a mapping',
            entity_id,
        )
        return {}
    provenance = provenance_from_mapping(raw)
    if '_invalid_provenance' in provenance:
        report.add(
            'WARNING',
            'PROVENANCE_FORMAT_INVALID',
            f'{entity_kind} {entity_id} provenance should be a mapping',
            entity_id,
        )
    for field_name in RANGE_PROVENANCE_FIELDS:
        value = provenance.get(field_name)
        if isinstance(value, (list, tuple)) and len(value) != 2:
            report.add(
                'WARNING',
                'PROVENANCE_RANGE_INVALID',
                f'{entity_kind} {entity_id} {field_name} should have two entries when given as a list',
                entity_id,
            )
    return provenance


def reaction_provenance(reaction: Any, rate_model: dict[str, Any] | None = None) -> dict[str, Any]:
    out = provenance_from_mapping(getattr(reaction, 'provenance', {}) or {})
    if getattr(reaction, 'notes', '') and out:
        out.setdefault('notes', getattr(reaction, 'notes'))
    if rate_model:
        model_prov = provenance_from_mapping(rate_model)
        for key, value in model_prov.items():
            out.setdefault(key, value)
        if str(rate_model.get('backend', '')).lower() == 'electron_impact_xsec':
            out.setdefault('cross_section_id', rate_model.get('cross_section_id'))
    return {str(k): v for k, v in out.items() if not _is_empty(v)}


def _cross_section_has_provenance(spec: Any) -> bool:
    metadata = getattr(spec, 'metadata', {}) or {}
    return bool(provenance_from_mapping(metadata) or not _is_empty(getattr(spec, 'source', None)))


def chemistry_provenance_summary(mechanism: Any) -> dict[str, Any]:
    reactions = list(getattr(mechanism, 'gas_reactions', [])) + list(getattr(mechanism, 'surface_reactions', []))
    rate_models = getattr(mechanism, 'rate_models', {})
    cross_sections = getattr(mechanism, 'cross_sections', {})
    return {
        'reaction_count': len(reactions),
        'reactions_with_provenance': sum(
            1
            for reaction in reactions
            if reaction_provenance(reaction, rate_models.get(reaction.rate_model_key, {}))
        ),
        'rate_model_count': len(rate_models),
        'rate_models_with_provenance': sum(
            1
            for model in rate_models.values()
            if provenance_from_mapping(model)
        ),
        'cross_section_count': len(cross_sections),
        'cross_sections_with_provenance': sum(
            1
            for spec in cross_sections.values()
            if _cross_section_has_provenance(spec)
        ),
    }
