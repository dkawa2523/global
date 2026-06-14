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


def merge_provenance(*items: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if isinstance(item, dict):
            out.update(provenance_from_mapping(item))
    return out


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


def chemistry_provenance_summary(mechanism: Any) -> dict[str, Any]:
    reactions = list(getattr(mechanism, 'gas_reactions', [])) + list(getattr(mechanism, 'surface_reactions', []))
    reaction_entries: list[dict[str, Any]] = []
    for reaction in reactions:
        model = getattr(mechanism, 'rate_models', {}).get(reaction.rate_model_key, {})
        provenance = reaction_provenance(reaction, model)
        if not provenance:
            continue
        reaction_entries.append(
            {
                'reaction_id': reaction.reaction_id,
                'phase': reaction.phase,
                'rate_model_key': reaction.rate_model_key,
                **provenance,
            }
        )

    rate_model_entries: list[dict[str, Any]] = []
    for key, model in sorted(getattr(mechanism, 'rate_models', {}).items()):
        provenance = provenance_from_mapping(model)
        if not provenance:
            continue
        rate_model_entries.append(
            {
                'rate_model_key': key,
                'backend': str(model.get('backend', '')),
                **provenance,
            }
        )

    cross_section_entries: list[dict[str, Any]] = []
    for key, spec in sorted(getattr(mechanism, 'cross_sections', {}).items()):
        metadata = getattr(spec, 'metadata', {}) or {}
        top_level = {
            'source': getattr(spec, 'source', None),
            'units': {
                'energy': getattr(spec, 'unit_energy', None),
                'sigma': getattr(spec, 'unit_sigma', None),
            },
        }
        provenance = merge_provenance(top_level, metadata)
        if not provenance:
            continue
        cross_section_entries.append(
            {
                'cross_section_id': key,
                'kind': getattr(spec, 'kind', ''),
                'target_species': getattr(spec, 'target_species', None),
                **provenance,
            }
        )

    return {
        'reaction_count': len(reactions),
        'reactions_with_provenance': len(reaction_entries),
        'reaction_entries': reaction_entries,
        'rate_model_count': len(getattr(mechanism, 'rate_models', {})),
        'rate_models_with_provenance': len(rate_model_entries),
        'rate_model_entries': rate_model_entries,
        'cross_section_count': len(getattr(mechanism, 'cross_sections', {})),
        'cross_sections_with_provenance': len(cross_section_entries),
        'cross_section_entries': cross_section_entries,
    }
