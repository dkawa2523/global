from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from plasma_global.chemistry.cross_sections import load_cross_sections_manifest
from plasma_global.chemistry.models import MechanismBundle, Reaction, Species
from plasma_global.chemistry.parser import parse_csv_bool, parse_equation, parse_pipe_list, parse_semicolon_map


def _load_species(path: Path) -> list[Species]:
    species: list[Species] = []
    with path.open('r', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh):
            species.append(
                Species(
                    canonical_id=row['canonical_id'].strip(),
                    display_name=row.get('display_name', '').strip() or row['canonical_id'].strip(),
                    phase=row['phase'].strip(),
                    charge=int(float(row.get('charge', 0) or 0)),
                    mass_amu=float(row.get('mass_amu', 0.0) or 0.0),
                    elements=parse_semicolon_map(row.get('elements', '')),
                    aliases=parse_pipe_list(row.get('aliases', '')),
                    state_tags=set(parse_pipe_list(row.get('state_tags', ''))),
                    zones=parse_pipe_list(row.get('zones', '')),
                    surfaces=parse_pipe_list(row.get('surfaces', '')),
                )
            )
    return species


def _load_reactions(path: Path) -> list[Reaction]:
    rxns: list[Reaction] = []
    with path.open('r', encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh):
            reactants, products = parse_equation(row['equation'])
            rxns.append(
                Reaction(
                    reaction_id=row['reaction_id'].strip(),
                    phase=row['phase'].strip(),
                    equation=row['equation'].strip(),
                    reactants=reactants,
                    products=products,
                    rate_model_key=row['rate_model_key'].strip(),
                    energy_model_key=(row.get('energy_model_key') or '').strip() or None,
                    zone_filter=parse_pipe_list(row.get('zone_filter', '')),
                    surface_filter=parse_pipe_list(row.get('surface_filter', '')),
                    enabled=parse_csv_bool(row.get('enabled', True)),
                    notes=row.get('notes', '').strip(),
                )
            )
    return rxns


def _load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def _resolve(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _load_chemistry_manifest(path: Path) -> dict[str, Any]:
    raw = _load_yaml(path)
    model_files = {
        str(k): _resolve(path.parent, v)
        for k, v in (raw.get('model_files', {}) or {}).items()
    }
    return {
        'species': _resolve(path.parent, raw.get('species_file') or raw.get('species')),
        'gas_reactions': _resolve(path.parent, raw.get('gas_reactions_file') or raw.get('gas_reactions')),
        'surface_reactions': _resolve(path.parent, raw.get('surface_reactions_file') or raw.get('surface_reactions')),
        'reaction_models': _resolve(path.parent, raw.get('reaction_models_file') or raw.get('reaction_models')),
        'model_files': model_files,
        'aliases': _resolve(path.parent, raw.get('aliases_file') or raw.get('aliases')),
        'cross_sections_manifest': _resolve(path.parent, raw.get('cross_sections_manifest')),
    }


_MODEL_FILE_BACKENDS = {
    'electron_impact': {'electron_impact_xsec'},
    'gas_rate': {'arrhenius', 'constant', 'first_order_loss', 'te_power_law', 'electron_temperature_power_law'},
    'energy_loss': {'constant_event_loss'},
}


def _load_rate_models_from_file(path: Path, category: str) -> dict[str, dict[str, Any]]:
    raw = _load_yaml(path)
    models = raw.get('rate_models', {}) or {}
    allowed = _MODEL_FILE_BACKENDS.get(category)
    if allowed is None:
        return models
    bad = [
        (key, str(model.get('backend', '')).lower())
        for key, model in models.items()
        if str(model.get('backend', '')).lower() not in allowed
    ]
    if bad:
        formatted = ', '.join(f'{key}:{backend}' for key, backend in bad)
        allowed_s = ', '.join(sorted(allowed))
        raise ValueError(
            f'Model file {path} is declared as {category!r}, but contains unsupported backends '
            f'{formatted}. Allowed backends: {allowed_s}'
        )
    return models


def _merge_rate_models(target: dict[str, dict[str, Any]], incoming: dict[str, dict[str, Any]], source: Path) -> None:
    duplicates = sorted(set(target) & set(incoming))
    if duplicates:
        raise ValueError(f'Duplicate rate/energy model keys in {source}: {duplicates}')
    target.update(incoming)


def load_mechanism_bundle(chemistry_source: str | Path) -> MechanismBundle:
    """Load chemistry either from a directory or from an explicit manifest YAML.

    Directory mode preserves backward compatibility with the earlier skeleton.
    Manifest mode is recommended for long-term maintenance because it makes the
    chemistry file set explicit and reviewable in a single YAML file.
    """

    chemistry_source = Path(chemistry_source)
    if chemistry_source.is_file():
        files = _load_chemistry_manifest(chemistry_source)
        species_file = files['species']
        gas_reactions_file = files['gas_reactions']
        surface_reactions_file = files['surface_reactions']
        reaction_models_file = files['reaction_models']
        model_files = files['model_files']
        aliases_file = files['aliases']
        cross_sections_manifest = files['cross_sections_manifest']
    else:
        chemistry_dir = chemistry_source
        species_file = chemistry_dir / 'species.csv'
        gas_reactions_file = chemistry_dir / 'gas_reactions.csv'
        surface_reactions_file = chemistry_dir / 'surface_reactions.csv'
        reaction_models_file = chemistry_dir / 'reaction_models.yaml'
        model_files = {}
        aliases_file = chemistry_dir / 'aliases.yaml'
        cross_sections_manifest = chemistry_dir / 'cross_sections_manifest.yaml'

    species = _load_species(Path(species_file))
    gas_reactions = _load_reactions(Path(gas_reactions_file))
    surface_reactions = _load_reactions(Path(surface_reactions_file))
    rate_models: dict[str, dict[str, Any]] = {}
    if reaction_models_file and Path(reaction_models_file).exists():
        legacy_models = _load_rate_models_from_file(Path(reaction_models_file), 'legacy')
        _merge_rate_models(rate_models, legacy_models, Path(reaction_models_file))
    for category, model_file in model_files.items():
        if model_file and Path(model_file).exists():
            incoming = _load_rate_models_from_file(Path(model_file), category)
            _merge_rate_models(rate_models, incoming, Path(model_file))

    aliases_yaml = _load_yaml(Path(aliases_file)) if aliases_file and Path(aliases_file).exists() else {}
    aliases = aliases_yaml.get('aliases', {})

    cross_sections = {}
    if cross_sections_manifest and Path(cross_sections_manifest).exists():
        cross_sections = load_cross_sections_manifest(Path(cross_sections_manifest))

    return MechanismBundle(
        species=species,
        gas_reactions=gas_reactions,
        surface_reactions=surface_reactions,
        rate_models=rate_models,
        cross_sections=cross_sections,
        aliases=aliases,
    )
