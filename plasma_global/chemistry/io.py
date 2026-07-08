from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

from plasma_global.chemistry.cross_sections import load_cross_sections_manifest
from plasma_global.chemistry.extensions import load_processes, load_state_variables
from plasma_global.chemistry.manifest import load_chemistry_manifest
from plasma_global.chemistry.models import MechanismBundle, Reaction, Species
from plasma_global.chemistry.parser import parse_csv_bool, parse_equation, parse_pipe_list, parse_semicolon_map
from plasma_global.chemistry.provenance import PROVENANCE_FIELDS, provenance_from_mapping
from plasma_global.chemistry.rate_model_io import load_rate_models_from_file, merge_rate_models


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
            provenance = _reaction_row_provenance(row)
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
                    provenance=provenance,
                )
            )
    return rxns


def _parse_provenance_cell(value: str | None) -> dict[str, Any]:
    if value is None or not str(value).strip():
        return {}
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        return {'_invalid_provenance': value}
    if isinstance(parsed, dict):
        return parsed
    return {'_invalid_provenance': value}


def _reaction_row_provenance(row: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    raw.update(_parse_provenance_cell(row.get('provenance')))
    for key in PROVENANCE_FIELDS:
        if key == 'notes':
            continue
        value = row.get(key)
        if value is not None and str(value).strip():
            raw[key] = value.strip() if isinstance(value, str) else value
    return provenance_from_mapping(raw)


def _require_manifest_file(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f'Chemistry manifest entry {label} does not exist: {path}')
    return path


def load_mechanism_bundle(chemistry_source: str | Path) -> MechanismBundle:
    """Load chemistry from an explicit manifest YAML."""

    chemistry_source = Path(chemistry_source)
    if not chemistry_source.is_file():
        raise ValueError(f'Chemistry source must be a manifest file: {chemistry_source}')
    files = load_chemistry_manifest(chemistry_source)
    species_file = files['species']
    gas_reactions_file = files['gas_reactions']
    surface_reactions_file = files['surface_reactions']
    model_files = files['model_files']
    cross_sections_manifest = files['cross_sections_manifest']
    state_variables_file = files['state_variables']
    processes_file = files['processes']

    species = _load_species(Path(species_file))
    gas_reactions = _load_reactions(Path(gas_reactions_file))
    surface_reactions = _load_reactions(Path(surface_reactions_file))
    rate_models: dict[str, dict[str, Any]] = {}
    for category, model_file in model_files.items():
        model_path = _require_manifest_file(Path(model_file) if model_file else None, f'model_files.{category}')
        if model_path is None:
            raise ValueError(f'Chemistry manifest entry model_files.{category} must be a file path')
        incoming = load_rate_models_from_file(model_path, category)
        merge_rate_models(rate_models, incoming, model_path)

    cross_sections = {}
    cross_sections_path = _require_manifest_file(Path(cross_sections_manifest) if cross_sections_manifest else None, 'cross_sections_manifest')
    if cross_sections_path is not None:
        cross_sections = load_cross_sections_manifest(cross_sections_path)
    state_variables = load_state_variables(_require_manifest_file(Path(state_variables_file) if state_variables_file else None, 'extensions.state_variables'))
    processes = load_processes(_require_manifest_file(Path(processes_file) if processes_file else None, 'extensions.processes'))

    return MechanismBundle(
        species=species,
        gas_reactions=gas_reactions,
        surface_reactions=surface_reactions,
        rate_models=rate_models,
        cross_sections=cross_sections,
        state_variables=state_variables,
        processes=processes,
    )
