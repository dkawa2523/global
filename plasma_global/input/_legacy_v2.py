"""Read-only schema-v2 compatibility boundary for :mod:`migrate_v2`.

It parses legacy inputs but never constructs a numerical model.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from plasma_global.errors import MigrationError


def _namespace(**values: Any) -> SimpleNamespace:
    return SimpleNamespace(**values)


def _mapping(value: object, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MigrationError(f"{where} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _list(value: object, where: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MigrationError(f"{where} must be a list")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise MigrationError(f"cannot read legacy YAML {path}: {exc}") from exc
    return _mapping(value, str(path))


def _expand_environment(value: Any) -> Any:
    """Reproduce v2 path expansion inside the migration boundary only."""

    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, value in override.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return override


def _load_with_includes(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    source = path.resolve(strict=False)
    if source in stack:
        chain = " -> ".join(str(item) for item in (*stack, source))
        raise MigrationError(f"legacy include cycle: {chain}")
    raw = _expand_environment(_read_yaml(source))
    include = raw.pop("include", None)
    includes = raw.pop("includes", None)
    if include is not None and includes is not None:
        raise MigrationError(f"{source} uses both include and includes")
    selected = include if include is not None else includes
    if selected is None:
        selected = list[object]()
    elif isinstance(selected, (str, Path)):
        selected = [selected]
    if not isinstance(selected, list):
        raise MigrationError(f"{source}: legacy include must be a path or path list")
    merged: dict[str, Any] = {}
    for item in selected:
        include_path = source.parent / Path(str(item))
        merged = _merge(merged, _load_with_includes(include_path, (*stack, source)))
    return _merge(merged, raw)


def _path(base: Path, value: object, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MigrationError(f"{where} must be a non-empty path")
    result = Path(value)
    if not result.is_absolute():
        result = base / result
    return result.resolve(strict=False)


def _defaults(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    return {**defaults, **raw}


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as exc:
        raise MigrationError(f"cannot read legacy {label} {path}: {exc}") from exc


def _species(row: dict[str, str], path: Path) -> SimpleNamespace:
    species_id = str(row.get("canonical_id") or "").strip()
    if not species_id:
        raise MigrationError(f"{path} contains a species without canonical_id")
    tags = {
        item.strip()
        for item in str(row.get("state_tags") or "").replace(";", "|").split("|")
        if item.strip()
    }
    return _namespace(
        canonical_id=species_id,
        charge=int(float(row.get("charge") or 0)),
        phase=str(row.get("phase") or "gas"),
        state_tags=tags,
    )


def _load_species(
    manifest_path: Path,
) -> tuple[list[SimpleNamespace], list[SimpleNamespace]]:
    manifest = _read_yaml(manifest_path)
    species_path = _path(
        manifest_path.parent,
        manifest.get("species_file"),
        "legacy chemistry species_file",
    )
    gas: list[SimpleNamespace] = []
    surface: list[SimpleNamespace] = []
    for row in _read_csv(species_path, "species"):
        item = _species(row, species_path)
        if item.phase == "surface":
            surface.append(item)
        elif item.canonical_id != "e":
            gas.append(item)
    return gas, surface


def _gas_reactants(equation: str, gas_species_ids: set[str]) -> list[str]:
    reactants: list[str] = []
    for token in equation.split("->", 1)[0].split("+"):
        fields = token.strip().split(maxsplit=1)
        species_id = (
            fields[1]
            if len(fields) == 2 and fields[0].replace(".", "", 1).isdigit()
            else token.strip()
        )
        if species_id in gas_species_ids:
            reactants.append(species_id)
    return reactants


def _surface_reaction(
    row: dict[str, str],
    path: Path,
    line: int,
    gas_species_ids: set[str],
) -> SimpleNamespace:
    reaction_id = str(row.get("reaction_id") or "").strip()
    equation = str(row.get("equation") or "")
    if not reaction_id or equation.count("->") != 1:
        raise MigrationError(f"{path}:{line} has an invalid reaction ID/equation")
    enabled = str(row.get("enabled") or "true").strip().lower()
    if enabled not in {"true", "false", "1", "0", "yes", "no"}:
        raise MigrationError(f"{path}:{line} has an invalid enabled value")
    return _namespace(
        reaction_id=reaction_id,
        gas_reactants=_gas_reactants(equation, gas_species_ids),
        surface_filter=[
            item.strip()
            for item in str(row.get("surface_filter") or "").split("|")
            if item.strip()
        ],
        enabled=enabled in {"true", "1", "yes"},
    )


def _load_surface_reactions(
    manifest_path: Path, gas_species_ids: set[str]
) -> list[SimpleNamespace]:
    manifest = _read_yaml(manifest_path)
    filename = manifest.get("surface_reactions_file")
    if not filename:
        return []
    path = _path(
        manifest_path.parent,
        filename,
        "legacy chemistry surface_reactions_file",
    )
    return [
        _surface_reaction(row, path, line, gas_species_ids)
        for line, row in enumerate(_read_csv(path, "surface reactions"), start=2)
    ]


def _as_dict(value: object) -> dict[str, Any]:
    return dict(value or {})


def _load_chamber(path: Path) -> SimpleNamespace:
    raw = _read_yaml(path)
    zones = []
    for index, value in enumerate(_list(raw.get("zones"), f"{path}.zones")):
        item = _mapping(value, f"{path}.zones[{index}]")
        zones.append(
            _namespace(
                zone_id=str(item.get("zone_id", "")),
                description=str(item.get("description", "")),
                volume_m3=float(item.get("volume_m3", 0.0)),
                pressure_Pa=float(item.get("pressure_Pa", 0.0)),
                gas_temperature_K=float(item.get("gas_temperature_K", 300.0)),
                role=str(item.get("role", "process")),
                initial_densities_m3=_as_dict(item.get("initial_densities_m3")),
            )
        )
    edges = []
    for value in _list(raw.get("edges"), f"{path}.edges"):
        item = _mapping(value, f"{path}.edges[]")
        edges.append(
            _namespace(
                edge_id=str(item.get("edge_id", "")),
                from_zone=str(item.get("from_zone", "")),
                to_zone=str(item.get("to_zone", "")),
                conductance_m3_s=float(item.get("conductance_m3_s", 0.0)),
                notes=str(item.get("notes", "")),
            )
        )
    surfaces = []
    for value in _list(raw.get("surfaces"), f"{path}.surfaces"):
        item = _mapping(value, f"{path}.surfaces[]")
        surfaces.append(
            _namespace(
                surface_id=str(item.get("surface_id", "")),
                zone_id=str(item.get("zone_id", "")),
                kind=str(item.get("kind", "wall")),
                area_m2=float(item.get("area_m2", 0.0)),
                material=str(item.get("material", "")),
                temperature_K=float(item.get("temperature_K", 300.0)),
                site_density_m2=float(item.get("site_density_m2", 0.0)),
                initial_coverages=_as_dict(item.get("initial_coverages")),
                initial_inventory=_as_dict(item.get("initial_inventory")),
                models=_as_dict(item.get("models")),
            )
        )
    inlets = []
    for value in _list(raw.get("gas_inlets"), f"{path}.gas_inlets"):
        item = _mapping(value, f"{path}.gas_inlets[]")
        inlets.append(
            _namespace(
                inlet_id=str(item.get("inlet_id", "")),
                zone_id=str(item.get("zone_id", "")),
                flow_sccm=_as_dict(item.get("flow_sccm")),
                temperature_K=float(item.get("temperature_K", 300.0)),
            )
        )
    pumps = []
    for value in _list(raw.get("pumps"), f"{path}.pumps"):
        item = _mapping(value, f"{path}.pumps[]")
        pumps.append(
            _namespace(
                pump_id=str(item.get("pump_id", "")),
                zone_id=str(item.get("zone_id", "")),
                speed_m3_s=float(item.get("speed_m3_s", 0.0)),
            )
        )
    ports = []
    for value in _list(raw.get("power_ports"), f"{path}.power_ports"):
        item = _mapping(value, f"{path}.power_ports[]")
        ports.append(
            _namespace(
                port_id=str(item.get("port_id", "")),
                kind=str(item.get("kind", "")),
                zone_id=str(item.get("zone_id", "")),
                coupling_target=str(item.get("coupling_target", "")),
                parameters=_as_dict(item.get("parameters")),
            )
        )
    return _namespace(
        chamber_id=str(raw.get("chamber_id", "")),
        description=str(raw.get("description", "")),
        zones=zones,
        edges=edges,
        surfaces=surfaces,
        gas_inlets=inlets,
        inlet_by_id={item.inlet_id: item for item in inlets},
        pumps=pumps,
        power_ports=ports,
    )


def _load_recipe(path: Path) -> SimpleNamespace:
    raw = _read_yaml(path)
    steps = []
    for index, value in enumerate(_list(raw.get("steps"), f"{path}.steps")):
        item = _mapping(value, f"{path}.steps[{index}]")
        steps.append(
            _namespace(
                step_id=str(item.get("step_id", "")),
                t_start_s=float(item.get("t_start_s", 0.0)),
                t_end_s=float(item.get("t_end_s", 0.0)),
                gas_inlets=dict(item.get("gas_inlets") or {}),
                power_ports=dict(item.get("power_ports") or {}),
                surface_overrides=dict(item.get("surface_overrides") or {}),
                imported_inputs=dict(item.get("imported_inputs") or {}),
            )
        )
    return _namespace(
        recipe_id=str(raw.get("recipe_id", "")),
        description=str(raw.get("description", "")),
        steps=steps,
    )


def _swarm(raw: dict[str, Any]) -> SimpleNamespace:
    cache = _defaults(
        _mapping(raw.get("cache"), "swarm.cache"),
        {"max_entries": 12, "fraction_decimals": 3},
    )
    two_term = _mapping(raw.get("boltzmann_2term"), "swarm.boltzmann_2term")
    energy = _defaults(
        _mapping(two_term.get("energy_grid"), "energy_grid"),
        {"min_eV": 1.0e-3, "max_eV": 160.0, "n": 360},
    )
    raw_field = _mapping(two_term.get("reduced_field_grid_Td"), "reduced_field_grid_Td")
    field = _defaults(
        raw_field,
        {"min": 0.2, "max": 2500.0, "n": 48},
    )
    table = _defaults(
        _mapping(raw.get("table"), "swarm.table"),
        {
            "file": None,
            "lookup": None,
            "bounds_policy": "clip",
            "electron_energy_mode": None,
            "energy_relaxation_time_s": 1.0e-6,
        },
    )
    profile = _defaults(
        _mapping(
            raw.get("prescribed_electron_profile"), "swarm.prescribed_electron_profile"
        ),
        {
            "file": None,
            "file_key": None,
            "zone_columns": {},
            "interpolation": "linear",
            "hold": "edge",
        },
    )
    return _namespace(
        model_name=str(raw.get("model_name", "table")),
        closure=str(raw.get("closure", "auto")),
        mixture_key_species=list(raw.get("mixture_key_species") or []),
        cache=_namespace(**cache),
        boltzmann_2term=_namespace(
            energy_grid=_namespace(**energy),
            reduced_field_grid_Td=_namespace(**field),
            reduced_field_grid_was_explicit=bool(raw_field),
            max_shape_iterations=int(two_term.get("max_shape_iterations", 48)),
        ),
        table=_namespace(**table),
        prescribed_electron_profile=_namespace(**profile),
    )


def _extension_records(manifest_path: Path) -> tuple[list[Any], list[Any]]:
    manifest = _read_yaml(manifest_path)
    extension = manifest.get("extensions") or manifest.get("extensions_file")
    if isinstance(extension, str):
        extension = _read_yaml(
            _path(manifest_path.parent, extension, "chemistry.extensions")
        )
    data = extension if isinstance(extension, dict) else dict[str, Any]()
    return list(data.get("state_variables") or []), list(data.get("processes") or [])


def load_legacy_v2_case(path: str | Path) -> SimpleNamespace:
    """Read enough of schema v2 to produce one explicit schema-v3 document."""

    source = Path(path).resolve(strict=False)
    raw = _load_with_includes(source)
    case = _defaults(
        _mapping(raw.get("case"), "case"),
        {
            "name": "",
            "description": "",
            "tags": [],
            "schema_version": 2,
            "kind": "plasma_global_case",
        },
    )
    if int(case.get("schema_version", 2)) != 2:
        raise MigrationError(f"{source} is not a schema-v2 case")
    files = _mapping(raw.get("files"), "files")
    chemistry_entry = _mapping(files.get("chemistry"), "files.chemistry")
    chamber_path = _path(source.parent, files.get("chamber"), "files.chamber")
    recipe_path = _path(source.parent, files.get("recipe"), "files.recipe")
    chemistry_path = _path(
        source.parent,
        chemistry_entry.get("manifest"),
        "files.chemistry.manifest",
    )
    external = {
        str(key): str(_path(source.parent, value, f"files.external_inputs.{key}"))
        for key, value in _mapping(
            files.get("external_inputs"), "files.external_inputs"
        ).items()
    }

    physics = _defaults(
        _mapping(raw.get("physics"), "physics"),
        {
            "mode": "transient",
            "gas_model": "multi_zone_global",
            "eedf_backend": "maxwell",
            "electrical_backend": "icp",
            "integrator": "scipy_bdf",
            "enable_gas_temperature": True,
            "enable_surface_coverages": True,
            "enable_wall_inventory": True,
            "electron_density_closure": "quasi_neutral",
            "gas_heating_fraction": 0.15,
            "wall_relaxation_s_inv": 500.0,
        },
    )
    numerics = _defaults(
        _mapping(raw.get("numerics"), "numerics"),
        {
            "rtol": 1.0e-6,
            "atol": 1.0e-14,
            "first_step": 1.0e-10,
            "max_step": 1.0e-6,
            "positivity": None,
            "events": None,
        },
    )
    output_raw = _mapping(raw.get("outputs"), "outputs")
    formats = _defaults(
        _mapping(output_raw.get("formats"), "outputs.formats"),
        {"solution_h5": True, "observables_csv": True, "summary_yaml": True},
    )
    plots = _defaults(
        _mapping(output_raw.get("plots"), "outputs.plots"),
        {"enabled": False, "format": ["png"], "dpi": 150, "items": []},
    )
    budgets = _defaults(
        _mapping(output_raw.get("budgets"), "outputs.budgets"),
        {"enabled": False},
    )
    gas_species, surface_species = _load_species(chemistry_path)
    surface_reactions = _load_surface_reactions(
        chemistry_path, {str(species.canonical_id) for species in gas_species}
    )
    state_variables, processes = _extension_records(chemistry_path)

    chamber = _load_chamber(chamber_path)
    recipe = _load_recipe(recipe_path)
    if not recipe.steps:
        raise MigrationError(f"legacy recipe {recipe_path} contains no steps")
    return _namespace(
        run_config=_namespace(
            case=_namespace(**case),
            physics=_namespace(**physics),
            numerics=_namespace(**numerics),
            outputs=_namespace(
                formats=_namespace(**formats),
                plots=_namespace(**plots),
                budgets=_namespace(**budgets),
            ),
            swarm=_swarm(_mapping(raw.get("swarm"), "swarm")),
        ),
        chamber=chamber,
        recipe=recipe,
        mechanism=_namespace(
            gas_state_species=gas_species,
            surface_species=surface_species,
            surface_reactions=surface_reactions,
            state_variables=state_variables,
            processes=processes,
        ),
        resolved_paths=_namespace(
            base_dir=str(source.parent),
            recipe_file=str(recipe_path),
            chemistry_manifest=str(chemistry_path),
            chemistry_dir=str(chemistry_path.parent),
            external_inputs=external,
        ),
    )


__all__ = ["load_legacy_v2_case"]
