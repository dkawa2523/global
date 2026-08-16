"""Canonical chemistry manifest traversal and YAML validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global._yaml import YamlLoadError, load_unique_yaml
from plasma_global.chemistry._contracts import cross_section_support_error
from plasma_global.chemistry._data_csv import (
    electron_energy_values,
    load_cross_section_curve,
    load_reactions,
    load_species,
)
from plasma_global.chemistry.data import (
    ChemistryData,
    CrossSectionData,
    RateModelData,
)
from plasma_global.errors import ChemistryError


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChemistryError(f"{where} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _only_keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ChemistryError(f"{where} contains unknown keys: {', '.join(unknown)}")


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = load_unique_yaml(stream)
    except (OSError, YamlLoadError) as exc:
        raise ChemistryError(
            f"cannot read canonical chemistry YAML {path}: {exc}"
        ) from exc
    return _mapping(document or {}, str(path))


def _required_path(base: Path, value: object, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ChemistryError(f"{where} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    if not path.is_file():
        raise ChemistryError(f"{where} does not exist: {path}")
    return path


def _required_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChemistryError(f"{where} must be a non-empty string")
    return value


def _optional_path(base: Path, value: object, where: str) -> Path | None:
    if value is None:
        return None
    return _required_path(base, value, where)


_RATE_KEYS: dict[str, set[str]] = {
    "electron_impact": {"cross_section", "branching_yield"},
    "arrhenius": {"A", "beta", "activation_eV", "overall_order"},
    "constant": {"value", "overall_order"},
    "first_order": {"rate_s_inv"},
    "experimental.electron_temperature_power_law": {
        "A",
        "reference_temperature_K",
        "exponent",
        "overall_order",
    },
    "tabulated_1d": {"axis", "file", "bounds", "overall_order"},
    "sticking": {"value", "coverage"},
    "ion_assisted": {
        "yield",
        "threshold_eV",
        "reference_energy_eV",
        "exponent",
        "coverage",
    },
    "desorption": {"frequency_s_inv", "activation_eV"},
    "langmuir_hinshelwood": {"A_m2_s_inv", "activation_eV"},
}

_RATE_REQUIRED: dict[str, set[str]] = {
    "electron_impact": {"cross_section", "branching_yield"},
    "arrhenius": {"A", "beta", "activation_eV"},
    "constant": {"value"},
    "first_order": {"rate_s_inv"},
    "experimental.electron_temperature_power_law": {
        "A",
        "reference_temperature_K",
        "exponent",
    },
    "tabulated_1d": {"axis", "file", "bounds"},
    "sticking": {"value"},
    "ion_assisted": {
        "yield",
        "threshold_eV",
        "reference_energy_eV",
        "exponent",
    },
    "desorption": {"frequency_s_inv", "activation_eV"},
    "langmuir_hinshelwood": {"A_m2_s_inv", "activation_eV"},
}


def _finite_number(
    model: Mapping[str, Any], name: str, where: str, *, minimum: float | None = None
) -> float:
    value = model[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChemistryError(f"{where}.{name} must be a number")
    number = float(value)
    if not np.isfinite(number) or (minimum is not None and number < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum:g}"
        raise ChemistryError(f"{where}.{name} must be {suffix}")
    return number


def _validate_coverage(value: object, where: str) -> None:
    if value is None:
        return
    config = _mapping(value, where)
    kind = config.get("kind")
    if kind == "constant":
        _only_keys(config, {"kind"}, where)
        return
    if kind not in {"site_blocking", "species_power"}:
        raise ChemistryError(f"{where}.kind is unsupported: {kind!r}")
    species_key = "site_species" if kind == "site_blocking" else "species"
    _only_keys(config, {"kind", species_key, "exponent"}, where)
    missing = [name for name in (species_key, "exponent") if name not in config]
    if missing:
        raise ChemistryError(f"{where} is missing: {', '.join(missing)}")
    if not isinstance(config[species_key], str) or not config[species_key].strip():
        raise ChemistryError(f"{where}.{species_key} must be a non-empty species ID")
    _finite_number(config, "exponent", where, minimum=0.0)


def _validate_tabulated_rate_parameters(model: Mapping[str, Any], where: str) -> None:
    if model["axis"] not in {
        "gas_temperature_K",
        "mean_energy_eV",
        "electron_temperature_eV",
        "reduced_field_Td",
        "pressure_Pa",
    }:
        raise ChemistryError(f"{where}.axis is unsupported: {model['axis']!r}")
    if model["bounds"] not in {"error", "clip"}:
        raise ChemistryError(f"{where}.bounds must be error or clip")


def _validate_gas_rate_parameters(
    kind: str, model: Mapping[str, Any], where: str
) -> None:
    if kind == "electron_impact":
        if (
            not isinstance(model["cross_section"], str)
            or not model["cross_section"].strip()
        ):
            raise ChemistryError(f"{where}.cross_section must be a non-empty ID")
        _finite_number(model, "branching_yield", where, minimum=0.0)
    elif kind == "arrhenius":
        _finite_number(model, "A", where, minimum=0.0)
        _finite_number(model, "beta", where)
        _finite_number(model, "activation_eV", where, minimum=0.0)
    elif kind == "constant":
        _finite_number(model, "value", where, minimum=0.0)
    elif kind == "first_order":
        _finite_number(model, "rate_s_inv", where, minimum=0.0)
    elif kind == "experimental.electron_temperature_power_law":
        _finite_number(model, "A", where, minimum=0.0)
        reference = _finite_number(model, "reference_temperature_K", where, minimum=0.0)
        if reference == 0.0:
            raise ChemistryError(f"{where}.reference_temperature_K must be positive")
        _finite_number(model, "exponent", where)
    elif kind == "tabulated_1d":
        _validate_tabulated_rate_parameters(model, where)


def _validate_surface_rate_parameters(
    kind: str, model: Mapping[str, Any], where: str
) -> None:
    if kind == "sticking":
        value = _finite_number(model, "value", where, minimum=0.0)
        if value > 1.0:
            raise ChemistryError(f"{where}.value must not exceed one")
        _validate_coverage(model.get("coverage"), f"{where}.coverage")
    elif kind == "ion_assisted":
        _finite_number(model, "yield", where, minimum=0.0)
        threshold = _finite_number(model, "threshold_eV", where, minimum=0.0)
        reference = _finite_number(model, "reference_energy_eV", where, minimum=0.0)
        if reference <= threshold:
            raise ChemistryError(
                f"{where}.reference_energy_eV must exceed threshold_eV"
            )
        _finite_number(model, "exponent", where, minimum=0.0)
        _validate_coverage(model.get("coverage"), f"{where}.coverage")
    elif kind in {"desorption", "langmuir_hinshelwood"}:
        amplitude = "frequency_s_inv" if kind == "desorption" else "A_m2_s_inv"
        _finite_number(model, amplitude, where, minimum=0.0)
        _finite_number(model, "activation_eV", where, minimum=0.0)


def _validate_rate_parameters(kind: str, model: Mapping[str, Any], where: str) -> None:
    missing = sorted(_RATE_REQUIRED[kind] - set(model))
    if missing:
        raise ChemistryError(f"{where} is missing: {', '.join(missing)}")
    if "overall_order" in model:
        _finite_number(model, "overall_order", where, minimum=0.0)
    if kind in {
        "electron_impact",
        "arrhenius",
        "constant",
        "first_order",
        "experimental.electron_temperature_power_law",
        "tabulated_1d",
    }:
        _validate_gas_rate_parameters(kind, model, where)
    else:
        _validate_surface_rate_parameters(kind, model, where)


def _load_rate_models(
    path: Path, source_files: list[Path]
) -> Mapping[str, RateModelData]:
    raw = _yaml_mapping(path)
    _only_keys(raw, {"rate_models"}, str(path))
    models = _mapping(raw.get("rate_models"), f"{path}.rate_models")
    result: dict[str, RateModelData] = {}
    for model_id, value in models.items():
        model = _mapping(value, f"{path}.rate_models.{model_id}")
        kind = model.pop("kind", None)
        if kind not in _RATE_KEYS:
            raise ChemistryError(
                f"{path}: rate model {model_id!r} has unsupported kind {kind!r}"
            )
        _only_keys(model, _RATE_KEYS[kind], f"{path}.rate_models.{model_id}")
        _validate_rate_parameters(kind, model, f"{path}.rate_models.{model_id}")
        if "file" in model:
            model["file"] = _required_path(
                path.parent, model["file"], f"rate model {model_id}.file"
            )
            source_files.append(model["file"])
        result[model_id] = RateModelData(model_id, kind, MappingProxyType(model))
    return MappingProxyType(result)


_CROSS_SECTION_KEYS = {
    "id",
    "kind",
    "target",
    "threshold_eV",
    "energy_loss_eV",
    "electron_energy_transfer_eV",
    "file",
    "metadata",
}
_REQUIRED_CROSS_SECTION_KEYS = ("id", "kind", "target", "threshold_eV", "file")


def _load_cross_section_entry(
    path: Path,
    index: int,
    value: object,
    source_files: list[Path],
    existing: Mapping[str, CrossSectionData],
) -> CrossSectionData:
    location = f"{path}.cross_sections[{index}]"
    entry = _mapping(value, location)
    _only_keys(entry, _CROSS_SECTION_KEYS, location)
    missing = [
        key
        for key in _REQUIRED_CROSS_SECTION_KEYS
        if key not in entry
        or entry[key] is None
        or (isinstance(entry[key], str) and not entry[key].strip())
    ]
    if missing:
        raise ChemistryError(f"{location} missing: {', '.join(missing)}")
    cross_section_id = _required_text(entry["id"], f"{location}.id")
    if cross_section_id in existing:
        raise ChemistryError(f"duplicate cross-section id: {cross_section_id}")
    curve_path = _required_path(
        path.parent, entry["file"], f"cross section {cross_section_id}.file"
    )
    source_files.append(curve_path)
    kind = _required_text(entry["kind"], f"{location}.kind")
    target = _required_text(entry["target"], f"{location}.target")
    raw_threshold = entry["threshold_eV"]
    energy, sigma = load_cross_section_curve(
        curve_path,
        cross_section_id=cross_section_id,
        kind=kind,
        target=target,
        threshold_eV=raw_threshold,
    )
    loss, transfer = electron_energy_values(
        entry, f"cross section {cross_section_id}", required=True
    )
    metadata = _mapping(
        entry.get("metadata", {}), f"cross section {cross_section_id}.metadata"
    )
    cross_section = CrossSectionData(
        id=cross_section_id,
        kind=kind,
        target=target,
        threshold_eV=float(raw_threshold),
        energy_loss_eV=loss,
        energy_eV=energy,
        sigma_m2=sigma,
        metadata=MappingProxyType(metadata),
        electron_energy_transfer_eV=transfer,
    )
    support_error = cross_section_support_error(cross_section)
    if support_error is not None:
        raise ChemistryError(support_error)
    return cross_section


def _load_cross_sections(
    path: Path | None, source_files: list[Path]
) -> Mapping[str, CrossSectionData]:
    if path is None:
        return MappingProxyType({})
    raw = _yaml_mapping(path)
    _only_keys(raw, {"cross_sections"}, str(path))
    entries = raw.get("cross_sections")
    if not isinstance(entries, list):
        raise ChemistryError(f"{path}.cross_sections must be a list")
    result: dict[str, CrossSectionData] = {}
    for index, value in enumerate(entries):
        cross_section = _load_cross_section_entry(
            path, index, value, source_files, result
        )
        result[cross_section.id] = cross_section
    return MappingProxyType(result)


def load_chemistry(path: str | Path) -> ChemistryData:
    """Load a canonical v3 chemistry manifest and all referenced SI data."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ChemistryError(f"chemistry manifest does not exist: {source}")
    raw = _yaml_mapping(source)
    allowed = {
        "schema_version",
        "species",
        "gas_reactions",
        "boundary_reactions",
        "surface_reactions",
        "rate_models",
        "cross_sections",
        "experimental",
        "provenance",
    }
    _only_keys(raw, allowed, str(source))
    if raw.get("schema_version") != 3:
        raise ChemistryError(f"{source}: chemistry schema_version must be 3")
    base = source.parent
    species_path = _required_path(base, raw.get("species"), "chemistry.species")
    gas_path = _required_path(base, raw.get("gas_reactions"), "chemistry.gas_reactions")
    rate_path = _required_path(base, raw.get("rate_models"), "chemistry.rate_models")
    boundary_path = _optional_path(
        base, raw.get("boundary_reactions"), "chemistry.boundary_reactions"
    )
    surface_path = _optional_path(
        base, raw.get("surface_reactions"), "chemistry.surface_reactions"
    )
    cross_section_path = _optional_path(
        base, raw.get("cross_sections"), "chemistry.cross_sections"
    )
    experimental = _mapping(raw.get("experimental", {}), "chemistry.experimental")
    provenance = _mapping(raw.get("provenance", {}), "chemistry.provenance")
    source_files = [source, species_path, gas_path, rate_path]
    source_files.extend(
        item
        for item in (boundary_path, surface_path, cross_section_path)
        if item is not None
    )
    return ChemistryData(
        source=source,
        species=load_species(species_path),
        gas_reactions=load_reactions(gas_path),
        boundary_reactions=load_reactions(boundary_path, family="boundary"),
        surface_reactions=load_reactions(surface_path, family="surface"),
        rate_models=_load_rate_models(rate_path, source_files),
        cross_sections=_load_cross_sections(cross_section_path, source_files),
        experimental=MappingProxyType(experimental),
        provenance=MappingProxyType(provenance),
        source_files=tuple(dict.fromkeys(source_files)),
    )
