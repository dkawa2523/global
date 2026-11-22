"""Canonical chemistry manifest traversal and YAML validation."""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from plasma_global._input_sources import InputSourceStore
from plasma_global._yaml import YamlLoadError, load_unique_yaml
from plasma_global.chemistry._contracts import cross_section_support_error
from plasma_global.chemistry._data_csv import (
    electron_energy_transfer,
    load_cross_section_curve,
)
from plasma_global.chemistry._rate_models import validate_rate_model
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


def _yaml_mapping(
    path: Path,
    sources: InputSourceStore | None = None,
    *,
    role: str = "chemistry.definition",
) -> dict[str, Any]:
    try:
        store = sources if sources is not None else InputSourceStore()
        document = load_unique_yaml(StringIO(store.read_text(path, role)))
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


def _load_rate_models(
    path: Path, source_files: list[Path], sources: InputSourceStore | None = None
) -> Mapping[str, RateModelData]:
    raw = _yaml_mapping(path, sources, role="chemistry.rate_models")
    _only_keys(raw, {"rate_models"}, str(path))
    models = raw.get("rate_models")
    if not isinstance(models, dict):
        raise ChemistryError(f"{path}.rate_models must be a mapping")
    return parse_rate_models(models, path, source_files, sources=sources)


def parse_rate_models(
    models: dict[str, Any],
    path: Path,
    source_files: list[Path],
    *,
    sources: InputSourceStore | None = None,
) -> Mapping[str, RateModelData]:
    store = sources if sources is not None else InputSourceStore()
    result: dict[str, RateModelData] = {}
    for model_id, value in models.items():
        model = _mapping(value, f"{path}.rate_models.{model_id}")
        kind = model.pop("kind", None)
        record = RateModelData(model_id, str(kind), MappingProxyType(model))
        validate_rate_model(record)
        if "file" in model:
            model["file"] = _required_path(
                path.parent, model["file"], f"rate model {model_id}.file"
            )
            store.read_bytes(model["file"], "declared_rate_table")
            source_files.append(model["file"])
        result[model_id] = RateModelData(model_id, kind, MappingProxyType(model))
    return MappingProxyType(result)


_CROSS_SECTION_KEYS = {
    "id",
    "kind",
    "target",
    "threshold_eV",
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
    sources: InputSourceStore | None = None,
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
        sources=sources,
    )
    transfer = electron_energy_transfer(
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
        energy_eV=energy,
        sigma_m2=sigma,
        metadata=MappingProxyType(metadata),
        electron_energy_transfer_eV=cast(float, transfer),
    )
    support_error = cross_section_support_error(cross_section)
    if support_error is not None:
        raise ChemistryError(support_error)
    return cross_section


def _load_cross_sections(
    path: Path | None, source_files: list[Path], sources: InputSourceStore | None = None
) -> Mapping[str, CrossSectionData]:
    if path is None:
        return MappingProxyType({})
    raw = _yaml_mapping(path, sources, role="chemistry.cross_sections")
    _only_keys(raw, {"cross_sections"}, str(path))
    entries = raw.get("cross_sections")
    if not isinstance(entries, list):
        raise ChemistryError(f"{path}.cross_sections must be a list")
    result: dict[str, CrossSectionData] = {}
    for index, value in enumerate(entries):
        cross_section = _load_cross_section_entry(
            path, index, value, source_files, result, sources
        )
        result[cross_section.id] = cross_section
    return MappingProxyType(result)


def load_chemistry(
    path: str | Path, *, sources: InputSourceStore | None = None
) -> ChemistryData:
    """Read one reusable schema-v4 chemistry definition."""
    from plasma_global.chemistry._data_yaml import load_definition

    source = Path(path).resolve()
    store = sources if sources is not None else InputSourceStore()
    return load_definition(
        _yaml_mapping(source, store), source, source.parent, sources=store
    )
