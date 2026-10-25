"""Inline YAML definitions and external CSVs share the same chemistry objects."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from plasma_global.chemistry._data_csv import (
    electron_energy_values,
    load_reactions,
    load_species,
)
from plasma_global.chemistry._data_manifest import (
    _load_cross_section_entry,
    _load_cross_sections,
    _load_rate_models,
    _only_keys,
    _required_path,
    parse_rate_models,
)
from plasma_global.chemistry.data import (
    ChemistryData,
    RateModelData,
    ReactionData,
    SpeciesData,
    parse_equation,
)
from plasma_global.errors import ChemistryError


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Species(_Record):
    id: str
    phase: Literal["gas", "surface"] = "gas"
    charge: int = 0
    mass_amu: float
    elements: dict[str, float] = Field(default_factory=dict)
    state_tags: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)
    cv_over_kb: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _Reaction(_Record):
    id: str
    equation: str
    rate: str | dict[str, Any] | None = None
    energy_loss_eV: float | None = None
    electron_energy_transfer_eV: float | None = None
    gas_heating_eV: float = 0.0
    zones: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _path(value: Any, base: Path, files: list[Path]) -> Path:
    path = _required_path(base, value, "chemistry file")
    files.append(path)
    return path


def _species(values: list[Any]) -> tuple[SpeciesData, ...]:
    result = []
    for value in values:
        species = _Species.model_validate(value)
        data = species.model_dump()
        data["state_tags"] = frozenset(species.state_tags)
        data["surfaces"] = tuple(species.surfaces)
        result.append(SpeciesData(**data))
    return tuple(result)


def _reactions(
    values: Any,
    family: Literal["gas", "surface", "boundary"],
    base: Path,
    source: Path,
    files: list[Path],
    models: dict[str, Any],
) -> tuple[ReactionData, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return load_reactions(_path(values, base, files), family=family)
    if not isinstance(values, list):
        raise ChemistryError(f"{source}: {family} reactions must be a list or CSV path")
    result = []
    for value in values:
        record = _Reaction.model_validate(value)
        rate_id = record.rate
        if isinstance(rate_id, dict):
            key = f"reaction:{record.id}"
            if key in models:
                raise ChemistryError(f"inline rate ID collision: {key}")
            models[key] = rate_id
            rate_id = key
        if family == "boundary" and rate_id is not None:
            raise ChemistryError("boundary reactions cannot declare a rate")
        reactants, products = parse_equation(record.equation)
        loss, transfer = electron_energy_values(
            record.model_dump(), record.id, required=False
        )
        result.append(
            ReactionData(
                record.id,
                reactants,
                products,
                rate_id,
                loss,
                record.gas_heating_eV,
                tuple(record.zones),
                tuple(record.surfaces),
                record.metadata,
                transfer,
            )
        )
    return tuple(result)


def _curves(curves: Any, source: Path, base: Path, files: list[Path]) -> dict[str, Any]:
    cross_sections = {}
    if isinstance(curves, str):
        cross_sections.update(_load_cross_sections(_path(curves, base, files), files))
    elif isinstance(curves, list):
        for index, curve in enumerate(curves):
            record = _load_cross_section_entry(
                base / source.name, index, curve, files, cross_sections
            )
            cross_sections[record.id] = record
    else:
        raise ChemistryError("cross_sections must be a list or YAML path")
    return cross_sections


def load_definition(raw: dict[str, Any], source: Path, base: Path) -> ChemistryData:
    _only_keys(
        raw,
        {
            "schema_version",
            "species",
            "gas_reactions",
            "boundary_reactions",
            "surface_reactions",
            "rate_models",
            "cross_sections",
            "experimental",
            "provenance",
        },
        str(source),
    )
    if raw.get("schema_version", 4) != 4:
        raise ChemistryError(f"{source}: chemistry schema_version must be 4")
    files = [source] if source.is_file() else []
    species_input = raw.get("species")
    species = (
        load_species(_path(species_input, base, files))
        if isinstance(species_input, str)
        else _species(species_input or [])
    )
    rates = raw.get("rate_models", {})
    external_rates: Mapping[str, RateModelData] = (
        _load_rate_models(_path(rates, base, files), files)
        if isinstance(rates, str)
        else {}
    )
    models = {} if isinstance(rates, str) else dict(rates)
    gas = _reactions(raw.get("gas_reactions", []), "gas", base, source, files, models)
    boundary = _reactions(
        raw.get("boundary_reactions"), "boundary", base, source, files, models
    )
    surface = _reactions(
        raw.get("surface_reactions"), "surface", base, source, files, models
    )
    parsed = dict(parse_rate_models(models, base / source.name, files))
    if collision := set(parsed) & set(external_rates):
        raise ChemistryError(f"duplicate rate model IDs: {sorted(collision)}")
    parsed.update(external_rates)
    cross_sections = _curves(raw.get("cross_sections", []), source, base, files)
    return ChemistryData(
        source,
        species,
        gas,
        boundary,
        surface,
        MappingProxyType(parsed),
        MappingProxyType(cross_sections),
        raw.get("experimental", {}),
        raw.get("provenance", {}),
        tuple(dict.fromkeys(files)),
    )
