"""Public data contract for canonical schema-v4 chemistry.

The immutable objects in this module are the boundary between canonical input
and model compilation.  Strict file decoding lives in the private
``_data_csv`` and ``_data_manifest`` modules so callers have one short,
stable path: ``load_chemistry`` -> ``ChemistryData``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from plasma_global.chemistry._contracts import (
    chemistry_identifier,
    validate_rate_identifiers,
    validate_species_record,
)
from plasma_global.errors import ChemistryError

if TYPE_CHECKING:
    from plasma_global._input_sources import InputSourceStore

AMU_TO_KG = 1.66053906660e-27


def _empty_metadata() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class SpeciesData:
    id: str
    phase: str
    charge: int
    mass_amu: float
    elements: Mapping[str, float]
    state_tags: frozenset[str] = frozenset()
    surfaces: tuple[str, ...] = ()
    cv_over_kb: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        validate_species_record(self)
        object.__setattr__(self, "elements", MappingProxyType(dict(self.elements)))
        object.__setattr__(self, "state_tags", frozenset(self.state_tags))
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU_TO_KG


@dataclass(frozen=True, slots=True)
class ReactionData:
    id: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_model: str | None
    electron_energy_transfer_eV: float | None
    gas_heating_eV: float = 0.0
    zones: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        label = f"reaction {self.id!r}"
        chemistry_identifier(self.id, f"{label}.id")
        for name, values in (
            ("reactants", self.reactants),
            ("products", self.products),
        ):
            for species in values:
                chemistry_identifier(species, f"{label}.{name}")
            object.__setattr__(self, name, MappingProxyType(dict(values)))
        if self.rate_model is not None:
            chemistry_identifier(self.rate_model, f"{label}.rate_model")
        for name, values in (("zones", self.zones), ("surfaces", self.surfaces)):
            for identifier in values:
                chemistry_identifier(identifier, f"{label}.{name}")
            object.__setattr__(self, name, tuple(values))


@dataclass(frozen=True, slots=True)
class RateModelData:
    id: str
    kind: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        label = f"rate model {self.id!r}"
        chemistry_identifier(self.id, f"{label}.id")
        validate_rate_identifiers(self.parameters, label)
        parameters = dict(self.parameters)
        coverage = parameters.get("coverage")
        if isinstance(coverage, Mapping):
            parameters["coverage"] = MappingProxyType(dict(coverage))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


@dataclass(frozen=True, slots=True)
class CrossSectionData:
    id: str
    kind: str
    target: str
    threshold_eV: float
    electron_energy_transfer_eV: float
    energy_eV: np.ndarray
    sigma_m2: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        chemistry_identifier(self.id, "cross section id")
        chemistry_identifier(self.target, f"cross section {self.id!r}.target")


@dataclass(frozen=True, slots=True)
class ChemistryData:
    source: Path
    species: tuple[SpeciesData, ...]
    gas_reactions: tuple[ReactionData, ...]
    boundary_reactions: tuple[ReactionData, ...]
    surface_reactions: tuple[ReactionData, ...]
    rate_models: Mapping[str, RateModelData]
    cross_sections: Mapping[str, CrossSectionData]
    provenance: Mapping[str, Any] = field(default_factory=_empty_metadata)
    source_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        for name, records in (
            ("rate_models", self.rate_models),
            ("cross_sections", self.cross_sections),
        ):
            for identifier in records:
                chemistry_identifier(identifier, f"chemistry.{name} key")
            object.__setattr__(self, name, MappingProxyType(dict(records)))


_TERM = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s+)?([^\s].*?)\s*$")


def parse_equation(
    equation: str, *, allow_wall_absorption: bool = False
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    """Parse one irreversible equation without aliases or implicit species."""

    if equation.count("->") != 1:
        raise ChemistryError(f"reaction equation must contain one '->': {equation!r}")

    def side(text: str) -> Mapping[str, float]:
        values: dict[str, float] = {}
        for token in (part.strip() for part in text.split("+")):
            if not token:
                continue
            match = _TERM.match(token)
            if match is None:
                raise ChemistryError(f"invalid reaction term: {token!r}")
            coefficient_text, species_id = match.groups()
            coefficient = float(coefficient_text) if coefficient_text else 1.0
            species_id = species_id.strip()
            if coefficient <= 0.0 or not np.isfinite(coefficient) or not species_id:
                raise ChemistryError(f"invalid reaction term: {token!r}")
            values[species_id] = values.get(species_id, 0.0) + coefficient
        return MappingProxyType(values)

    lhs, rhs = equation.split("->")
    reactants = side(lhs)
    if not reactants:
        raise ChemistryError(f"reaction has no reactants: {equation!r}")
    if allow_wall_absorption and rhs.strip() == "0":
        return reactants, MappingProxyType({})
    products = side(rhs)
    if allow_wall_absorption and not products:
        raise ChemistryError(f"write '-> 0' for wall absorption: {equation!r}")
    return reactants, products


def load_chemistry(
    path: str | Path, *, sources: InputSourceStore | None = None
) -> ChemistryData:
    """Load a schema-v4 chemistry definition and its referenced SI data."""

    # The private loader imports these public data classes.  Deferring this
    # implementation import keeps the dependency one-way at module load time.
    from plasma_global.chemistry._data_manifest import load_chemistry as load

    return load(path, sources=sources)


__all__ = [
    "AMU_TO_KG",
    "ChemistryData",
    "CrossSectionData",
    "RateModelData",
    "ReactionData",
    "SpeciesData",
    "load_chemistry",
    "parse_equation",
]
