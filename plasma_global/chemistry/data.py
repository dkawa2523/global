"""Public data contract for canonical schema-v3 chemistry.

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
from typing import Any

import numpy as np

from plasma_global.errors import ChemistryError

AMU_TO_KG = 1.66053906660e-27


def _empty_metadata() -> Mapping[str, Any]:
    return MappingProxyType({})


def _resolved_reaction_electron_energy_transfer(
    reaction: ReactionData,
) -> float | None:
    """Return signed electron gain, migrating legacy positive loss to a sink."""

    if reaction.electron_energy_transfer_eV is not None:
        return reaction.electron_energy_transfer_eV
    if reaction.energy_loss_eV is None:
        return None
    return -reaction.energy_loss_eV


def _resolved_cross_section_electron_energy_transfer(
    cross_section: CrossSectionData,
) -> float:
    """Return signed electron gain, migrating legacy positive loss to a sink."""

    if cross_section.electron_energy_transfer_eV is not None:
        return cross_section.electron_energy_transfer_eV
    if cross_section.energy_loss_eV is None:
        raise ChemistryError(
            f"cross section {cross_section.id!r} has no electron-energy transfer"
        )
    return -cross_section.energy_loss_eV


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

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU_TO_KG


@dataclass(frozen=True, slots=True)
class ReactionData:
    id: str
    reactants: Mapping[str, float]
    products: Mapping[str, float]
    rate_model: str | None
    energy_loss_eV: float | None
    gas_heating_eV: float = 0.0
    zones: tuple[str, ...] = ()
    surfaces: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    electron_energy_transfer_eV: float | None = None

    resolved_electron_energy_transfer_eV = property(
        _resolved_reaction_electron_energy_transfer
    )


@dataclass(frozen=True, slots=True)
class RateModelData:
    id: str
    kind: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CrossSectionData:
    id: str
    kind: str
    target: str
    threshold_eV: float
    energy_loss_eV: float | None
    energy_eV: np.ndarray
    sigma_m2: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    electron_energy_transfer_eV: float | None = None

    resolved_electron_energy_transfer_eV = property(
        _resolved_cross_section_electron_energy_transfer
    )


@dataclass(frozen=True, slots=True)
class ChemistryData:
    source: Path
    species: tuple[SpeciesData, ...]
    gas_reactions: tuple[ReactionData, ...]
    boundary_reactions: tuple[ReactionData, ...]
    surface_reactions: tuple[ReactionData, ...]
    rate_models: Mapping[str, RateModelData]
    cross_sections: Mapping[str, CrossSectionData]
    experimental: Mapping[str, Any] = field(default_factory=_empty_metadata)
    provenance: Mapping[str, Any] = field(default_factory=_empty_metadata)
    source_files: tuple[Path, ...] = ()


_TERM = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s+)?([^\s].*?)\s*$")


def parse_equation(equation: str) -> tuple[Mapping[str, float], Mapping[str, float]]:
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
    reactants, products = side(lhs), side(rhs)
    if not reactants:
        raise ChemistryError(f"reaction has no reactants: {equation!r}")
    return reactants, products


def load_chemistry(path: str | Path) -> ChemistryData:
    """Load a canonical v3 chemistry manifest and all referenced SI data."""

    # The private loader imports these public data classes.  Deferring this
    # implementation import keeps the dependency one-way at module load time.
    from plasma_global.chemistry._data_manifest import load_chemistry as load

    return load(path)


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
