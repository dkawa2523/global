from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


AMU_TO_KG = 1.66053906660e-27
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23


@dataclass
class Species:
    canonical_id: str
    display_name: str
    phase: str
    charge: int
    mass_amu: float
    elements: dict[str, float]
    aliases: list[str] = field(default_factory=list)
    state_tags: set[str] = field(default_factory=set)
    zones: list[str] = field(default_factory=list)
    surfaces: list[str] = field(default_factory=list)

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU_TO_KG


@dataclass
class Reaction:
    reaction_id: str
    phase: str
    equation: str
    reactants: dict[str, float]
    products: dict[str, float]
    rate_model_key: str
    energy_model_key: str | None
    zone_filter: list[str]
    surface_filter: list[str]
    enabled: bool
    notes: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossSectionSpec:
    cross_section_id: str
    kind: str = 'inelastic'
    target_species: str | None = None
    collider_species: str = 'e'
    threshold_eV: float = 0.0
    energy_loss_eV: float | None = None
    prefactor_m3_s: float = 1.0e-15
    exponent: float = 0.5
    source: str | None = None
    file: str | None = None
    format: str | None = None
    selector: str | None = None
    dataset_index: int | None = None
    unit_energy: str = 'eV'
    unit_sigma: str = 'm2'
    metadata: dict[str, Any] = field(default_factory=dict)
    energy_eV: np.ndarray | None = None
    sigma_m2: np.ndarray | None = None

    def has_tabulated_data(self) -> bool:
        return self.energy_eV is not None and self.sigma_m2 is not None and len(self.energy_eV) >= 2

    def sigma_interp(self, energy_eV: np.ndarray) -> np.ndarray:
        if not self.has_tabulated_data():
            raise ValueError(f'Cross section {self.cross_section_id} has no tabulated data')
        return np.interp(
            energy_eV,
            self.energy_eV,
            self.sigma_m2,
            left=0.0,
            right=float(self.sigma_m2[-1]),
        )


@dataclass
class MechanismBundle:
    species: list[Species]
    gas_reactions: list[Reaction]
    surface_reactions: list[Reaction]
    rate_models: dict[str, dict[str, Any]]
    cross_sections: dict[str, CrossSectionSpec]
    aliases: dict[str, str]

    def __post_init__(self) -> None:
        self.species_by_id = {s.canonical_id: s for s in self.species}
        self.gas_species = [s for s in self.species if s.phase == 'gas']
        self.surface_species = [s for s in self.species if s.phase == 'surface']
        self.electron_species_id = next((s.canonical_id for s in self.gas_species if (s.charge == -1 and 'electron' in s.state_tags) or s.canonical_id == 'e'), None)
        if self.electron_species_id is None and any(s.canonical_id == 'e' for s in self.gas_species):
            self.electron_species_id = 'e'
        self.gas_state_species = [s for s in self.gas_species if s.canonical_id != self.electron_species_id]
        self.species_index = {s.canonical_id: i for i, s in enumerate(self.species)}
        self.gas_state_index = {s.canonical_id: i for i, s in enumerate(self.gas_state_species)}
        self.cross_section_ids = list(self.cross_sections)
        self.cross_sections_by_target: dict[str, list[CrossSectionSpec]] = {}
        for cs in self.cross_sections.values():
            if cs.target_species:
                self.cross_sections_by_target.setdefault(cs.target_species, []).append(cs)
        self.momentum_transfer_cross_sections = {
            cs_id: cs for cs_id, cs in self.cross_sections.items()
            if str(cs.kind).lower() in {'elastic', 'elastic_momentum', 'momentum_transfer', 'mt'}
        }

    def model(self, key: str) -> dict[str, Any]:
        if key not in self.rate_models:
            raise KeyError(f'Unknown rate/energy model: {key}')
        return self.rate_models[key]
