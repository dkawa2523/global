from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from plasma_global.eedf.base import EEDFResult
from plasma_global.electrical.base import PowerResult


@dataclass
class CompiledGasReaction:
    reaction_id: str
    zones: list[str]
    reactant_gas: list[tuple[int, float, str]]
    electron_reactant_stoich: float
    delta_gas: list[tuple[int, float, str]]
    rate_model: dict[str, Any]
    energy_model: dict[str, Any] | None


@dataclass
class GasReactionTerm:
    zone_id: str
    reaction_id: str
    rate_m3_s: float
    species_changes: list[tuple[int, float, str]]
    electron_energy_loss_J_m3_s: float = 0.0


@dataclass
class IonWallLossTerm:
    zone_id: str
    species_index: int
    species_id: str
    loss_m3_s: float
    electron_energy_loss_J_m3_s: float


@dataclass
class CompiledSurfaceReaction:
    reaction_id: str
    zone_id: str
    surface_id: str
    gas_reactants: list[tuple[int, float, str]]
    surface_reactants: list[tuple[int, float, str]]
    delta_gas: list[tuple[int, float, str]]
    delta_surface: list[tuple[int, float, str]]
    area_over_volume: float
    area_m2: float
    site_density_m2: float
    rate_model: dict[str, Any]
    inventory_idx: int | None
    film_factor: float


@dataclass
class SurfaceRateEvaluation:
    rate_m2_s: float
    d_gas: dict[int, float] = field(default_factory=dict)
    d_surface: dict[int, float] = field(default_factory=dict)
    dTg: float = 0.0


@dataclass
class SurfaceRateContext:
    reaction: CompiledSurfaceReaction
    gas_row: np.ndarray
    gas_temperature_K: float
    state: np.ndarray
    step: Any
    coupled: CoupledPlasmaEvaluation
    surface_temperature_K: float
    ion_energy_eV: float
    positive_ion_density_m3: float
    ion_flux_m2_s: float


@dataclass
class CoupledPlasmaEvaluation:
    gas: np.ndarray
    electron_energy: np.ndarray
    gas_temperature: np.ndarray
    power: PowerResult
    eedf_by_zone: dict[str, EEDFResult]
    ne_by_zone: dict[str, float]
    mean_e_by_zone: dict[str, float]
    pos_by_zone: dict[str, float]
    ion_mass_by_zone: dict[str, float]
    pressure_by_zone: dict[str, float]
    total_density_by_zone: dict[str, float]
