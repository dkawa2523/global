from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ZoneElectricalState:
    electron_density_m3: float
    mean_energy_eV: float
    positive_ion_density_m3: float
    dominant_ion_mass_kg: float
    pressure_Pa: float
    gas_temperature_K: float
    total_density_m3: float
    electron_mobility_m2_V_s: float | None = None


@dataclass
class SurfaceIED:
    ion_flux_m2_s: float
    mean_ion_energy_eV: float


@dataclass
class PowerRequest:
    time_s: float
    state_vector: Any
    recipe_step: Any
    chamber: Any
    zone_state: dict[str, ZoneElectricalState] = field(default_factory=dict)


@dataclass
class PowerResult:
    absorbed_power_W_by_zone: dict[str, float]
    port_power_W: dict[str, float] = field(default_factory=dict)
    self_bias_V: float = 0.0
    plasma_potential_V: float = 0.0
    zone_reduced_field_Td: dict[str, float] = field(default_factory=dict)
    surface_ied: dict[str, SurfaceIED] = field(default_factory=dict)


class ElectricalBackend:
    def prepare(self, chamber: Any, recipe: Any, run_config: Any, resolved_paths: Any) -> None:
        self.chamber = chamber
        self.recipe = recipe
        self.run_config = run_config
        self.resolved_paths = resolved_paths

    def evaluate(self, request: PowerRequest) -> PowerResult:  # pragma: no cover
        raise NotImplementedError
