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
class ElectricalPortSnapshot:
    """Dict-like backend snapshot for postprocessed electrical observables."""

    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __contains__(self, key: object) -> bool:
        return key in self.values

    def items(self):
        return self.values.items()


@dataclass
class PowerResult:
    """Electrical backend result for one state/time evaluation.

    The solver-facing payload is `absorbed_power_W_by_zone`. Other fields are
    optional backend outputs for EEDF field lookup, surface/wall coupling, and
    postprocessed observables.
    """

    absorbed_power_W_by_zone: dict[str, float]
    port_power_W: dict[str, float] = field(default_factory=dict)
    port_observables: dict[str, ElectricalPortSnapshot] = field(default_factory=dict)
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
