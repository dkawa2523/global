from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _PowerPortStep:
    power_ports: dict[str, Any]


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

    def with_power_ports(self, power_ports: dict[str, Any]) -> PowerRequest:
        return PowerRequest(
            time_s=self.time_s,
            state_vector=self.state_vector,
            recipe_step=_PowerPortStep(dict(power_ports)),
            chamber=self.chamber,
            zone_state=self.zone_state,
        )


def merged_power_port_config(chamber: Any, port_id: str, step_cfg: dict[str, Any] | None) -> tuple[Any, dict[str, Any]]:
    port = chamber.power_port_by_id[port_id]
    cfg = dict(getattr(port, 'parameters', {}) or {})
    cfg.update(step_cfg or {})
    return port, cfg


def iter_power_port_configs(request: PowerRequest):
    for port_id, step_cfg in request.recipe_step.power_ports.items():
        port, cfg = merged_power_port_config(request.chamber, port_id, step_cfg)
        yield port_id, port, cfg


def zone_value_map(chamber: Any, default: float = 0.0) -> dict[str, float]:
    return {z.zone_id: float(default) for z in chamber.zones}


def add_zone_value(values: dict[str, float], zone_id: str, value: float) -> float:
    number = float(value)
    values[zone_id] = values.get(zone_id, 0.0) + number
    return number


def set_zone_max(values: dict[str, float], zone_id: str, value: float) -> float:
    number = float(value)
    values[zone_id] = max(values.get(zone_id, 0.0), number)
    return number


def add_port_power(
    zone_power_W: dict[str, float],
    port_power_W: dict[str, float],
    *,
    port_id: str,
    zone_id: str,
    absorbed_power_W: float,
) -> float:
    absorbed = add_zone_value(zone_power_W, zone_id, absorbed_power_W)
    port_power_W[port_id] = absorbed
    return absorbed


@dataclass
class PowerResult:
    """Electrical backend result for one state/time evaluation."""

    absorbed_power_W_by_zone: dict[str, float]
    port_power_W: dict[str, float] = field(default_factory=dict)
    port_observables: dict[str, dict[str, float]] = field(default_factory=dict)
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
