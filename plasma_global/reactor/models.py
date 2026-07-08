from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Zone:
    zone_id: str
    description: str
    volume_m3: float
    pressure_Pa: float
    gas_temperature_K: float
    role: str = 'process'
    initial_densities_m3: dict[str, float] = field(default_factory=dict)


@dataclass
class Edge:
    edge_id: str
    from_zone: str
    to_zone: str
    conductance_m3_s: float
    notes: str = ''


@dataclass
class Surface:
    surface_id: str
    zone_id: str
    kind: str
    area_m2: float
    material: str
    temperature_K: float
    site_density_m2: float
    initial_coverages: dict[str, float] = field(default_factory=dict)
    initial_inventory: dict[str, float] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inlet:
    inlet_id: str
    zone_id: str
    flow_sccm: dict[str, float]
    temperature_K: float


@dataclass
class Pump:
    pump_id: str
    zone_id: str
    speed_m3_s: float


@dataclass
class PowerPort:
    port_id: str
    kind: str
    zone_id: str
    coupling_target: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChamberConfig:
    chamber_id: str
    description: str
    zones: list[Zone]
    edges: list[Edge]
    surfaces: list[Surface]
    gas_inlets: list[Inlet]
    pumps: list[Pump]
    power_ports: list[PowerPort]

    def __post_init__(self) -> None:
        self.zone_by_id = {z.zone_id: z for z in self.zones}
        self.surface_by_id = {s.surface_id: s for s in self.surfaces}
        self.inlet_by_id = {i.inlet_id: i for i in self.gas_inlets}
        self.power_port_by_id = {p.port_id: p for p in self.power_ports}
        self.surfaces_by_zone: dict[str, list[Surface]] = {z.zone_id: [] for z in self.zones}
        for surface in self.surfaces:
            self.surfaces_by_zone.setdefault(surface.zone_id, []).append(surface)


@dataclass
class RecipeStep:
    step_id: str
    t_start_s: float
    t_end_s: float
    gas_inlets: dict[str, dict[str, float]] = field(default_factory=dict)
    power_ports: dict[str, dict[str, Any]] = field(default_factory=dict)
    surface_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    imported_inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RecipeConfig:
    recipe_id: str
    description: str
    steps: list[RecipeStep]
