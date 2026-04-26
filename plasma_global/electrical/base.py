from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PowerRequest:
    time_s: float
    state_vector: Any
    recipe_step: Any
    chamber: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerResult:
    absorbed_power_W_by_zone: dict[str, float]
    port_power_W: dict[str, float] = field(default_factory=dict)
    self_bias_V: float = 0.0
    plasma_potential_V: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ElectricalBackend:
    def prepare(self, chamber: Any, recipe: Any, run_config: Any) -> None:
        self.chamber = chamber
        self.recipe = recipe
        self.run_config = run_config

    def evaluate(self, request: PowerRequest) -> PowerResult:  # pragma: no cover
        raise NotImplementedError
