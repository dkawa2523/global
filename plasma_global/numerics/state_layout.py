from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateSlice:
    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass
class StateLayout:
    slices: dict[str, StateSlice]
    size: int
    gas_index: dict[str, dict[str, int]] = field(default_factory=dict)
    electron_energy_index: dict[str, int] = field(default_factory=dict)
    gas_temperature_index: dict[str, int] = field(default_factory=dict)
    surface_index: dict[str, dict[str, int]] = field(default_factory=dict)
    inventory_index: dict[str, dict[str, int]] = field(default_factory=dict)
    film_index: dict[str, int] = field(default_factory=dict)
    gas_species_ids: list[str] = field(default_factory=list)
    zone_ids: list[str] = field(default_factory=list)

    def make_state(self) -> np.ndarray:
        return np.zeros(self.size, dtype=float)

    def slice(self, name: str) -> slice:
        s = self.slices[name]
        return slice(s.start, s.stop)
