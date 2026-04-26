from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EEDFRequest:
    time_s: float
    zone_id: str
    composition: dict[str, float]
    electron_density_m3: float
    mean_energy_eV: float
    reduced_field_Td: float | None
    gas_temperature_K: float
    pressure_Pa: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EEDFResult:
    rate_coefficients: dict[str, float]
    d_rate_d_mean_energy_eV: dict[str, float]
    transport: dict[str, float] = field(default_factory=dict)


class EEDFBackend:
    def prepare(self, mechanism: Any, chamber: Any, run_config: Any) -> None:
        self.mechanism = mechanism
        self.chamber = chamber
        self.run_config = run_config

    def evaluate(self, request: EEDFRequest) -> EEDFResult:  # pragma: no cover - interface
        raise NotImplementedError
