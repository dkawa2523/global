from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class EEDFTransport:
    mean_energy_eV: float
    mobility_m2_V_s: float
    diffusion_m2_s: float
    effective_field_Td: float
    lookup_mode: str

    def __post_init__(self) -> None:
        if self.lookup_mode not in {'mean_energy', 'field'}:
            raise ValueError("EEDFTransport.lookup_mode must be 'mean_energy' or 'field'")


@dataclass
class EEDFResult:
    rate_coefficients: dict[str, float]
    d_rate_d_mean_energy_eV: dict[str, float]
    transport: EEDFTransport


class EEDFBackend:
    def prepare(self, mechanism: Any, chamber: Any, run_config: Any, resolved_paths: Any) -> None:
        self.mechanism = mechanism
        self.chamber = chamber
        self.run_config = run_config
        self.resolved_paths = resolved_paths

    def evaluate(self, request: EEDFRequest) -> EEDFResult:  # pragma: no cover - interface
        raise NotImplementedError
