from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plasma_global.diagnostics.names import observable_id


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
class RateTableLookupDiagnostics:
    table_path: str
    grid_column: str
    lookup_mode: str
    lookup_value: float
    lookup_clipped_value: float
    axis_min: float
    axis_max: float
    lookup_clipped: bool
    lookup_clipped_low: bool
    lookup_clipped_high: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    backend: str = 'rate_table'

    def as_dict(self) -> dict[str, Any]:
        return {
            'backend': self.backend,
            'table_path': self.table_path,
            'grid_column': self.grid_column,
            'lookup_mode': self.lookup_mode,
            'lookup_value': self.lookup_value,
            'lookup_clipped_value': self.lookup_clipped_value,
            'axis_min': self.axis_min,
            'axis_max': self.axis_max,
            'lookup_clipped': self.lookup_clipped,
            'lookup_clipped_low': self.lookup_clipped_low,
            'lookup_clipped_high': self.lookup_clipped_high,
            'metadata': dict(self.metadata),
        }

    def to_observable_fields(self, zone_id: str) -> dict[str, float | str | int]:
        zone_key = observable_id(zone_id)
        return {
            f'rate_table_lookup_value_{zone_key}': float(self.lookup_value),
            f'rate_table_lookup_clipped_value_{zone_key}': float(self.lookup_clipped_value),
            f'rate_table_axis_min_{zone_key}': float(self.axis_min),
            f'rate_table_axis_max_{zone_key}': float(self.axis_max),
            f'rate_table_lookup_clipped_{zone_key}': int(self.lookup_clipped),
            f'rate_table_lookup_clipped_low_{zone_key}': int(self.lookup_clipped_low),
            f'rate_table_lookup_clipped_high_{zone_key}': int(self.lookup_clipped_high),
            f'rate_table_lookup_mode_{zone_key}': str(self.lookup_mode),
            f'rate_table_grid_column_{zone_key}': str(self.grid_column),
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __contains__(self, key: object) -> bool:
        return key in self.as_dict()

    def __iter__(self):
        return iter(self.as_dict())


@dataclass
class EEDFResult:
    rate_coefficients: dict[str, float]
    d_rate_d_mean_energy_eV: dict[str, float]
    transport: EEDFTransport
    diagnostics: RateTableLookupDiagnostics | None = None


class EEDFBackend:
    def prepare(self, mechanism: Any, chamber: Any, run_config: Any, resolved_paths: Any) -> None:
        self.mechanism = mechanism
        self.chamber = chamber
        self.run_config = run_config
        self.resolved_paths = resolved_paths

    def evaluate(self, request: EEDFRequest) -> EEDFResult:  # pragma: no cover - interface
        raise NotImplementedError
