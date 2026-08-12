"""Small, input-agnostic domain objects used by the compiled solver."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, SupportsFloat

from plasma_global.core.exceptions import ModelConfigurationError
from plasma_global.models.power import CompiledPowerCommand

if TYPE_CHECKING:
    from plasma_global.core.transport import SegmentTransport


_MIN_BDF_RTOL = 100.0 * math.ulp(1.0)


def _frozen_float_mapping(
    values: Mapping[str, float], *, field_name: str
) -> Mapping[str, float]:
    converted = {str(key): float(value) for key, value in values.items()}
    if any(not math.isfinite(value) for value in converted.values()):
        raise ModelConfigurationError(f"{field_name} must contain only finite values")
    return MappingProxyType(converted)


@dataclass(frozen=True)
class Zone:
    """One well-mixed control volume."""

    zone_id: str
    volume_m3: float
    gas_temperature_K: float = 300.0

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ModelConfigurationError("zone_id must not be empty")
        if not math.isfinite(self.volume_m3) or self.volume_m3 <= 0.0:
            raise ModelConfigurationError(
                f"Zone {self.zone_id!r} volume_m3 must be positive"
            )
        if not math.isfinite(self.gas_temperature_K) or self.gas_temperature_K <= 0.0:
            raise ModelConfigurationError(
                f"Zone {self.zone_id!r} gas_temperature_K must be positive"
            )


@dataclass(frozen=True)
class RecipeSegment:
    """A smooth forcing interval whose commands are bound to one RHS."""

    segment_id: str
    start_s: float
    end_s: float
    absorbed_power_W_by_zone: Mapping[str, float] = field(default_factory=dict)
    reduced_field_Td_by_zone: Mapping[str, float] = field(default_factory=dict)
    surface_temperature_K_by_surface: Mapping[str, float] = field(default_factory=dict)
    wall_temperature_K_by_zone: Mapping[str, float] = field(default_factory=dict)
    prescribed_electron_density_m3_by_zone: Mapping[str, float] = field(
        default_factory=dict
    )
    transport: SegmentTransport | None = None
    port_commands: Mapping[str, CompiledPowerCommand] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ModelConfigurationError("segment_id must not be empty")
        if (
            not math.isfinite(self.start_s)
            or not math.isfinite(self.end_s)
            or self.end_s <= self.start_s
        ):
            raise ModelConfigurationError(
                f"Recipe segment {self.segment_id!r} must have end_s > start_s"
            )
        power = _frozen_float_mapping(
            self.absorbed_power_W_by_zone, field_name="absorbed_power_W_by_zone"
        )
        field_values = _frozen_float_mapping(
            self.reduced_field_Td_by_zone, field_name="reduced_field_Td_by_zone"
        )
        surface_temperatures = _frozen_float_mapping(
            self.surface_temperature_K_by_surface,
            field_name="surface_temperature_K_by_surface",
        )
        wall_temperatures = _frozen_float_mapping(
            self.wall_temperature_K_by_zone,
            field_name="wall_temperature_K_by_zone",
        )
        electron_densities = _frozen_float_mapping(
            self.prescribed_electron_density_m3_by_zone,
            field_name="prescribed_electron_density_m3_by_zone",
        )
        if any(value < 0.0 for value in power.values()):
            raise ModelConfigurationError(
                f"Recipe segment {self.segment_id!r} contains negative absorbed power"
            )
        if any(value < 0.0 for value in field_values.values()):
            raise ModelConfigurationError(
                f"Recipe segment {self.segment_id!r} contains negative reduced field"
            )
        if any(value <= 0.0 for value in surface_temperatures.values()):
            raise ModelConfigurationError(
                f"Recipe segment {self.segment_id!r} contains nonpositive surface temperature"
            )
        if any(value <= 0.0 for value in wall_temperatures.values()):
            raise ModelConfigurationError(
                f"Recipe segment {self.segment_id!r} contains nonpositive wall temperature"
            )
        if any(value < 0.0 for value in electron_densities.values()):
            raise ModelConfigurationError(
                f"Recipe segment {self.segment_id!r} contains negative electron density"
            )
        object.__setattr__(self, "absorbed_power_W_by_zone", power)
        object.__setattr__(self, "reduced_field_Td_by_zone", field_values)
        object.__setattr__(
            self,
            "surface_temperature_K_by_surface",
            surface_temperatures,
        )
        object.__setattr__(self, "wall_temperature_K_by_zone", wall_temperatures)
        object.__setattr__(
            self,
            "prescribed_electron_density_m3_by_zone",
            electron_densities,
        )
        commands = {str(key): value for key, value in self.port_commands.items()}
        if any(not key for key in commands):
            raise ModelConfigurationError(
                "port_commands must not contain empty port IDs"
            )
        if any(
            not isinstance(command, CompiledPowerCommand)
            for command in commands.values()
        ):
            raise ModelConfigurationError(
                "port_commands must contain only compiled power commands"
            )
        object.__setattr__(self, "port_commands", MappingProxyType(commands))


@dataclass(frozen=True)
class InitialState:
    """Physical initial data, before conversion to a flat ODE vector."""

    densities_m3_by_zone: Mapping[str, Mapping[str, float]]
    mean_energy_eV_by_zone: Mapping[str, float] = field(default_factory=dict)
    gas_temperature_K_by_zone: Mapping[str, float] = field(default_factory=dict)
    surface_coverages: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        density_data: dict[str, Mapping[str, float]] = {}
        for zone_id, values in self.densities_m3_by_zone.items():
            frozen = _frozen_float_mapping(
                values, field_name=f"densities_m3_by_zone[{zone_id!r}]"
            )
            if any(value < 0.0 for value in frozen.values()):
                raise ModelConfigurationError(
                    f"Initial densities in zone {zone_id!r} must be non-negative"
                )
            density_data[str(zone_id)] = frozen
        energies = _frozen_float_mapping(
            self.mean_energy_eV_by_zone, field_name="mean_energy_eV_by_zone"
        )
        if any(value < 0.0 for value in energies.values()):
            raise ModelConfigurationError(
                "Initial mean electron energies must be non-negative"
            )
        gas_temperatures = _frozen_float_mapping(
            self.gas_temperature_K_by_zone,
            field_name="gas_temperature_K_by_zone",
        )
        if any(value <= 0.0 for value in gas_temperatures.values()):
            raise ModelConfigurationError("Initial gas temperatures must be positive")
        coverages: dict[str, Mapping[str, float]] = {}
        for surface_id, values in self.surface_coverages.items():
            frozen = _frozen_float_mapping(
                values, field_name=f"surface_coverages[{surface_id!r}]"
            )
            if any(value < 0.0 for value in frozen.values()):
                raise ModelConfigurationError(
                    f"Initial coverages on surface {surface_id!r} must be nonnegative"
                )
            coverages[str(surface_id)] = frozen
        object.__setattr__(self, "densities_m3_by_zone", MappingProxyType(density_data))
        object.__setattr__(self, "mean_energy_eV_by_zone", energies)
        object.__setattr__(self, "gas_temperature_K_by_zone", gas_temperatures)
        object.__setattr__(self, "surface_coverages", MappingProxyType(coverages))


def _validate_solver_tolerances(rtol: float, atol: float) -> None:
    if not math.isfinite(rtol) or not _MIN_BDF_RTOL <= rtol < 1.0:
        raise ModelConfigurationError(
            f"rtol must be finite and satisfy {_MIN_BDF_RTOL:.16g} <= rtol < 1"
        )
    if not math.isfinite(atol) or atol <= 0.0:
        raise ModelConfigurationError("atol must be positive")


def _validate_step_controls(
    first_step_s: float | None,
    max_step_s: float | None,
    sample_interval_s: float | None,
) -> None:
    for name, value in (
        ("first_step_s", first_step_s),
        ("max_step_s", max_step_s),
        ("sample_interval_s", sample_interval_s),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ModelConfigurationError(f"{name} must be positive when provided")
    if (
        first_step_s is not None
        and max_step_s is not None
        and first_step_s > max_step_s
    ):
        raise ModelConfigurationError("first_step_s must not exceed max_step_s")


def _normalized_save_times(
    values: tuple[SupportsFloat, ...] | None,
) -> tuple[float, ...] | None:
    if values is None:
        return None
    save_at = tuple(float(value) for value in values)
    if not save_at:
        raise ModelConfigurationError("save_at_s must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in save_at):
        raise ModelConfigurationError("save_at_s must contain finite nonnegative times")
    if any(right <= left for left, right in pairwise(save_at)):
        raise ModelConfigurationError("save_at_s must be strictly increasing")
    return save_at


def _validate_quasi_steady_controls(
    threshold_s_inv: float | None, min_time_s: float
) -> None:
    if threshold_s_inv is not None and (
        not math.isfinite(threshold_s_inv) or threshold_s_inv < 0.0
    ):
        raise ModelConfigurationError(
            "experimental quasi-steady threshold must be finite and nonnegative"
        )
    if not math.isfinite(min_time_s) or min_time_s < 0.0:
        raise ModelConfigurationError(
            "experimental quasi-steady min_time_s must be finite and nonnegative"
        )


@dataclass(frozen=True)
class SolverSettings:
    """Numerical controls for sequential BDF integration.

    ``atol`` is scalar and dimensionless; the solver applies it to its scaled
    state and derives component-wise physical domain tolerances from that same
    scale.
    """

    rtol: float = 1.0e-6
    atol: float = 1.0e-12
    first_step_s: float | None = None
    max_step_s: float | None = None
    sample_interval_s: float | None = None
    save_at_s: tuple[float, ...] | None = None
    experimental_quasi_steady_threshold_s_inv: float | None = None
    experimental_quasi_steady_min_time_s: float = 0.0

    def __post_init__(self) -> None:
        _validate_solver_tolerances(self.rtol, self.atol)
        _validate_step_controls(
            self.first_step_s,
            self.max_step_s,
            self.sample_interval_s,
        )
        if self.sample_interval_s is not None and self.save_at_s is not None:
            raise ModelConfigurationError(
                "sample_interval_s and save_at_s are mutually exclusive"
            )
        object.__setattr__(self, "save_at_s", _normalized_save_times(self.save_at_s))
        _validate_quasi_steady_controls(
            self.experimental_quasi_steady_threshold_s_inv,
            self.experimental_quasi_steady_min_time_s,
        )


__all__ = [
    "InitialState",
    "RecipeSegment",
    "SolverSettings",
    "Zone",
]
