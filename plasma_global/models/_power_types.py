"""Pure helpers shared by the public power value objects."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal, TypeVar

import numpy as np

from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.external_table import (
    ExternalTableBinding,
    ExternalTableSample,
)

_K = TypeVar("_K")
_V = TypeVar("_V")


def _empty_observables() -> Mapping[str, float]:
    return MappingProxyType({})


def validate_power_state(
    electron_density_m3: float,
    neutral_density_m3: float,
    electron_temperature_eV: float,
    electron_mobility_m2_V_s: float | None,
) -> None:
    values = (
        electron_density_m3,
        neutral_density_m3,
        electron_temperature_eV,
    )
    if not np.isfinite(values).all() or min(values) < 0.0:
        raise ModelDomainError("power state values must be finite and nonnegative")
    if electron_mobility_m2_V_s is not None and (
        not math.isfinite(electron_mobility_m2_V_s) or electron_mobility_m2_V_s <= 0.0
    ):
        raise ModelDomainError("electron mobility must be finite and positive")


def validate_compiled_power_command(
    kind: Literal["off", "power", "voltage", "external_table"],
    power_W: float | None,
    voltage_V: float | None,
    external_table: ExternalTableBinding | ExternalTableSample | None,
    reduced_field_capability: bool | None,
) -> None:
    _validate_reduced_field_capability(kind, reduced_field_capability)
    validator = _COMMAND_VALIDATORS.get(kind)
    if validator is None:
        raise CaseValidationError(f"unsupported compiled power command {kind!r}")
    validator(power_W, voltage_V, external_table)


def _validate_reduced_field_capability(
    kind: str, reduced_field_capability: bool | None
) -> None:
    if reduced_field_capability is not None and (
        kind != "off" or not isinstance(reduced_field_capability, bool)
    ):
        raise CaseValidationError(
            "reduced_field_capability is a boolean annotation for off commands"
        )


def _validate_off_command(
    power_W: float | None,
    voltage_V: float | None,
    external_table: ExternalTableBinding | ExternalTableSample | None,
) -> None:
    if any(value is not None for value in (power_W, voltage_V, external_table)):
        raise CaseValidationError("off power commands must not carry a value")


def _validate_power_command(
    power_W: float | None,
    voltage_V: float | None,
    external_table: ExternalTableBinding | ExternalTableSample | None,
) -> None:
    if (
        power_W is None
        or not math.isfinite(power_W)
        or power_W < 0.0
        or voltage_V is not None
        or external_table is not None
    ):
        raise CaseValidationError(
            "compiled power commands require one finite nonnegative power_W"
        )


def _validate_voltage_command(
    power_W: float | None,
    voltage_V: float | None,
    external_table: ExternalTableBinding | ExternalTableSample | None,
) -> None:
    if (
        voltage_V is None
        or not math.isfinite(voltage_V)
        or power_W is not None
        or external_table is not None
    ):
        raise CaseValidationError(
            "compiled voltage commands require one finite voltage_V"
        )


def _validate_external_table_command(
    power_W: float | None,
    voltage_V: float | None,
    external_table: ExternalTableBinding | ExternalTableSample | None,
) -> None:
    if (
        not isinstance(external_table, (ExternalTableBinding, ExternalTableSample))
        or power_W is not None
        or voltage_V is not None
    ):
        raise CaseValidationError(
            "compiled external-table commands require one table binding"
        )


_CommandValidator = Callable[
    [
        float | None,
        float | None,
        ExternalTableBinding | ExternalTableSample | None,
    ],
    None,
]
_COMMAND_VALIDATORS: Mapping[str, _CommandValidator] = MappingProxyType(
    {
        "off": _validate_off_command,
        "power": _validate_power_command,
        "voltage": _validate_voltage_command,
        "external_table": _validate_external_table_command,
    }
)


def validate_power_port_result(
    electron_power_W: float,
    gas_power_W: float,
    reduced_field_Td: float | None,
) -> None:
    if (
        min(electron_power_W, gas_power_W) < 0.0
        or not np.isfinite([electron_power_W, gas_power_W]).all()
    ):
        raise ModelDomainError("power partitions must be finite and nonnegative")
    if reduced_field_Td is not None and (
        not math.isfinite(reduced_field_Td) or reduced_field_Td < 0.0
    ):
        raise ModelDomainError("reduced_field_Td must be finite and nonnegative")


def frozen_mapping(values: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(values))


def finite_observables(values: Mapping[str, float]) -> Mapping[str, float]:
    """Freeze diagnostic scalars and reject nonfinite artifact content early."""

    normalized = dict(values)
    if any(not name for name in normalized):
        raise ModelDomainError("power observable names must not be empty")
    if not np.isfinite(tuple(normalized.values())).all():
        raise ModelDomainError("power observables must be finite")
    return MappingProxyType(normalized)
