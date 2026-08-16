"""Translate schema-v2 per-step power commands to schema v3."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_common import (
    external_file,
    first,
    record_extra_keys,
)
from plasma_global.input._migrate_v2_power_models import (
    _DC_STEP_KEYS,
    _DC_VOLTAGE_KEYS,
    _DELIVERED_POWER_KEYS,
    _EXPERIMENTAL_STEP_KEYS,
    _EXTERNAL_TABLE_STEP_KEYS,
    _PRESCRIBED_POWER_KEYS,
    _RF_ENVELOPE_COMMAND_KEYS,
    _normalized_bounds_policy,
)


def match_experimental_setpoint_to_model(
    command: dict[str, Any], model: Mapping[str, Any], prefix: str
) -> dict[str, Any]:
    """Normalize legacy zero/off commands to the reactor's static control mode."""

    if model.get("kind") not in {
        "experimental.rf_envelope",
        "experimental.ccp",
    }:
        return command
    control = str(model.get("control", "absorbed_power"))
    selected, incompatible = (
        ("voltage_rms_V", "absorbed_power_W")
        if control == "voltage"
        else ("absorbed_power_W", "voltage_rms_V")
    )
    if incompatible not in command:
        return command
    value = float(command[incompatible])
    if selected in command:
        if value == 0.0:
            command.pop(incompatible)
        return command
    if value != 0.0:
        raise MigrationError(
            f"{prefix} provides {incompatible}={value:g}, but its reactor model "
            f"uses {control!r} control"
        )
    command.pop(incompatible)
    command[selected] = 0.0
    return command


def _waveform(
    values: Mapping[str, Any], prefix: str, unused: set[str], warnings: list[str]
) -> tuple[dict[str, Any], bool, set[str]]:
    consumed = {"waveform"}
    raw_value = values.get("waveform", "cw")
    raw = (
        ("continuous" if raw_value else "off")
        if isinstance(raw_value, bool)
        else str(raw_value).strip().lower()
    )
    if raw in {"cw", "continuous", "none"}:
        return {"kind": "continuous"}, False, consumed
    if raw == "off":
        return {"kind": "continuous"}, True, consumed
    if raw in {"pulsed_square", "square", "square_pulse"}:
        consumed.update({"duty_cycle", "repetition_Hz"})
        return (
            {
                "kind": "square_pulse",
                "duty_cycle": float(values.get("duty_cycle", 0.5)),
                "repetition_Hz": float(values.get("repetition_Hz", 1.0)),
            },
            False,
            consumed,
        )
    unused.add(f"{prefix}.waveform")
    warnings.append(f"{prefix}.waveform={raw!r} was treated as a continuous waveform")
    return {"kind": "continuous"}, False, consumed


def _numeric_setpoint(value: Any, *, off: bool) -> float | None:
    if off:
        return 0.0
    return None if value is None else float(value)


def _prescribed_power_command(
    raw: Mapping[str, Any],
    *,
    waveform: Mapping[str, Any],
    is_off: bool,
) -> tuple[dict[str, Any], set[str]]:
    command: dict[str, Any] = {
        "kind": "prescribed_power",
        "waveform": dict(waveform),
    }
    power = _numeric_setpoint(first(raw, _PRESCRIBED_POWER_KEYS), off=is_off)
    if power is not None:
        command["absorbed_power_W"] = power
    return command, set(_PRESCRIBED_POWER_KEYS)


def _dc_series_command(
    raw: Mapping[str, Any],
    *,
    waveform: Mapping[str, Any],
    is_off: bool,
    prefix: str,
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], set[str]]:
    command: dict[str, Any] = {"kind": "dc_series"}
    consumed = set(_DC_STEP_KEYS)
    voltage = first(raw, _DC_VOLTAGE_KEYS)
    nested = raw.get("voltage")
    if isinstance(nested, Mapping):
        consumed.add("voltage")
        voltage = first(
            nested,
            ("high_V", "on_V", *_DC_VOLTAGE_KEYS),
            voltage,
        )
        command["off_voltage_V"] = float(first(nested, ("low_V", "off_V"), 0.0))
        nested_waveform = dict(nested)
        if "frequency_Hz" in nested_waveform and "repetition_Hz" not in nested_waveform:
            nested_waveform["repetition_Hz"] = nested_waveform["frequency_Hz"]
        waveform, nested_off, _ = _waveform(
            nested_waveform, f"{prefix}.voltage", unused, warnings
        )
        is_off = is_off or nested_off
    voltage = _numeric_setpoint(voltage, off=is_off)
    if voltage is not None:
        command["source_voltage_V"] = voltage
    command["waveform"] = dict(waveform)
    return command, consumed


def _external_table_command(
    raw: Mapping[str, Any],
    *,
    base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
) -> tuple[dict[str, Any], set[str]]:
    command: dict[str, Any] = {"kind": "external_table"}
    file = external_file(
        raw,
        base_dir=base_dir,
        external_inputs=external_inputs,
        used_external=used_external,
    )
    if file is not None:
        command["file"] = file
    for key in (
        "power_scale",
        "voltage_scale",
        "current_scale",
        "time_offset_s",
        "gap_m",
        "total_density_m3",
        "plasma_potential_V",
    ):
        if raw.get(key) is not None:
            command[key] = float(raw[key])
    if raw.get("interpolation") is not None:
        command["interpolation"] = str(raw["interpolation"]).lower()
    legacy_bounds = raw.get("bounds_policy", raw.get("hold"))
    if legacy_bounds is not None:
        command["bounds_policy"] = _normalized_bounds_policy(legacy_bounds)
    return command, set(_EXTERNAL_TABLE_STEP_KEYS)


def _experimental_power_command(
    raw: Mapping[str, Any],
    *,
    kind: str,
    control: str,
    waveform: Mapping[str, Any],
    is_off: bool,
) -> tuple[dict[str, Any], set[str]]:
    command: dict[str, Any] = {"kind": kind, "waveform": dict(waveform)}
    power = first(raw, ("absorbed_power_W", "power_W", "value_W"))
    voltage = first(raw, ("voltage_rms_V", "voltage_V", "value_V"))
    if power is None and voltage is None and raw.get("value") is not None:
        if control == "voltage":
            voltage = raw["value"]
        else:
            power = raw["value"]
    power = _numeric_setpoint(power, off=is_off and control == "absorbed_power")
    voltage = _numeric_setpoint(voltage, off=is_off and control == "voltage")
    if power is not None:
        command["absorbed_power_W"] = power
    if voltage is not None:
        command["voltage_rms_V"] = voltage
    consumed = set(_EXPERIMENTAL_STEP_KEYS)
    if kind == "experimental.rf_envelope":
        consumed.update(_RF_ENVELOPE_COMMAND_KEYS)
    return command, consumed


def _delivered_power_command(
    raw: Mapping[str, Any],
    *,
    kind: str,
    waveform: Mapping[str, Any],
    is_off: bool,
) -> tuple[dict[str, Any], set[str]]:
    command: dict[str, Any] = {"kind": kind, "waveform": dict(waveform)}
    power = _numeric_setpoint(
        first(raw, ("delivered_power_W", "power_W", "value_W", "value")),
        off=is_off,
    )
    if power is not None:
        command["delivered_power_W"] = power
    return command, set(_DELIVERED_POWER_KEYS)


def power_command(
    *,
    values: Mapping[str, Any],
    kind: str,
    base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
    prefix: str,
    unused: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    raw = dict(values)
    waveform, is_off, waveform_keys = (
        (dict[str, Any](), False, set[str]())
        if kind == "external_table"
        else _waveform(raw, prefix, unused, warnings)
    )
    consumed = set(waveform_keys) | {"mode", "control_mode", "zone_id"}
    if "zone_id" in raw:
        unused.add(f"{prefix}.zone_id")

    mode = str(first(raw, ("control_mode", "mode"), "absorbed_power"))
    control = "voltage" if "voltage" in mode.lower() else "absorbed_power"
    if kind == "prescribed_power":
        command, kind_keys = _prescribed_power_command(
            raw, waveform=waveform, is_off=is_off
        )
    elif kind == "dc_series":
        command, kind_keys = _dc_series_command(
            raw,
            waveform=waveform,
            is_off=is_off,
            prefix=prefix,
            unused=unused,
            warnings=warnings,
        )
    elif kind == "external_table":
        command, kind_keys = _external_table_command(
            raw,
            base_dir=base_dir,
            external_inputs=external_inputs,
            used_external=used_external,
        )
    elif kind in {"experimental.rf_envelope", "experimental.ccp"}:
        command, kind_keys = _experimental_power_command(
            raw,
            kind=kind,
            control=control,
            waveform=waveform,
            is_off=is_off,
        )
    else:
        command, kind_keys = _delivered_power_command(
            raw, kind=kind, waveform=waveform, is_off=is_off
        )

    consumed.update(kind_keys)
    record_extra_keys(raw, consumed, prefix, unused)
    return command
