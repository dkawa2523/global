"""Translate schema-v2 reactor power-port models to schema v3."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_common import (
    external_file,
    first,
    record_extra_keys,
)

_PRESCRIBED_POWER_KEYS = ("absorbed_power_W", "power_W", "value_W", "value")
_DC_VOLTAGE_KEYS = ("source_voltage_V", "voltage_V", "value_V", "value")
_DC_STEP_KEYS = {
    *_DC_VOLTAGE_KEYS,
    "ballast_resistance_ohm",
    "series_resistance_ohm",
    "gap_m",
    "electrode_gap_m",
    "electrode_area_m2",
    "area_m2",
    "power_absorption_fraction",
    "electron_mobility_m2_V_s",
}
_DC_STATIC_KEYS = set(_DC_STEP_KEYS)
_EXTERNAL_TABLE_STEP_KEYS = {
    "file",
    "file_key",
    "power_scale",
    "voltage_scale",
    "current_scale",
    "time_offset_s",
    "interpolation",
    "hold",
    "bounds_policy",
    "gap_m",
    "total_density_m3",
    "plasma_potential_V",
}
_EXTERNAL_TABLE_STATIC_KEYS = _EXTERNAL_TABLE_STEP_KEYS - {"time_offset_s"}
_EXPERIMENTAL_STEP_KEYS = {
    "frequency_Hz",
    "carrier_frequency_Hz",
    "absorbed_power_W",
    "power_W",
    "value_W",
    "voltage_rms_V",
    "voltage_V",
    "value_V",
    "value",
}
_EXPERIMENTAL_STATIC_KEYS = _EXPERIMENTAL_STEP_KEYS - {"value"} | {
    "control_mode",
    "mode",
}
_RF_ENVELOPE_COMMAND_KEYS = {
    "role",
    "coupling_efficiency",
    "reduced_field_Td",
    "effective_impedance_ohm",
    "base_reduced_field_Td",
    "reduced_field_per_sqrt_W_Td",
    "reduced_field_per_sqrt_W",
    "self_bias_fraction",
    "plasma_potential_offset_V",
    "plasma_potential_per_sqrt_W",
}
_RF_ENVELOPE_STATIC_KEYS = _EXPERIMENTAL_STATIC_KEYS | _RF_ENVELOPE_COMMAND_KEYS
_DELIVERED_POWER_KEYS = {
    "frequency_Hz",
    "carrier_frequency_Hz",
    "delivered_power_W",
    "power_W",
    "value_W",
    "value",
}


def _power_kind(backend: str, legacy_port_kind: str) -> str:
    backend = backend.strip().lower()
    port_kind = legacy_port_kind.strip().lower()
    if backend == "direct_power":
        return "prescribed_power"
    if backend == "dc_series_circuit":
        return "dc_series"
    if backend == "external_circuit_table":
        return "external_table"
    if backend == "rf_envelope":
        return "experimental.rf_envelope"
    if backend == "ccp":
        return "experimental.ccp"
    if backend == "icp":
        if "icp" in port_kind or "source" in port_kind:
            return "experimental.icp"
        return "experimental.ccp"
    raise MigrationError(f"unsupported v2 electrical backend {backend!r}")


def _commands_for_port(recipe: Any, port_id: str) -> list[dict[str, Any]]:
    return [
        dict(step.power_ports.get(port_id, {}) or {})
        for step in recipe.steps
        if port_id in step.power_ports
    ]


def _from_static_or_commands(
    static: Mapping[str, Any],
    commands: Iterable[Mapping[str, Any]],
    names: tuple[str, ...],
    default: Any = None,
) -> Any:
    value = first(static, names)
    if value is not None:
        return value
    for command in commands:
        value = first(command, names)
        if value is not None:
            return value
    return default


def _power_command_magnitude(command: Mapping[str, Any]) -> float:
    value = first(
        command,
        (
            "absorbed_power_W",
            "delivered_power_W",
            "power_W",
            "value_W",
            "voltage_rms_V",
            "voltage_V",
            "value_V",
            "value",
        ),
        0.0,
    )
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _representative_power_command(
    commands: list[dict[str, Any]],
) -> Mapping[str, Any]:
    """Choose the highest-setpoint step when legacy static data lived in recipes."""

    return max(commands, key=_power_command_magnitude, default={})


def _normalized_rf_role(value: object) -> str:
    return "bias" if "bias" in str(value).strip().lower() else "source"


def _normalized_bounds_policy(value: object) -> str:
    return "hold" if str(value).lower() in {"edge", "clip", "hold"} else "error"


def _prescribed_power_model(
    static: Mapping[str, Any], *, kind: str, gas_fraction: float
) -> tuple[dict[str, Any], set[str]]:
    model: dict[str, Any] = {
        "kind": kind,
        "electron_fraction": 1.0 - gas_fraction,
        "gas_fraction": gas_fraction,
    }
    power = first(static, _PRESCRIBED_POWER_KEYS)
    if power is not None:
        model["default_absorbed_power_W"] = float(power)
    return model, set(_PRESCRIBED_POWER_KEYS)


def _dc_series_model(
    static: Mapping[str, Any],
    commands: list[dict[str, Any]],
    *,
    kind: str,
    prefix: str,
) -> tuple[dict[str, Any], set[str]]:
    ballast = _from_static_or_commands(
        static, commands, ("ballast_resistance_ohm", "series_resistance_ohm")
    )
    gap = _from_static_or_commands(static, commands, ("gap_m", "electrode_gap_m"))
    area = _from_static_or_commands(static, commands, ("electrode_area_m2", "area_m2"))
    if ballast is None or gap is None or area is None:
        raise MigrationError(
            f"{prefix}: dc_series requires ballast resistance, gap, and electrode area"
        )
    model: dict[str, Any] = {
        "kind": kind,
        "ballast_resistance_ohm": float(ballast),
        "gap_m": float(gap),
        "electrode_area_m2": float(area),
        "power_absorption_fraction": float(
            _from_static_or_commands(
                static, commands, ("power_absorption_fraction",), 1.0
            )
        ),
    }
    voltage = first(static, _DC_VOLTAGE_KEYS)
    mobility = _from_static_or_commands(static, commands, ("electron_mobility_m2_V_s",))
    if voltage is not None:
        model["source_voltage_V"] = float(voltage)
    if mobility is not None:
        model["electron_mobility_m2_V_s"] = float(mobility)
    return model, set(_DC_STATIC_KEYS)


def _resolve_external_model_file(
    static: Mapping[str, Any],
    commands: list[dict[str, Any]],
    *,
    base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
    prefix: str,
) -> Path:
    table = external_file(
        static,
        base_dir=base_dir,
        external_inputs=external_inputs,
        used_external=used_external,
    )
    if table is not None:
        return table
    for command in commands:
        table = external_file(
            command,
            base_dir=base_dir,
            external_inputs=external_inputs,
            used_external=used_external,
        )
        if table is not None:
            return table
    fallback = external_inputs.get("circuit_result_csv")
    if fallback:
        used_external.add("circuit_result_csv")
        return Path(fallback).resolve(strict=False)
    raise MigrationError(f"{prefix}: external_table has no input file")


def _external_table_model(
    static: Mapping[str, Any],
    commands: list[dict[str, Any]],
    *,
    kind: str,
    base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
    prefix: str,
) -> tuple[dict[str, Any], set[str]]:
    model: dict[str, Any] = {
        "kind": kind,
        "file": _resolve_external_model_file(
            static,
            commands,
            base_dir=base_dir,
            external_inputs=external_inputs,
            used_external=used_external,
            prefix=prefix,
        ),
        "interpolation": str(static.get("interpolation", "linear")).lower(),
        "bounds_policy": _normalized_bounds_policy(
            static.get("bounds_policy", static.get("hold", "error"))
        ),
        "power_scale": float(static.get("power_scale", 1.0)),
        "voltage_scale": float(static.get("voltage_scale", 1.0)),
        "current_scale": float(static.get("current_scale", 1.0)),
        "plasma_potential_V": float(static.get("plasma_potential_V", 0.0)),
    }
    for key in ("gap_m", "total_density_m3"):
        if static.get(key) is not None:
            model[key] = float(static[key])
    return model, set(_EXTERNAL_TABLE_STATIC_KEYS)


def _experimental_default_setpoints(static: Mapping[str, Any]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    power = first(static, ("absorbed_power_W", "power_W", "value_W"))
    voltage = first(static, ("voltage_rms_V", "voltage_V", "value_V"))
    if power is not None:
        defaults["default_absorbed_power_W"] = float(power)
    if voltage is not None:
        defaults["default_voltage_rms_V"] = float(voltage)
    return defaults


def _rf_envelope_model(
    static: Mapping[str, Any],
    commands: list[dict[str, Any]],
    *,
    kind: str,
    legacy_port_kind: object,
) -> tuple[dict[str, Any], set[str]]:
    representative = _representative_power_command(commands)

    def setting(names: tuple[str, ...], default: Any) -> Any:
        return _from_static_or_commands(static, (representative,), names, default)

    mode = str(setting(("control_mode", "mode"), "absorbed_power"))
    model: dict[str, Any] = {
        "kind": kind,
        "role": _normalized_rf_role(setting(("role",), legacy_port_kind)),
        "frequency_Hz": float(
            setting(("frequency_Hz", "carrier_frequency_Hz"), 13.56e6)
        ),
        "control": "voltage" if "voltage" in mode.lower() else "absorbed_power",
        "effective_impedance_ohm": float(setting(("effective_impedance_ohm",), 50.0)),
        "coupling_efficiency": float(setting(("coupling_efficiency",), 1.0)),
        "base_reduced_field_Td": float(
            setting(("base_reduced_field_Td", "reduced_field_Td"), 0.0)
        ),
        "reduced_field_per_sqrt_W_Td": float(
            setting(
                ("reduced_field_per_sqrt_W_Td", "reduced_field_per_sqrt_W"),
                0.0,
            )
        ),
        "self_bias_fraction": float(setting(("self_bias_fraction",), 0.35)),
        "plasma_potential_offset_V": float(
            setting(("plasma_potential_offset_V",), 0.0)
        ),
        "plasma_potential_per_sqrt_W": float(
            setting(("plasma_potential_per_sqrt_W",), 0.0)
        ),
    }
    model.update(_experimental_default_setpoints(static))
    return model, set(_RF_ENVELOPE_STATIC_KEYS)


def _ccp_model(
    static: Mapping[str, Any],
    commands: list[dict[str, Any]],
    *,
    kind: str,
) -> tuple[dict[str, Any], set[str]]:
    representative = _representative_power_command(commands)
    representative_mode = None
    if _power_command_magnitude(representative) > 0.0:
        representative_mode = first(representative, ("control_mode", "mode"))
    mode = str(
        representative_mode
        if representative_mode is not None
        else first(static, ("control_mode", "mode"), "absorbed_power")
    )
    model: dict[str, Any] = {
        "kind": kind,
        "frequency_Hz": float(
            _from_static_or_commands(
                static,
                (representative,),
                ("frequency_Hz", "carrier_frequency_Hz"),
                2.0e6,
            )
        ),
        "control": "voltage" if "voltage" in mode.lower() else "absorbed_power",
    }
    model.update(_experimental_default_setpoints(static))
    return model, set(_EXPERIMENTAL_STATIC_KEYS)


def _delivered_power_model(
    static: Mapping[str, Any],
    commands: list[dict[str, Any]],
    *,
    kind: str,
) -> tuple[dict[str, Any], set[str]]:
    representative = _representative_power_command(commands)
    model: dict[str, Any] = {
        "kind": kind,
        "frequency_Hz": float(
            _from_static_or_commands(
                static,
                (representative,),
                ("frequency_Hz", "carrier_frequency_Hz"),
                13.56e6,
            )
        ),
    }
    power = first(static, ("delivered_power_W", "power_W", "value_W", "value"))
    if power is not None:
        model["default_delivered_power_W"] = float(power)
    return model, set(_DELIVERED_POWER_KEYS)


def port_model(
    *,
    port: Any,
    backend: str,
    recipe: Any,
    legacy_base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
    unused: set[str],
    prefix: str,
    gas_fraction: float,
) -> dict[str, Any]:
    kind = _power_kind(backend, str(port.kind))
    static = dict(port.parameters or {})
    commands = _commands_for_port(recipe, str(port.port_id))

    if kind == "prescribed_power":
        model, known = _prescribed_power_model(
            static, kind=kind, gas_fraction=gas_fraction
        )
    elif kind == "dc_series":
        model, known = _dc_series_model(
            static,
            commands,
            kind=kind,
            prefix=prefix,
        )
    elif kind == "external_table":
        model, known = _external_table_model(
            static,
            commands,
            kind=kind,
            base_dir=legacy_base_dir,
            external_inputs=external_inputs,
            used_external=used_external,
            prefix=prefix,
        )
    elif kind == "experimental.rf_envelope":
        model, known = _rf_envelope_model(
            static,
            commands,
            kind=kind,
            legacy_port_kind=port.kind,
        )
    elif kind == "experimental.ccp":
        model, known = _ccp_model(static, commands, kind=kind)
    else:
        model, known = _delivered_power_model(static, commands, kind=kind)

    record_extra_keys(static, known, f"{prefix}.parameters", unused)
    return model
