"""One-way migration from split schema-v2 inputs to one schema-v3 case."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plasma_global.errors import MigrationError
from plasma_global.input.schema import CaseSpec

_LEGACY_APPROXIMATE_FIELD_GRID = (0.2, 2500.0, 48)


@dataclass(frozen=True, slots=True)
class MigrationReport:
    source_path: Path
    unused_keys: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.unused_keys and not self.warnings


@dataclass(frozen=True, slots=True)
class MigrationResult:
    case: CaseSpec
    report: MigrationReport

    @property
    def data(self) -> dict[str, Any]:
        """Return the migrated v3 mapping suitable for YAML serialization."""

        return self.case.model_dump(mode="python", exclude_none=True)


def _first(values: Mapping[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in values and values[name] is not None:
            return values[name]
    return default


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _path_from(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _external_file(
    values: Mapping[str, Any],
    *,
    base_dir: Path,
    external_inputs: Mapping[str, str | None],
    used_external: set[str],
) -> Path | None:
    raw_file = values.get("file")
    if raw_file:
        return _path_from(raw_file, base_dir)
    raw_key = values.get("file_key")
    if raw_key:
        key = str(raw_key)
        target = external_inputs.get(key)
        if not target:
            raise MigrationError(
                f"v2 external input {key!r} is referenced but has no resolved file"
            )
        used_external.add(key)
        return Path(target).resolve(strict=False)
    return None


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
    commands: list[dict[str, Any]],
    names: tuple[str, ...],
    default: Any = None,
) -> Any:
    value = _first(static, names)
    if value is not None:
        return value
    for command in commands:
        value = _first(command, names)
        if value is not None:
            return value
    return default


def _power_command_magnitude(command: Mapping[str, Any]) -> float:
    value = _first(
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


def _extra_keys(
    values: Mapping[str, Any], known: set[str], prefix: str, unused: set[str]
) -> None:
    unused.update(f"{prefix}.{key}" for key in values if key not in known)


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


def _normalized_bounds_policy(value: object) -> str:
    return "hold" if str(value).lower() in {"edge", "clip", "hold"} else "error"


def _reconcile_experimental_setpoint(
    command: dict[str, Any],
    *,
    selected: str,
    incompatible: str,
    prefix: str,
    control: str,
) -> dict[str, Any]:
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


def _match_experimental_setpoint_to_model(
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
    return _reconcile_experimental_setpoint(
        command,
        selected=selected,
        incompatible=incompatible,
        prefix=prefix,
        control=control,
    )


def _prescribed_power_model(
    static: Mapping[str, Any], *, kind: str, gas_fraction: float
) -> tuple[dict[str, Any], set[str]]:
    model: dict[str, Any] = {
        "kind": kind,
        "electron_fraction": 1.0 - gas_fraction,
        "gas_fraction": gas_fraction,
    }
    power = _first(static, _PRESCRIBED_POWER_KEYS)
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
    voltage = _first(static, _DC_VOLTAGE_KEYS)
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
    table = _external_file(
        static,
        base_dir=base_dir,
        external_inputs=external_inputs,
        used_external=used_external,
    )
    if table is not None:
        return table
    for command in commands:
        table = _external_file(
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


def _static_or_representative(
    static: Mapping[str, Any],
    representative: Mapping[str, Any],
    names: tuple[str, ...],
    default: Any,
) -> Any:
    value = _first(static, names)
    return _first(representative, names, default) if value is None else value


def _experimental_default_setpoints(static: Mapping[str, Any]) -> dict[str, float]:
    defaults: dict[str, float] = {}
    power = _first(static, ("absorbed_power_W", "power_W", "value_W"))
    voltage = _first(static, ("voltage_rms_V", "voltage_V", "value_V"))
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
        return _static_or_representative(static, representative, names, default)

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
        representative_mode = _first(representative, ("control_mode", "mode"))
    mode = str(
        representative_mode
        if representative_mode is not None
        else _first(static, ("control_mode", "mode"), "absorbed_power")
    )
    model: dict[str, Any] = {
        "kind": kind,
        "frequency_Hz": float(
            _first(
                static,
                ("frequency_Hz", "carrier_frequency_Hz"),
                _first(
                    representative,
                    ("frequency_Hz", "carrier_frequency_Hz"),
                    2.0e6,
                ),
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
            _first(
                static,
                ("frequency_Hz", "carrier_frequency_Hz"),
                _first(
                    representative,
                    ("frequency_Hz", "carrier_frequency_Hz"),
                    13.56e6,
                ),
            )
        ),
    }
    power = _first(static, ("delivered_power_W", "power_W", "value_W", "value"))
    if power is not None:
        model["default_delivered_power_W"] = float(power)
    return model, set(_DELIVERED_POWER_KEYS)


def _port_model(
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

    _extra_keys(static, known, f"{prefix}.parameters", unused)
    return model


def _legacy_waveform_name(raw_value: object) -> str:
    if isinstance(raw_value, bool):
        return "continuous" if raw_value else "off"
    return str(raw_value).strip().lower()


def _waveform(
    values: Mapping[str, Any], prefix: str, unused: set[str], warnings: list[str]
) -> tuple[dict[str, Any], bool, set[str]]:
    consumed = {"waveform"}
    raw = _legacy_waveform_name(values.get("waveform", "cw"))
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
    power = _numeric_setpoint(_first(raw, _PRESCRIBED_POWER_KEYS), off=is_off)
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
    voltage = _first(raw, _DC_VOLTAGE_KEYS)
    nested = raw.get("voltage")
    if isinstance(nested, Mapping):
        consumed.add("voltage")
        voltage = _first(
            nested,
            ("high_V", "on_V", *_DC_VOLTAGE_KEYS),
            voltage,
        )
        command["off_voltage_V"] = float(_first(nested, ("low_V", "off_V"), 0.0))
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
    file = _external_file(
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
    power = _first(raw, ("absorbed_power_W", "power_W", "value_W"))
    voltage = _first(raw, ("voltage_rms_V", "voltage_V", "value_V"))
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
        _first(raw, ("delivered_power_W", "power_W", "value_W", "value")),
        off=is_off,
    )
    if power is not None:
        command["delivered_power_W"] = power
    return command, set(_DELIVERED_POWER_KEYS)


def _command_waveform(
    kind: str,
    values: Mapping[str, Any],
    prefix: str,
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], bool, set[str]]:
    if kind == "external_table":
        return {}, False, set()
    return _waveform(values, prefix, unused, warnings)


def _power_command(
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
    waveform, is_off, waveform_keys = _command_waveform(
        kind, raw, prefix, unused, warnings
    )
    consumed = set(waveform_keys) | {"mode", "control_mode", "zone_id"}
    if "zone_id" in raw:
        unused.add(f"{prefix}.zone_id")

    mode = str(_first(raw, ("control_mode", "mode"), "absorbed_power"))
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
    _extra_keys(raw, consumed, prefix, unused)
    return command


def _table_electron_model(
    run: Any,
    resolved: Any,
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    table = run.swarm.table
    if not table.file:
        raise MigrationError("v2 swarm table model has no table file")
    requested_lookup = str(table.lookup or run.swarm.closure or "auto").lower()
    closure_kind = (
        "local_field" if requested_lookup == "local_field" else "electron_energy"
    )
    if table.electron_energy_mode:
        unused.add("swarm.table.electron_energy_mode")
        unused.add("swarm.table.energy_relaxation_time_s")
    if str(table.bounds_policy).lower() != "error":
        warnings.append(
            "swarm.table.bounds_policy was changed to 'error' to prevent "
            "silent extrapolation/clipping"
        )
    return (
        {
            "kind": "table",
            "file": _path_from(table.file, Path(resolved.chemistry_dir)),
            "lookup": (
                "local_field" if closure_kind == "local_field" else "mean_energy"
            ),
            "bounds_policy": "error",
        },
        closure_kind,
    )


def _warn_approximate_closure_migration(run: Any, warnings: list[str]) -> None:
    """Record when v2 requested a closure unavailable to the prepared model."""

    requested_closure = str(run.swarm.closure or "auto").lower()
    if requested_closure not in {"auto", "local_field"}:
        warnings.append(
            "boltzmann_2term closure was changed to local_field because the "
            "v3 approximate model is compile-time prepared and has no "
            "electron-energy RHS coupling"
        )


def _approximate_reduced_field_grid(
    cfg: Any,
    warnings: list[str],
) -> dict[str, float | int] | None:
    """Preserve an explicit v2 field grid and safely replace legacy defaults."""

    requested_min = float(cfg.reduced_field_grid_Td.min)
    requested_max = float(cfg.reduced_field_grid_Td.max)
    requested_count = int(cfg.reduced_field_grid_Td.n)
    uses_legacy_defaults = (
        not cfg.reduced_field_grid_was_explicit
        and (
            requested_min,
            requested_max,
            requested_count,
        )
        == _LEGACY_APPROXIMATE_FIELD_GRID
    )
    if uses_legacy_defaults:
        warnings.append(
            "boltzmann_2term used the legacy default reduced-field grid; migration "
            "selected the current v3 defaults because the strict approximate "
            "closure validates a power-balance root at every requested point"
        )
        return None
    return {
        "min_Td": requested_min,
        "max_Td": requested_max,
        "n": requested_count,
    }


def _approximate_two_term_electron_model(
    run: Any, warnings: list[str]
) -> tuple[dict[str, Any], str]:
    cfg = run.swarm.boltzmann_2term
    _warn_approximate_closure_migration(run, warnings)
    reduced_field_grid = _approximate_reduced_field_grid(cfg, warnings)
    electron_model: dict[str, Any] = {
        "kind": "experimental.approximate_two_term",
        "mixture_key_species": list(run.swarm.mixture_key_species),
        "cache": {
            "max_entries": int(run.swarm.cache.max_entries),
            "fraction_decimals": int(run.swarm.cache.fraction_decimals),
        },
        "energy_grid": {
            "min_eV": float(cfg.energy_grid.min_eV),
            "max_eV": float(cfg.energy_grid.max_eV),
            "n": int(cfg.energy_grid.n),
        },
        "max_shape_iterations": int(cfg.max_shape_iterations),
    }
    if reduced_field_grid is not None:
        electron_model["reduced_field_grid"] = reduced_field_grid
    return (
        electron_model,
        "local_field",
    )


def _electron_kinetics_model(
    run: Any,
    resolved: Any,
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    backend = str(run.physics.eedf_backend).strip().lower()
    if backend == "maxwell":
        return {"kind": "maxwellian"}, "electron_energy"
    model_name = str(run.swarm.model_name).lower()
    if backend == "swarm" and model_name == "table":
        return _table_electron_model(run, resolved, unused, warnings)
    if backend == "swarm" and model_name == "boltzmann_2term":
        return _approximate_two_term_electron_model(run, warnings)
    raise MigrationError(
        f"unsupported v2 EEDF selection {backend!r}/{run.swarm.model_name!r}"
    )


def _electron_density_model(
    run: Any, resolved: Any, warnings: list[str]
) -> dict[str, Any]:
    closure = str(run.physics.electron_density_closure).strip().lower()
    if closure == "quasi_neutral":
        return {"kind": "quasineutral"}
    if closure != "prescribed_profile":
        raise MigrationError(f"unsupported v2 electron density closure {closure!r}")

    cfg = run.swarm.prescribed_electron_profile
    if cfg.file:
        profile_file = _path_from(cfg.file, Path(resolved.base_dir))
    elif cfg.file_key and resolved.external_inputs.get(str(cfg.file_key)):
        profile_file = Path(str(resolved.external_inputs[str(cfg.file_key)])).resolve(
            strict=False
        )
    else:
        raise MigrationError("v2 prescribed electron profile has no input file")
    if str(cfg.hold).lower() != "error":
        warnings.append(
            "prescribed electron profile hold was changed to 'error' to "
            "prevent silent endpoint extension"
        )
    return {
        "kind": "experimental.prescribed_profile",
        "file": profile_file,
        "zone_columns": dict(cfg.zone_columns),
        "interpolation": str(cfg.interpolation).lower(),
        "hold": "error",
    }


def _electron_models(
    loaded: Any, unused: set[str], warnings: list[str]
) -> dict[str, Any]:
    run = loaded.run_config
    electrons, closure_kind = _electron_kinetics_model(
        run,
        loaded.resolved_paths,
        unused,
        warnings,
    )
    density = _electron_density_model(run, loaded.resolved_paths, warnings)
    return {
        "electrons": electrons,
        "electron_closure": {"kind": closure_kind},
        "electron_density": density,
        "gas_energy": {
            "kind": "evolved" if run.physics.enable_gas_temperature else "fixed"
        },
        "surface_kinetics": (
            {}
            if run.physics.enable_surface_coverages
            and loaded.chamber.surfaces
            and loaded.mechanism.surface_species
            else None
        ),
    }


def _solver(
    run: Any,
    *,
    sample_interval_s: float,
    unused: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    integrator = str(run.physics.integrator).strip().lower()
    if integrator != "scipy_bdf":
        raise MigrationError(f"unsupported v2 integrator {run.physics.integrator!r}")
    positivity = dict(run.numerics.positivity or {})
    unused.update(f"numerics.positivity.{key}" for key in positivity)
    unused.add("numerics.atol")
    warnings.append(
        "v2 scalar atol used physical state units and cannot be transferred to the "
        "dimensionless v3 state; solver.atol was set explicitly to 1e-14"
    )
    result: dict[str, Any] = {
        "method": "BDF",
        "rtol": float(run.numerics.rtol),
        "atol": 1.0e-14,
        "first_step_s": _float_or_none(run.numerics.first_step),
        "max_step_s": _float_or_none(run.numerics.max_step),
        "sample_interval_s": sample_interval_s,
    }
    return result


def _experimental_quasi_steady(run: Any, unused: set[str]) -> dict[str, float] | None:
    events = dict(run.numerics.events or {})
    steady = dict(events.get("steady_state", {}) or {})
    _extra_keys(events, {"steady_state"}, "numerics.events", unused)
    _extra_keys(
        steady,
        {"enabled", "relative_rhs_norm_s_inv", "min_step_time_s"},
        "numerics.events.steady_state",
        unused,
    )
    if not bool(steady.get("enabled", False)):
        if "steady_state" in events:
            unused.add("numerics.events.steady_state")
            unused.update(f"numerics.events.steady_state.{key}" for key in steady)
        return None
    return {
        "relative_rhs_norm_s_inv": float(steady.get("relative_rhs_norm_s_inv", 1.0e-3)),
        "min_time_s": float(steady.get("min_step_time_s", 0.0)),
    }


def _wall_transport(
    values: Mapping[str, Any], prefix: str, unused: set[str]
) -> dict[str, Any]:
    raw = dict(values or {})
    mode = str(raw.get("ion_loss", "bohm")).strip().lower()
    if mode == "bohm":
        known = {
            "ion_loss",
            "characteristic_length_m",
            "h_factor",
            "min_h_factor",
            "max_h_factor",
            "ion_neutral_cross_section_m2",
        }
        transport: dict[str, Any] = {"kind": "bohm"}
        for key in (
            "characteristic_length_m",
            "h_factor",
            "min_h_factor",
            "max_h_factor",
            "ion_neutral_cross_section_m2",
        ):
            if raw.get(key) is not None:
                value = raw[key]
                transport[key] = (
                    "auto"
                    if key == "h_factor" and str(value).lower() == "auto"
                    else float(value)
                )
    elif mode == "prescribed_loss_frequency":
        known = {"ion_loss", "frequency_s"}
        if raw.get("frequency_s") is None:
            raise MigrationError(f"{prefix}: prescribed ion loss needs frequency_s")
        transport = {
            "kind": "prescribed_frequency",
            "frequency_s_inv": float(raw["frequency_s"]),
        }
    elif mode == "ambipolar_diffusion":
        known = {
            "ion_loss",
            "diffusion_coefficient_m2_s",
            "diffusion_length_m",
        }
        if raw.get("diffusion_coefficient_m2_s") is None:
            raise MigrationError(
                f"{prefix}: ambipolar ion loss needs diffusion_coefficient_m2_s"
            )
        transport = {
            "kind": "ambipolar",
            "diffusion_coefficient_m2_s": float(raw["diffusion_coefficient_m2_s"]),
        }
        if raw.get("diffusion_length_m") is not None:
            transport["diffusion_length_m"] = float(raw["diffusion_length_m"])
    elif mode == "off":
        known = {"ion_loss"}
        transport = {"kind": "off"}
    else:
        raise MigrationError(f"{prefix}: unsupported ion_loss mode {mode!r}")
    _extra_keys(raw, known, prefix, unused)
    return transport


_LEGACY_ION_SEED_M3 = 1.0e13
_BOLTZMANN_J_K = 1.380649e-23


def _positive_ion_ids(gas_species: Sequence[Any]) -> list[str]:
    return [
        str(species.canonical_id)
        for species in gas_species
        if int(getattr(species, "charge", 0)) > 0
    ]


def _species_seed_template(gas_species: Sequence[Any]) -> dict[str, float]:
    positive_ions = set(_positive_ion_ids(gas_species))
    return {
        str(species.canonical_id): (
            _LEGACY_ION_SEED_M3 if str(species.canonical_id) in positive_ions else 0.0
        )
        for species in gas_species
    }


def _warn_seeded_ions(
    zone_id: object, ion_ids: Sequence[str], warnings: list[str]
) -> None:
    if ion_ids:
        warnings.append(
            f"zone {zone_id!r} positive-ion seeds were made explicit at "
            f"{_LEGACY_ION_SEED_M3:g} m^-3 for {list(ion_ids)}"
        )


def _explicit_zone_densities(
    zone: Any,
    gas_species: Sequence[Any],
    seeded: Mapping[str, float],
    warnings: list[str],
) -> dict[str, float] | None:
    explicit = {
        str(species): float(density)
        for species, density in (zone.initial_densities_m3 or {}).items()
    }
    if not explicit or not any(value > 0.0 for value in explicit.values()):
        return None
    densities = dict(seeded)
    densities.update(explicit)
    defaulted_ions = [
        species_id
        for species_id in _positive_ion_ids(gas_species)
        if species_id not in explicit
    ]
    _warn_seeded_ions(zone.zone_id, defaulted_ions, warnings)
    return densities


def _first_step_flows(loaded: Any, zone_id: object) -> dict[str, float]:
    first_step = loaded.recipe.steps[0]
    flows: dict[str, float] = {}
    for inlet_id, species_flows in first_step.gas_inlets.items():
        inlet = loaded.chamber.inlet_by_id.get(inlet_id)
        if inlet is None or inlet.zone_id != zone_id:
            continue
        for species, flow in species_flows.items():
            flows[str(species)] = flows.get(str(species), 0.0) + float(flow)
    if flows:
        return flows
    for species_flows in first_step.gas_inlets.values():
        for species, flow in species_flows.items():
            flows[str(species)] = flows.get(str(species), 0.0) + float(flow)
    return flows


def _initial_neutral_flows(
    loaded: Any, zone: Any, gas_species: Sequence[Any]
) -> dict[str, float]:
    neutral_ids = [
        str(species.canonical_id)
        for species in gas_species
        if int(getattr(species, "charge", 0)) == 0
    ]
    flows = {
        species: flow
        for species, flow in _first_step_flows(loaded, zone.zone_id).items()
        if species in neutral_ids and flow > 0.0
    }
    if flows:
        return flows
    if not neutral_ids:
        raise MigrationError(
            f"zone {zone.zone_id!r} has no initial density and chemistry has no "
            "neutral gas species"
        )
    return {neutral_ids[0]: 1.0}


def _ideal_gas_densities(zone: Any, flows: Mapping[str, float]) -> dict[str, float]:
    total_density = float(zone.pressure_Pa) / (
        _BOLTZMANN_J_K * max(float(zone.gas_temperature_K), 1.0)
    )
    flow_total = sum(flows.values())
    return {
        species: total_density * flow / flow_total for species, flow in flows.items()
    }


def _initial_zone_densities(
    loaded: Any, zone: Any, warnings: list[str]
) -> dict[str, float]:
    gas_species = list(getattr(loaded.mechanism, "gas_state_species", []) or [])
    seeded = _species_seed_template(gas_species)
    explicit = _explicit_zone_densities(zone, gas_species, seeded, warnings)
    if explicit is not None:
        return explicit

    flows = _initial_neutral_flows(loaded, zone, gas_species)
    seeded.update(_ideal_gas_densities(zone, flows))
    _warn_seeded_ions(zone.zone_id, _positive_ion_ids(gas_species), warnings)
    warnings.append(
        f"zone {zone.zone_id!r} initial_densities_m3 was derived from pressure, "
        "temperature, and the first recipe gas composition"
    )
    return seeded


def _migrate_reactor_zones(
    loaded: Any,
    chamber: Any,
    models: Mapping[str, Any],
    unused: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for index, zone in enumerate(chamber.zones):
        if str(zone.role) != "process":
            unused.add(f"reactor.zones[{index}].role")
        zones.append(
            {
                "zone_id": str(zone.zone_id),
                "description": str(zone.description),
                "volume_m3": float(zone.volume_m3),
                "pressure_Pa": float(zone.pressure_Pa),
                "gas_temperature_K": float(zone.gas_temperature_K),
                "initial_densities_m3": _initial_zone_densities(loaded, zone, warnings),
                "initial_mean_energy_eV": (
                    3.0
                    if models["electron_closure"]["kind"] == "electron_energy"
                    else None
                ),
            }
        )
    return zones


def _migrate_reactor_edges(chamber: Any, unused: set[str]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for index, edge in enumerate(chamber.edges):
        if edge.notes:
            unused.add(f"reactor.edges[{index}].notes")
        edges.append(
            {
                "edge_id": str(edge.edge_id),
                "from_zone": str(edge.from_zone),
                "to_zone": str(edge.to_zone),
                "conductance_m3_s": float(edge.conductance_m3_s),
            }
        )
    return edges


def _migrate_reactor_surfaces(
    loaded: Any, chamber: Any, unused: set[str]
) -> list[dict[str, Any]]:
    free_site_ids, film_fragment_ids = _legacy_surface_species_groups(loaded)
    return [
        _migrate_reactor_surface(
            surface,
            index=index,
            free_site_ids=free_site_ids,
            film_fragment_ids=film_fragment_ids,
            unused=unused,
        )
        for index, surface in enumerate(chamber.surfaces)
    ]


def _legacy_surface_species_groups(loaded: Any) -> tuple[set[str], set[str]]:
    surface_species = getattr(loaded.mechanism, "surface_species", [])
    free_site_ids = {
        str(species.canonical_id)
        for species in surface_species
        if "site" in set(getattr(species, "state_tags", set()) or set())
    }
    film_fragment_ids = {
        str(species.canonical_id)
        for species in surface_species
        if "film_fragment" in set(getattr(species, "state_tags", set()) or set())
    }
    return free_site_ids, film_fragment_ids


def _migrate_reactor_surface(
    surface: Any,
    *,
    index: int,
    free_site_ids: set[str],
    film_fragment_ids: set[str],
    unused: set[str],
) -> dict[str, Any]:
    prefix = f"reactor.surfaces[{index}]"
    unused.update((f"{prefix}.kind", f"{prefix}.material"))
    excluded = free_site_ids | film_fragment_ids
    coverages = {
        str(key): float(value)
        for key, value in surface.initial_coverages.items()
        if str(key) not in excluded
    }
    ignored_coverages = set(surface.initial_coverages) - set(coverages)
    unused.update(f"{prefix}.initial_coverages.{key}" for key in ignored_coverages)
    return {
        "surface_id": str(surface.surface_id),
        "zone_id": str(surface.zone_id),
        "area_m2": float(surface.area_m2),
        "temperature_K": float(surface.temperature_K),
        "site_density_m2": float(surface.site_density_m2),
        "initial_coverages": coverages,
        "wall_transport": _wall_transport(surface.models, f"{prefix}.models", unused),
    }


def _migrate_reactor_flow_devices(
    chamber: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inlets = [
        {
            "inlet_id": str(inlet.inlet_id),
            "zone_id": str(inlet.zone_id),
            "flow_sccm": {
                str(key): float(value) for key, value in inlet.flow_sccm.items()
            },
            "temperature_K": float(inlet.temperature_K),
        }
        for inlet in chamber.gas_inlets
    ]
    pumps = [
        {
            "pump_id": str(pump.pump_id),
            "zone_id": str(pump.zone_id),
            "speed_m3_s": float(pump.speed_m3_s),
        }
        for pump in chamber.pumps
    ]
    return inlets, pumps


def _migrate_power_ports(
    *,
    chamber: Any,
    run: Any,
    recipe: Any,
    resolved: Any,
    gas_fraction: float,
    used_external: set[str],
    unused: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    ports: list[dict[str, Any]] = []
    model_by_id: dict[str, Mapping[str, Any]] = {}
    for index, port in enumerate(chamber.power_ports):
        model = _port_model(
            port=port,
            backend=str(run.physics.electrical_backend),
            recipe=recipe,
            legacy_base_dir=Path(resolved.base_dir),
            external_inputs=resolved.external_inputs,
            used_external=used_external,
            unused=unused,
            prefix=f"reactor.power_ports[{index}]",
            gas_fraction=gas_fraction,
        )
        port_id = str(port.port_id)
        model_by_id[port_id] = model
        ports.append(
            {
                "port_id": port_id,
                "zone_id": str(port.zone_id),
                "coupling_target": str(port.coupling_target),
                "model": model,
            }
        )
    return ports, model_by_id


def _migrate_reactor(
    *,
    loaded: Any,
    models: Mapping[str, Any],
    gas_fraction: float,
    used_external: set[str],
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    chamber = loaded.chamber
    inlets, pumps = _migrate_reactor_flow_devices(chamber)
    ports, model_by_id = _migrate_power_ports(
        chamber=chamber,
        run=loaded.run_config,
        recipe=loaded.recipe,
        resolved=loaded.resolved_paths,
        gas_fraction=gas_fraction,
        used_external=used_external,
        unused=unused,
    )
    reactor = {
        "chamber_id": str(chamber.chamber_id),
        "description": str(chamber.description),
        "zones": _migrate_reactor_zones(loaded, chamber, models, unused, warnings),
        "edges": _migrate_reactor_edges(chamber, unused),
        "surfaces": _migrate_reactor_surfaces(loaded, chamber, unused),
        "gas_inlets": inlets,
        "pumps": pumps,
        "power_ports": ports,
    }
    return reactor, model_by_id


def _migrate_step_power_commands(
    *,
    step: Any,
    step_index: int,
    port_model_by_id: Mapping[str, Mapping[str, Any]],
    recipe_dir: Path,
    external_inputs: Mapping[str, Any],
    used_external: set[str],
    unused: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    for raw_port_id, values in step.power_ports.items():
        port_id = str(raw_port_id)
        if port_id not in port_model_by_id:
            raise MigrationError(
                f"recipe.steps[{step_index}] references unknown power port {port_id!r}"
            )
        port_model = port_model_by_id[port_id]
        prefix = f"recipe.steps[{step_index}].power_ports.{port_id}"
        migrated = _power_command(
            values=dict(values or {}),
            kind=str(port_model["kind"]),
            base_dir=recipe_dir,
            external_inputs=external_inputs,
            used_external=used_external,
            prefix=prefix,
            unused=unused,
            warnings=warnings,
        )
        commands[port_id] = _match_experimental_setpoint_to_model(
            migrated, port_model, prefix
        )
    return commands


def _migrate_step_surfaces(
    step: Any, step_index: int, unused: set[str]
) -> dict[str, Any]:
    surfaces: dict[str, Any] = {}
    for surface_id, raw_values in step.surface_overrides.items():
        values = dict(raw_values or {})
        prefix = f"recipe.steps[{step_index}].surface_overrides.{surface_id}"
        _extra_keys(values, {"temperature_K"}, prefix, unused)
        surfaces[str(surface_id)] = {
            "temperature_K": _float_or_none(values.get("temperature_K"))
        }
    return surfaces


def _migrate_recipe_steps(
    *,
    recipe: Any,
    resolved: Any,
    port_model_by_id: Mapping[str, Mapping[str, Any]],
    used_external: set[str],
    unused: set[str],
    warnings: list[str],
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    previous_end: float | None = None
    recipe_dir = Path(resolved.recipe_file).parent
    for index, step in enumerate(recipe.steps):
        start, end = float(step.t_start_s), float(step.t_end_s)
        if previous_end is not None and abs(start - previous_end) > max(
            1.0e-15, 1.0e-12 * max(abs(start), abs(previous_end), 1.0)
        ):
            warnings.append(
                f"recipe.steps[{index}] began at {start:g}s instead of the previous "
                f"end {previous_end:g}s; v3 makes steps contiguous"
            )
        previous_end = end
        power_commands = _migrate_step_power_commands(
            step=step,
            step_index=index,
            port_model_by_id=port_model_by_id,
            recipe_dir=recipe_dir,
            external_inputs=resolved.external_inputs,
            used_external=used_external,
            unused=unused,
            warnings=warnings,
        )
        surfaces = _migrate_step_surfaces(step, index, unused)
        unused.update(
            f"recipe.steps[{index}].imported_inputs.{key}"
            for key in step.imported_inputs
        )
        steps.append(
            {
                "step_id": str(step.step_id),
                "duration_s": end - start,
                "commands": {
                    "gas_inlets": {
                        str(inlet_id): {
                            "flow_sccm": {
                                str(species): float(flow)
                                for species, flow in values.items()
                            }
                        }
                        for inlet_id, values in step.gas_inlets.items()
                    },
                    "power_ports": power_commands,
                    "surfaces": surfaces,
                },
            }
        )
    return steps


def _migrate_output(
    run: Any,
    recipe_steps: Sequence[Mapping[str, Any]],
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], float]:
    formats = run.outputs.formats
    if not formats.solution_h5:
        warnings.append("v3 always writes the HDF5 result bundle")
    if not formats.summary_yaml:
        warnings.append("v3 always writes summary.yaml")
    if formats.observables_csv:
        unused.add("outputs.formats.observables_csv")
    if run.outputs.plots.enabled:
        unused.add("outputs.plots")
    if run.outputs.budgets.enabled:
        unused.add("outputs.budgets")
    unused.add("files.output_dir")
    total_duration = sum(float(item["duration_s"]) for item in recipe_steps)
    return {}, max(total_duration / 199.0, 1.0e-15)


def _legacy_initial_inventory(chamber: Any) -> dict[str, dict[str, float]]:
    return {
        str(surface.surface_id): {
            str(key): float(value) for key, value in surface.initial_inventory.items()
        }
        for surface in chamber.surfaces
        if surface.initial_inventory
    }


def _inventory_reaction(
    *,
    loaded: Any,
    surface_id: str,
    inventory_id: str,
    gas_species_ids: set[str],
) -> Any:
    matching = [
        reaction
        for reaction in loaded.mechanism.surface_reactions
        if _reaction_matches_inventory(
            reaction,
            surface_id=surface_id,
            inventory_id=inventory_id,
            gas_species_ids=gas_species_ids,
        )
    ]
    if len(matching) != 1:
        raise MigrationError(
            "cannot migrate wall inventory without one unambiguous surface event: "
            f"{surface_id}.{inventory_id} matched "
            f"{[item.reaction_id for item in matching]}"
        )
    return matching[0]


def _reaction_matches_inventory(
    reaction: Any,
    *,
    surface_id: str,
    inventory_id: str,
    gas_species_ids: set[str],
) -> bool:
    if not reaction.enabled or (
        reaction.surface_filter and surface_id not in reaction.surface_filter
    ):
        return False
    gas_reactants = [
        species_id
        for species_id in reaction.gas_reactants
        if species_id in gas_species_ids
    ]
    return bool(gas_reactants) and f"{gas_reactants[0]}_reservoir" == inventory_id


def _migrate_wall_inventory(
    loaded: Any,
    initial_inventory: Mapping[str, Mapping[str, float]],
    warnings: list[str],
) -> dict[str, Any]:
    gas_species_ids = {
        str(species.canonical_id)
        for species in loaded.mechanism.gas_state_species
        if str(species.canonical_id) != "e"
    }
    event_yields: dict[tuple[str, str], dict[str, float]] = {}
    for surface_id, inventory_entries in initial_inventory.items():
        for inventory_id in inventory_entries:
            reaction = _inventory_reaction(
                loaded=loaded,
                surface_id=surface_id,
                inventory_id=inventory_id,
                gas_species_ids=gas_species_ids,
            )
            key = (str(reaction.reaction_id), surface_id)
            if key not in event_yields:
                event_yields[key] = {}
            event_yields[key][inventory_id] = 1.0
    warnings.append(
        "wall inventory event yields were made explicit from the unique v2 "
        "unit-per-event mapping; review each migrated yield"
    )
    return {
        "initial_by_surface": initial_inventory,
        "events": [
            {
                "reaction_id": reaction_id,
                "surface_id": surface_id,
                "inventory_particles_per_event": yields,
            }
            for (reaction_id, surface_id), yields in event_yields.items()
        ],
    }


def _migrate_experimental(
    *,
    loaded: Any,
    run: Any,
    chamber: Any,
    unused: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    experimental: dict[str, Any] = {}
    quasi_steady = _experimental_quasi_steady(run, unused)
    if quasi_steady is not None:
        experimental["stop_when_quasi_steady"] = quasi_steady
    has_film_state = any(
        "film_fragment" in set(getattr(species, "state_tags", ()) or ())
        for species in loaded.mechanism.surface_species
    )
    if run.physics.enable_surface_coverages and has_film_state:
        experimental["film"] = dict[str, Any]()
    initial_inventory = _legacy_initial_inventory(chamber)
    if run.physics.enable_wall_inventory and initial_inventory:
        experimental["wall_inventory"] = _migrate_wall_inventory(
            loaded, initial_inventory, warnings
        )
    if loaded.mechanism.state_variables or loaded.mechanism.processes:
        experimental["extensions"] = dict[str, Any]()
    return experimental


def migrate_v2(path: str | Path) -> MigrationResult:
    """Read a schema-v2 case at the migration boundary and convert it to v3."""

    from plasma_global.input._legacy_v2 import load_legacy_v2_case

    source = Path(path).resolve(strict=False)
    loaded = load_legacy_v2_case(source)
    run = loaded.run_config
    chamber = loaded.chamber
    recipe = loaded.recipe
    resolved = loaded.resolved_paths
    unused: set[str] = {
        "case.kind",
        "case.tags",
        "physics.mode",
        "physics.gas_model",
        "physics.wall_relaxation_s_inv",
        "runtime.export_effective_config",
        "runtime.export_resolved_paths",
        "outputs.formats.solution_h5",
        "outputs.formats.summary_yaml",
    }
    warnings: list[str] = []
    used_external: set[str] = set()
    gas_fraction = float(run.physics.gas_heating_fraction)
    if not 0.0 <= gas_fraction <= 1.0:
        raise MigrationError("v2 physics.gas_heating_fraction must be between 0 and 1")
    models = _electron_models(loaded, unused, warnings)

    reactor, port_model_by_id = _migrate_reactor(
        loaded=loaded,
        models=models,
        gas_fraction=gas_fraction,
        used_external=used_external,
        unused=unused,
        warnings=warnings,
    )

    recipe_steps = _migrate_recipe_steps(
        recipe=recipe,
        resolved=resolved,
        port_model_by_id=port_model_by_id,
        used_external=used_external,
        unused=unused,
        warnings=warnings,
    )

    for key in resolved.external_inputs:
        if key not in used_external:
            unused.add(f"files.external_inputs.{key}")

    if not any(
        port["model"]["kind"] == "prescribed_power" for port in reactor["power_ports"]
    ):
        unused.add("physics.gas_heating_fraction")

    output, sample_interval_s = _migrate_output(run, recipe_steps, unused, warnings)
    experimental = _migrate_experimental(
        loaded=loaded,
        run=run,
        chamber=chamber,
        unused=unused,
        warnings=warnings,
    )
    data = {
        "schema_version": 3,
        "case": {
            "name": str(run.case.name),
            "description": str(run.case.description),
        },
        "chemistry": {"manifest": Path(resolved.chemistry_manifest)},
        "reactor": reactor,
        "recipe": {
            "recipe_id": str(recipe.recipe_id),
            "description": str(recipe.description),
            "start_time_s": float(recipe.steps[0].t_start_s),
            "steps": recipe_steps,
        },
        "models": models,
        "solver": _solver(
            run,
            sample_interval_s=sample_interval_s,
            unused=unused,
            warnings=warnings,
        ),
        "output": output,
    }
    if experimental:
        data["experimental"] = experimental
    try:
        case = CaseSpec.model_validate(data)
    except Exception as exc:
        raise MigrationError(f"generated v3 case failed validation: {exc}") from exc
    report = MigrationReport(
        source_path=source,
        unused_keys=tuple(sorted(unused)),
        warnings=tuple(warnings),
    )
    return MigrationResult(case=case, report=report)


def _yaml_value(value: Any, *, relative_to: Path | None = None) -> Any:
    if isinstance(value, Path):
        if relative_to is not None and value.is_absolute():
            try:
                return Path(os.path.relpath(value, relative_to)).as_posix()
            except ValueError:
                pass
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _yaml_value(item, relative_to=relative_to)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item, relative_to=relative_to) for item in value]
    return value


def write_v3_case(result: MigrationResult, destination: str | Path) -> Path:
    """Write migrated v3 YAML without copying referenced assets."""

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        _yaml_value(result.data, relative_to=target.parent),
        sort_keys=False,
        allow_unicode=True,
    )
    target.write_text(text, encoding="utf-8")
    return target


def _migration_report_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}.migration.yaml")


def write_migration_report(result: MigrationResult, destination: str | Path) -> Path:
    """Persist the intentionally dropped keys and migration warnings."""

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(result.report.source_path),
        "unused_keys": list(result.report.unused_keys),
        "warnings": list(result.report.warnings),
    }
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


class _AssetStager:
    """Copy or convert each unique runtime asset into a publication bundle."""

    def __init__(self, staging_directory: Path, published_directory: Path) -> None:
        self.staging_directory = staging_directory
        self.published_directory = published_directory
        self.staged: dict[tuple[Path, str], Path] = {}
        self.warnings: list[str] = []

    def stage(self, value: Any, label: str, *, electron_table: bool = False) -> Path:
        source = Path(value).resolve()
        if not source.is_file():
            raise MigrationError(f"{label} does not exist: {source}")
        conversion = "electron_table" if electron_table else "copy"
        key = (source, conversion)
        if key in self.staged:
            return self.staged[key]
        suffix = source.suffix or ".dat"
        filename = f"{len(self.staged):02d}_{label.replace('.', '_')}{suffix}"
        temporary_path = self.staging_directory / filename
        published_path = (self.published_directory / filename).resolve()
        self.staging_directory.mkdir(parents=True, exist_ok=True)
        try:
            if electron_table:
                from tools.importers.rate_table import convert_v2_rate_table_h5

                _, dropped = convert_v2_rate_table_h5(source, temporary_path)
                if dropped:
                    self.warnings.append(
                        f"{label}: dropped unused legacy HDF5 datasets: "
                        f"{', '.join(dropped)}"
                    )
            else:
                shutil.copy2(source, temporary_path)
        except (OSError, ValueError) as exc:
            raise MigrationError(
                f"cannot migrate {label} from {source}: {exc}"
            ) from exc
        self.staged[key] = published_path
        return published_path


def _stage_electron_assets(data: dict[str, Any], stager: _AssetStager) -> None:
    models = data["models"]
    electrons = models["electrons"]
    if electrons["kind"] == "table":
        electrons["file"] = stager.stage(
            electrons["file"], "models_electrons", electron_table=True
        )
    electron_density = models["electron_density"]
    if electron_density["kind"] == "experimental.prescribed_profile":
        electron_density["file"] = stager.stage(
            electron_density["file"], "models_electron_density"
        )


def _stage_external_power_assets(data: dict[str, Any], stager: _AssetStager) -> None:
    for port_index, port in enumerate(data["reactor"]["power_ports"]):
        model = port["model"]
        if model["kind"] == "external_table":
            model["file"] = stager.stage(
                model["file"], f"power_model_{port_index}_{port['port_id']}"
            )

    for step_index, step in enumerate(data["recipe"]["steps"]):
        commands = step["commands"]["power_ports"]
        for port_id, command in commands.items():
            if command["kind"] == "external_table" and "file" in command:
                command["file"] = stager.stage(
                    command["file"],
                    f"power_command_{step_index}_{port_id}",
                )


def _stage_case_assets(
    case: CaseSpec,
    staging_directory: Path,
    published_directory: Path,
) -> tuple[CaseSpec, tuple[str, ...]]:
    """Bundle runtime file dependencies and replace their paths in a case."""

    data = case.model_dump(mode="python", exclude_none=True)
    stager = _AssetStager(staging_directory, published_directory)
    _stage_electron_assets(data, stager)
    _stage_external_power_assets(data, stager)
    try:
        migrated = CaseSpec.model_validate(data)
    except Exception as exc:
        raise MigrationError(f"bundled v3 case failed validation: {exc}") from exc
    return migrated, tuple(stager.warnings)


def migrate_v2_to_yaml(
    source: str | Path,
    destination: str | Path,
    *,
    boundary_products: Mapping[str, str] | None = None,
    cross_section_segments: Mapping[str, int] | None = None,
) -> MigrationResult:
    """Write a runnable v3 case, canonical chemistry, and migration report.

    Chemistry conversion is intentionally part of this one-way boundary.  A
    wall product or concatenated cross-section axis that cannot be selected
    uniquely is an error; the migrator never guesses scientific data.
    """

    from tools.importers.chemistry_v2 import convert_v2_chemistry

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    chemistry_target = target.parent / f"{target.stem}_chemistry"
    assets_target = target.parent / f"{target.stem}_assets"
    if target.exists():
        raise MigrationError(f"migration output already exists: {target}")
    if chemistry_target.exists():
        raise MigrationError(
            f"migration chemistry output already exists: {chemistry_target}"
        )
    if assets_target.exists():
        raise MigrationError(f"migration asset output already exists: {assets_target}")

    result = migrate_v2(source)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{target.stem}-migration-", dir=target.parent
        ) as temporary:
            temporary_root = Path(temporary)
            temporary_chemistry = temporary_root / "chemistry"
            temporary_assets = temporary_root / "assets"
            bundled_case, asset_warnings = _stage_case_assets(
                result.case,
                temporary_assets,
                assets_target,
            )
            chemistry_warnings: list[str] = []
            canonical_manifest = convert_v2_chemistry(
                result.case.chemistry.manifest,
                temporary_chemistry,
                boundary_products=boundary_products,
                cross_section_segments=cross_section_segments,
                migration_warnings=chemistry_warnings,
            )
            # Validate the scientific bundle before publishing it.
            from plasma_global.chemistry.compile import compile_chemistry
            from plasma_global.chemistry.data import load_chemistry

            compile_chemistry(load_chemistry(canonical_manifest))
            shutil.move(str(temporary_chemistry), str(chemistry_target))
            if temporary_assets.exists():
                shutil.move(str(temporary_assets), str(assets_target))

        published_manifest = (chemistry_target / "chemistry.yaml").resolve()
        chemistry = bundled_case.chemistry.model_copy(
            update={"manifest": published_manifest}
        )
        report = MigrationReport(
            source_path=result.report.source_path,
            unused_keys=result.report.unused_keys,
            warnings=(
                *result.report.warnings,
                *chemistry_warnings,
                *asset_warnings,
            ),
        )
        migrated = MigrationResult(
            case=bundled_case.model_copy(update={"chemistry": chemistry}),
            report=report,
        )
        write_v3_case(migrated, target)
        write_migration_report(migrated, _migration_report_path(target))
    except Exception:
        # Remove only paths created by this invocation.
        if chemistry_target.exists():
            shutil.rmtree(chemistry_target)
        if assets_target.exists():
            shutil.rmtree(assets_target)
        target.unlink(missing_ok=True)
        _migration_report_path(target).unlink(missing_ok=True)
        raise
    return migrated


__all__ = [
    "MigrationError",
    "MigrationReport",
    "MigrationResult",
    "migrate_v2",
    "migrate_v2_to_yaml",
    "write_migration_report",
    "write_v3_case",
]
