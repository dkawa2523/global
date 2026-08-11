"""One-way migration from split schema-v2 inputs to one schema-v3 case."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plasma_global.errors import MigrationError
from plasma_global.input.schema import CaseSpec


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
        """Return a standalone v3 mapping suitable for YAML serialization."""

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
    if selected in command or incompatible not in command:
        return command
    value = float(command[incompatible])
    if value != 0.0:
        raise MigrationError(
            f"{prefix} provides {incompatible}={value:g}, but its reactor model "
            f"uses {control!r} control"
        )
    command.pop(incompatible)
    command[selected] = 0.0
    return command


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
        known = {
            "absorbed_power_W",
            "power_W",
            "value_W",
            "value",
        }
        model: dict[str, Any] = {
            "kind": kind,
            "electron_fraction": 1.0 - gas_fraction,
            "gas_fraction": gas_fraction,
        }
        power = _first(static, ("absorbed_power_W", "power_W", "value_W", "value"))
        if power is not None:
            model["default_absorbed_power_W"] = float(power)
    elif kind == "dc_series":
        known = {
            "source_voltage_V",
            "voltage_V",
            "value_V",
            "value",
            "ballast_resistance_ohm",
            "series_resistance_ohm",
            "gap_m",
            "electrode_gap_m",
            "electrode_area_m2",
            "area_m2",
            "power_absorption_fraction",
            "electron_mobility_m2_V_s",
        }
        ballast = _from_static_or_commands(
            static, commands, ("ballast_resistance_ohm", "series_resistance_ohm")
        )
        gap = _from_static_or_commands(static, commands, ("gap_m", "electrode_gap_m"))
        area = _from_static_or_commands(
            static, commands, ("electrode_area_m2", "area_m2")
        )
        if ballast is None or gap is None or area is None:
            raise MigrationError(
                f"{prefix}: dc_series requires ballast resistance, gap, and electrode area"
            )
        model = {
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
        voltage = _first(static, ("source_voltage_V", "voltage_V", "value_V", "value"))
        mobility = _from_static_or_commands(
            static, commands, ("electron_mobility_m2_V_s",)
        )
        if voltage is not None:
            model["source_voltage_V"] = float(voltage)
        if mobility is not None:
            model["electron_mobility_m2_V_s"] = float(mobility)
    elif kind == "external_table":
        known = {
            "file",
            "file_key",
            "interpolation",
            "hold",
            "power_scale",
            "voltage_scale",
            "current_scale",
            "gap_m",
            "total_density_m3",
            "plasma_potential_V",
            "bounds_policy",
        }
        table = _external_file(
            static,
            base_dir=legacy_base_dir,
            external_inputs=external_inputs,
            used_external=used_external,
        )
        if table is None:
            for command in commands:
                table = _external_file(
                    command,
                    base_dir=legacy_base_dir,
                    external_inputs=external_inputs,
                    used_external=used_external,
                )
                if table is not None:
                    break
        if table is None and external_inputs.get("circuit_result_csv"):
            used_external.add("circuit_result_csv")
            table = Path(str(external_inputs["circuit_result_csv"])).resolve(
                strict=False
            )
        if table is None:
            raise MigrationError(f"{prefix}: external_table has no input file")
        model = {
            "kind": kind,
            "file": table,
            "interpolation": str(static.get("interpolation", "linear")).lower(),
            "bounds_policy": (
                "hold"
                if str(static.get("bounds_policy", static.get("hold", "error"))).lower()
                in {"edge", "clip", "hold"}
                else "error"
            ),
            "power_scale": float(static.get("power_scale", 1.0)),
            "voltage_scale": float(static.get("voltage_scale", 1.0)),
            "current_scale": float(static.get("current_scale", 1.0)),
            "plasma_potential_V": float(static.get("plasma_potential_V", 0.0)),
        }
        for key in ("gap_m", "total_density_m3"):
            if static.get(key) is not None:
                model[key] = float(static[key])
    elif kind == "experimental.rf_envelope":
        representative = _representative_power_command(commands)

        def rf_setting(names: tuple[str, ...], default: Any) -> Any:
            value = _first(static, names)
            return _first(representative, names, default) if value is None else value

        known = {
            "role",
            "frequency_Hz",
            "carrier_frequency_Hz",
            "control_mode",
            "mode",
            "absorbed_power_W",
            "power_W",
            "value_W",
            "voltage_rms_V",
            "voltage_V",
            "value_V",
            "effective_impedance_ohm",
            "coupling_efficiency",
            "base_reduced_field_Td",
            "reduced_field_Td",
            "reduced_field_per_sqrt_W_Td",
            "reduced_field_per_sqrt_W",
            "self_bias_fraction",
            "plasma_potential_offset_V",
            "plasma_potential_per_sqrt_W",
        }
        mode = str(rf_setting(("control_mode", "mode"), "absorbed_power"))
        control = "voltage" if "voltage" in mode.lower() else "absorbed_power"
        model = {
            "kind": kind,
            "role": _normalized_rf_role(rf_setting(("role",), port.kind)),
            "frequency_Hz": float(
                rf_setting(("frequency_Hz", "carrier_frequency_Hz"), 13.56e6)
            ),
            "control": control,
            "effective_impedance_ohm": float(
                rf_setting(("effective_impedance_ohm",), 50.0)
            ),
            "coupling_efficiency": float(rf_setting(("coupling_efficiency",), 1.0)),
            "base_reduced_field_Td": float(
                rf_setting(("base_reduced_field_Td", "reduced_field_Td"), 0.0)
            ),
            "reduced_field_per_sqrt_W_Td": float(
                rf_setting(
                    ("reduced_field_per_sqrt_W_Td", "reduced_field_per_sqrt_W"),
                    0.0,
                )
            ),
            "self_bias_fraction": float(rf_setting(("self_bias_fraction",), 0.35)),
            "plasma_potential_offset_V": float(
                rf_setting(("plasma_potential_offset_V",), 0.0)
            ),
            "plasma_potential_per_sqrt_W": float(
                rf_setting(("plasma_potential_per_sqrt_W",), 0.0)
            ),
        }
        power = _first(static, ("absorbed_power_W", "power_W", "value_W"))
        voltage = _first(static, ("voltage_rms_V", "voltage_V", "value_V"))
        if power is not None:
            model["default_absorbed_power_W"] = float(power)
        if voltage is not None:
            model["default_voltage_rms_V"] = float(voltage)
    elif kind == "experimental.ccp":
        representative = _representative_power_command(commands)
        known = {
            "frequency_Hz",
            "carrier_frequency_Hz",
            "control_mode",
            "mode",
            "absorbed_power_W",
            "power_W",
            "value_W",
            "voltage_rms_V",
            "voltage_V",
            "value_V",
        }
        representative_mode = (
            _first(representative, ("control_mode", "mode"))
            if _power_command_magnitude(representative) > 0.0
            else None
        )
        mode = str(
            representative_mode
            if representative_mode is not None
            else _first(static, ("control_mode", "mode"), "absorbed_power")
        )
        model = {
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
        power = _first(static, ("absorbed_power_W", "power_W", "value_W"))
        voltage = _first(static, ("voltage_rms_V", "voltage_V", "value_V"))
        if power is not None:
            model["default_absorbed_power_W"] = float(power)
        if voltage is not None:
            model["default_voltage_rms_V"] = float(voltage)
    else:
        representative = _representative_power_command(commands)
        known = {
            "frequency_Hz",
            "carrier_frequency_Hz",
            "delivered_power_W",
            "power_W",
            "value_W",
            "value",
        }
        model = {
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

    _extra_keys(static, known, f"{prefix}.parameters", unused)
    return model


def _waveform(
    values: Mapping[str, Any], prefix: str, unused: set[str], warnings: list[str]
) -> tuple[dict[str, Any], bool, set[str]]:
    consumed = {"waveform"}
    raw = str(values.get("waveform", "cw")).strip().lower()
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
    waveform, is_off, waveform_keys = _waveform(raw, prefix, unused, warnings)
    consumed = set(waveform_keys) | {"mode", "control_mode", "zone_id"}
    command: dict[str, Any] = {"kind": kind}

    if "zone_id" in raw:
        unused.add(f"{prefix}.zone_id")

    mode = str(_first(raw, ("control_mode", "mode"), "absorbed_power"))
    control = "voltage" if "voltage" in mode.lower() else "absorbed_power"

    if kind == "prescribed_power":
        keys = ("absorbed_power_W", "power_W", "value_W", "value")
        power = _first(raw, keys)
        consumed.update(keys)
        if power is not None or is_off:
            command["absorbed_power_W"] = 0.0 if is_off else float(power)
        command["waveform"] = waveform
    elif kind == "dc_series":
        keys = ("source_voltage_V", "voltage_V", "value_V", "value")
        voltage = _first(raw, keys)
        consumed.update(keys)
        nested = raw.get("voltage")
        if isinstance(nested, Mapping):
            consumed.add("voltage")
            voltage = _first(
                nested,
                ("high_V", "on_V", "source_voltage_V", "voltage_V", "value_V", "value"),
                voltage,
            )
            command["off_voltage_V"] = float(_first(nested, ("low_V", "off_V"), 0.0))
            nested_waveform = dict(nested)
            if (
                "frequency_Hz" in nested_waveform
                and "repetition_Hz" not in nested_waveform
            ):
                nested_waveform["repetition_Hz"] = nested_waveform["frequency_Hz"]
            waveform, nested_off, _ = _waveform(
                nested_waveform, f"{prefix}.voltage", unused, warnings
            )
            is_off = is_off or nested_off
        if voltage is not None or is_off:
            command["source_voltage_V"] = 0.0 if is_off else float(voltage)
        static_aliases = (
            (
                "ballast_resistance_ohm",
                "series_resistance_ohm",
            ),
            ("gap_m", "electrode_gap_m"),
            ("electrode_area_m2", "area_m2"),
            ("power_absorption_fraction",),
            ("electron_mobility_m2_V_s",),
        )
        for names in static_aliases:
            consumed.update(names)
        command["waveform"] = waveform
    elif kind == "external_table":
        consumed.update(
            {
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
        )
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
            command["bounds_policy"] = (
                "hold"
                if str(legacy_bounds).lower() in {"edge", "clip", "hold"}
                else "error"
            )
    elif kind == "experimental.rf_envelope":
        consumed.update(
            {
                "role",
                "frequency_Hz",
                "carrier_frequency_Hz",
                "absorbed_power_W",
                "power_W",
                "value_W",
                "voltage_rms_V",
                "voltage_V",
                "value_V",
                "value",
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
        )
        power = _first(raw, ("absorbed_power_W", "power_W", "value_W"))
        voltage = _first(raw, ("voltage_rms_V", "voltage_V", "value_V"))
        if power is None and voltage is None and raw.get("value") is not None:
            if control == "voltage":
                voltage = raw["value"]
            else:
                power = raw["value"]
        if power is not None or (is_off and control == "absorbed_power"):
            command["absorbed_power_W"] = 0.0 if is_off else float(power)
        if voltage is not None or (is_off and control == "voltage"):
            command["voltage_rms_V"] = 0.0 if is_off else float(voltage)
        command["waveform"] = waveform
    elif kind == "experimental.ccp":
        consumed.update(
            {
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
        )
        power = _first(raw, ("absorbed_power_W", "power_W", "value_W"))
        voltage = _first(raw, ("voltage_rms_V", "voltage_V", "value_V"))
        if power is None and voltage is None and raw.get("value") is not None:
            if control == "voltage":
                voltage = raw["value"]
            else:
                power = raw["value"]
        if power is not None or (is_off and control == "absorbed_power"):
            command["absorbed_power_W"] = 0.0 if is_off else float(power)
        if voltage is not None or (is_off and control == "voltage"):
            command["voltage_rms_V"] = 0.0 if is_off else float(voltage)
        command["waveform"] = waveform
    else:
        consumed.update(
            {
                "frequency_Hz",
                "carrier_frequency_Hz",
                "delivered_power_W",
                "power_W",
                "value_W",
                "value",
            }
        )
        power = _first(raw, ("delivered_power_W", "power_W", "value_W", "value"))
        if power is not None or is_off:
            command["delivered_power_W"] = 0.0 if is_off else float(power)
        command["waveform"] = waveform

    _extra_keys(raw, consumed, prefix, unused)
    return command


def _electron_models(
    loaded: Any, unused: set[str], warnings: list[str]
) -> dict[str, Any]:
    run = loaded.run_config
    resolved = loaded.resolved_paths
    backend = str(run.physics.eedf_backend).strip().lower()
    if backend == "maxwell":
        electrons: dict[str, Any] = {"kind": "maxwellian"}
        closure_kind = "electron_energy"
    elif backend == "swarm" and str(run.swarm.model_name).lower() == "table":
        table = run.swarm.table
        if not table.file:
            raise MigrationError("v2 swarm table model has no table file")
        requested_lookup = str(table.lookup or run.swarm.closure or "auto").lower()
        closure_kind = (
            "local_field" if requested_lookup == "local_field" else "electron_energy"
        )
        electrons = {
            "kind": "table",
            "file": _path_from(table.file, Path(resolved.chemistry_dir)),
            "lookup": (
                "local_field" if closure_kind == "local_field" else "mean_energy"
            ),
            "bounds_policy": "error",
        }
        if table.electron_energy_mode:
            unused.add("swarm.table.electron_energy_mode")
            unused.add("swarm.table.energy_relaxation_time_s")
        if str(table.bounds_policy).lower() != "error":
            warnings.append(
                "swarm.table.bounds_policy was changed to 'error' to prevent "
                "silent extrapolation/clipping"
            )
    elif backend == "swarm" and str(run.swarm.model_name).lower() == "boltzmann_2term":
        cfg = run.swarm.boltzmann_2term
        requested_closure = str(run.swarm.closure or "auto").lower()
        closure_kind = "local_field"
        if requested_closure not in {"auto", "local_field"}:
            warnings.append(
                "boltzmann_2term closure was changed to local_field because the "
                "v3 approximate model is compile-time prepared and has no "
                "electron-energy RHS coupling"
            )
        electrons = {
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
            "reduced_field_grid": {
                "min_Td": float(cfg.reduced_field_grid_Td.min),
                "max_Td": float(cfg.reduced_field_grid_Td.max),
                "n": int(cfg.reduced_field_grid_Td.n),
            },
            "max_shape_iterations": int(cfg.max_shape_iterations),
        }
    else:
        raise MigrationError(
            f"unsupported v2 EEDF selection {backend!r}/{run.swarm.model_name!r}"
        )

    closure = str(run.physics.electron_density_closure).strip().lower()
    if closure == "quasi_neutral":
        density: dict[str, Any] = {"kind": "quasineutral"}
    elif closure == "prescribed_profile":
        cfg = run.swarm.prescribed_electron_profile
        if cfg.file:
            file = _path_from(cfg.file, Path(resolved.base_dir))
        elif cfg.file_key and resolved.external_inputs.get(str(cfg.file_key)):
            file = Path(str(resolved.external_inputs[str(cfg.file_key)])).resolve(
                strict=False
            )
        else:
            raise MigrationError("v2 prescribed electron profile has no input file")
        density = {
            "kind": "experimental.prescribed_profile",
            "file": file,
            "zone_columns": dict(cfg.zone_columns),
            "interpolation": str(cfg.interpolation).lower(),
            "hold": "error",
        }
        if str(cfg.hold).lower() != "error":
            warnings.append(
                "prescribed electron profile hold was changed to 'error' to "
                "prevent silent endpoint extension"
            )
    else:
        raise MigrationError(f"unsupported v2 electron density closure {closure!r}")

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
        "sample_interval_s": float(sample_interval_s),
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


def _initial_zone_densities(
    loaded: Any, zone: Any, warnings: list[str]
) -> dict[str, float]:
    gas_species = list(getattr(loaded.mechanism, "gas_state_species", []) or [])
    legacy_ion_seed_m3 = 1.0e13
    concrete = {
        str(species.canonical_id): (
            legacy_ion_seed_m3 if int(getattr(species, "charge", 0)) > 0 else 0.0
        )
        for species in gas_species
    }
    explicit = {
        str(species): float(density)
        for species, density in (zone.initial_densities_m3 or {}).items()
    }
    if explicit and any(value > 0.0 for value in explicit.values()):
        concrete.update(explicit)
        defaulted_ions = [
            str(species.canonical_id)
            for species in gas_species
            if int(getattr(species, "charge", 0)) > 0
            and str(species.canonical_id) not in explicit
        ]
        if defaulted_ions:
            warnings.append(
                f"zone {zone.zone_id!r} positive-ion seeds were made explicit at "
                f"{legacy_ion_seed_m3:g} m^-3 for {defaulted_ions}"
            )
        return concrete

    first_step = loaded.recipe.steps[0]
    flows: dict[str, float] = {}
    for inlet_id, species_flows in first_step.gas_inlets.items():
        inlet = loaded.chamber.inlet_by_id.get(inlet_id)
        if inlet is None or inlet.zone_id != zone.zone_id:
            continue
        for species, flow in species_flows.items():
            flows[str(species)] = flows.get(str(species), 0.0) + float(flow)
    if not flows:
        for species_flows in first_step.gas_inlets.values():
            for species, flow in species_flows.items():
                flows[str(species)] = flows.get(str(species), 0.0) + float(flow)

    neutral_ids = [
        str(species.canonical_id)
        for species in gas_species
        if int(getattr(species, "charge", 0)) == 0
    ]
    flows = {
        species: max(flow, 0.0)
        for species, flow in flows.items()
        if species in neutral_ids and flow > 0.0
    }
    if not flows:
        if not neutral_ids:
            raise MigrationError(
                f"zone {zone.zone_id!r} has no initial density and chemistry has no "
                "neutral gas species"
            )
        flows = {neutral_ids[0]: 1.0}

    k_b = 1.380649e-23
    total_density = float(zone.pressure_Pa) / (
        k_b * max(float(zone.gas_temperature_K), 1.0)
    )
    flow_total = sum(flows.values())
    derived = {
        species: total_density * flow / flow_total for species, flow in flows.items()
    }
    concrete.update(derived)
    seeded_ions = [
        str(species.canonical_id)
        for species in gas_species
        if int(getattr(species, "charge", 0)) > 0
    ]
    if seeded_ions:
        warnings.append(
            f"zone {zone.zone_id!r} positive-ion seeds were made explicit at "
            f"{legacy_ion_seed_m3:g} m^-3 for {seeded_ions}"
        )
    warnings.append(
        f"zone {zone.zone_id!r} initial_densities_m3 was derived from pressure, "
        "temperature, and the first recipe gas composition"
    )
    return concrete


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
    warnings: list[str] = [
        "v3 always records effective input and resolved-path provenance",
    ]
    used_external: set[str] = set()
    gas_fraction = float(run.physics.gas_heating_fraction)
    if not 0.0 <= gas_fraction <= 1.0:
        raise MigrationError("v2 physics.gas_heating_fraction must be between 0 and 1")
    models = _electron_models(loaded, unused, warnings)

    reactor: dict[str, Any] = {
        "chamber_id": str(chamber.chamber_id),
        "description": str(chamber.description),
        "zones": [],
        "edges": [],
        "surfaces": [],
        "gas_inlets": [],
        "pumps": [],
        "power_ports": [],
    }
    for index, zone in enumerate(chamber.zones):
        if str(zone.role) != "process":
            unused.add(f"reactor.zones[{index}].role")
        reactor["zones"].append(
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
    for index, edge in enumerate(chamber.edges):
        if edge.notes:
            unused.add(f"reactor.edges[{index}].notes")
        reactor["edges"].append(
            {
                "edge_id": str(edge.edge_id),
                "from_zone": str(edge.from_zone),
                "to_zone": str(edge.to_zone),
                "conductance_m3_s": float(edge.conductance_m3_s),
            }
        )
    free_site_ids = {
        str(species.canonical_id)
        for species in getattr(loaded.mechanism, "surface_species", [])
        if "site" in set(getattr(species, "state_tags", set()) or set())
    }
    film_fragment_ids = {
        str(species.canonical_id)
        for species in getattr(loaded.mechanism, "surface_species", [])
        if "film_fragment" in set(getattr(species, "state_tags", set()) or set())
    }
    for index, surface in enumerate(chamber.surfaces):
        unused.add(f"reactor.surfaces[{index}].kind")
        unused.add(f"reactor.surfaces[{index}].material")
        coverages: dict[str, float] = {}
        for key, value in surface.initial_coverages.items():
            if str(key) in free_site_ids or str(key) in film_fragment_ids:
                unused.add(f"reactor.surfaces[{index}].initial_coverages.{key}")
                continue
            coverages[str(key)] = float(value)
        reactor["surfaces"].append(
            {
                "surface_id": str(surface.surface_id),
                "zone_id": str(surface.zone_id),
                "area_m2": float(surface.area_m2),
                "temperature_K": float(surface.temperature_K),
                "site_density_m2": float(surface.site_density_m2),
                "initial_coverages": coverages,
                "wall_transport": _wall_transport(
                    surface.models,
                    f"reactor.surfaces[{index}].models",
                    unused,
                ),
            }
        )
    for inlet in chamber.gas_inlets:
        reactor["gas_inlets"].append(
            {
                "inlet_id": str(inlet.inlet_id),
                "zone_id": str(inlet.zone_id),
                "flow_sccm": {
                    str(key): float(value) for key, value in inlet.flow_sccm.items()
                },
                "temperature_K": float(inlet.temperature_K),
            }
        )
    for pump in chamber.pumps:
        reactor["pumps"].append(
            {
                "pump_id": str(pump.pump_id),
                "zone_id": str(pump.zone_id),
                "speed_m3_s": float(pump.speed_m3_s),
            }
        )

    legacy_base_dir = Path(resolved.base_dir)
    port_kind_by_id: dict[str, str] = {}
    port_model_by_id: dict[str, Mapping[str, Any]] = {}
    for index, port in enumerate(chamber.power_ports):
        prefix = f"reactor.power_ports[{index}]"
        model = _port_model(
            port=port,
            backend=str(run.physics.electrical_backend),
            recipe=recipe,
            legacy_base_dir=legacy_base_dir,
            external_inputs=resolved.external_inputs,
            used_external=used_external,
            unused=unused,
            prefix=prefix,
            gas_fraction=gas_fraction,
        )
        port_kind_by_id[str(port.port_id)] = str(model["kind"])
        port_model_by_id[str(port.port_id)] = model
        reactor["power_ports"].append(
            {
                "port_id": str(port.port_id),
                "zone_id": str(port.zone_id),
                "coupling_target": str(port.coupling_target),
                "model": model,
            }
        )

    recipe_steps: list[dict[str, Any]] = []
    previous_end: float | None = None
    recipe_dir = Path(resolved.recipe_file).parent
    for index, step in enumerate(recipe.steps):
        start = float(step.t_start_s)
        end = float(step.t_end_s)
        if previous_end is not None and abs(start - previous_end) > max(
            1.0e-15, 1.0e-12 * max(abs(start), abs(previous_end), 1.0)
        ):
            warnings.append(
                f"recipe.steps[{index}] began at {start:g}s instead of the previous "
                f"end {previous_end:g}s; v3 makes steps contiguous"
            )
        previous_end = end
        power_commands: dict[str, Any] = {}
        for port_id, values in step.power_ports.items():
            port_id = str(port_id)
            if port_id not in port_kind_by_id:
                raise MigrationError(
                    f"recipe.steps[{index}] references unknown power port {port_id!r}"
                )
            migrated_command = _power_command(
                values=dict(values or {}),
                kind=port_kind_by_id[port_id],
                base_dir=recipe_dir,
                external_inputs=resolved.external_inputs,
                used_external=used_external,
                prefix=f"recipe.steps[{index}].power_ports.{port_id}",
                unused=unused,
                warnings=warnings,
            )
            power_commands[port_id] = _match_experimental_setpoint_to_model(
                migrated_command,
                port_model_by_id[port_id],
                f"recipe.steps[{index}].power_ports.{port_id}",
            )

        surfaces: dict[str, Any] = {}
        for surface_id, values in step.surface_overrides.items():
            values = dict(values or {})
            prefix = f"recipe.steps[{index}].surface_overrides.{surface_id}"
            _extra_keys(values, {"temperature_K"}, prefix, unused)
            surfaces[str(surface_id)] = {
                "temperature_K": _float_or_none(values.get("temperature_K"))
            }
        if step.imported_inputs:
            unused.update(
                f"recipe.steps[{index}].imported_inputs.{key}"
                for key in step.imported_inputs
            )
        recipe_steps.append(
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

    for key in resolved.external_inputs:
        if key not in used_external:
            unused.add(f"files.external_inputs.{key}")

    if not any(
        port["model"]["kind"] == "prescribed_power" for port in reactor["power_ports"]
    ):
        unused.add("physics.gas_heating_fraction")

    formats = run.outputs.formats
    plots = run.outputs.plots
    if not formats.solution_h5:
        warnings.append("v3 always writes the HDF5 result bundle")
    if not formats.summary_yaml:
        warnings.append("v3 always writes summary.yaml")
    if formats.observables_csv:
        unused.add("outputs.formats.observables_csv")
    if plots.enabled:
        unused.add("outputs.plots")
    if run.outputs.budgets.enabled:
        unused.add("outputs.budgets")
    unused.add("files.output_dir")
    output: dict[str, Any] = {}
    total_duration = sum(item["duration_s"] for item in recipe_steps)
    sample_interval_s = max(float(total_duration) / 199.0, 1.0e-15)
    experimental: dict[str, Any] = {}
    quasi_steady = _experimental_quasi_steady(run, unused)
    if quasi_steady is not None:
        experimental["stop_when_quasi_steady"] = quasi_steady
    has_film_state = any(
        "film_fragment" in set(getattr(species, "state_tags", ()) or ())
        for species in loaded.mechanism.surface_species
    )
    if run.physics.enable_surface_coverages and has_film_state:
        experimental["film"] = {}
    initial_inventory = {
        str(surface.surface_id): {
            str(key): float(value) for key, value in surface.initial_inventory.items()
        }
        for surface in chamber.surfaces
        if surface.initial_inventory
    }
    if run.physics.enable_wall_inventory and initial_inventory:
        gas_species_ids = {
            str(species.canonical_id)
            for species in loaded.mechanism.gas_state_species
            if str(species.canonical_id) != "e"
        }
        event_yields: dict[tuple[str, str], dict[str, float]] = {}
        for surface_id, inventory_entries in initial_inventory.items():
            for inventory_id in inventory_entries:
                matching_reactions: list[Any] = []
                for reaction in loaded.mechanism.surface_reactions:
                    if not reaction.enabled or (
                        reaction.surface_filter
                        and surface_id not in reaction.surface_filter
                    ):
                        continue
                    gas_reactants = [
                        species_id
                        for species_id in reaction.gas_reactants
                        if species_id in gas_species_ids
                    ]
                    if (
                        gas_reactants
                        and f"{gas_reactants[0]}_reservoir" == inventory_id
                    ):
                        matching_reactions.append(reaction)
                if len(matching_reactions) != 1:
                    raise MigrationError(
                        "cannot migrate wall inventory without one unambiguous "
                        f"surface event: {surface_id}.{inventory_id} matched "
                        f"{[item.reaction_id for item in matching_reactions]}"
                    )
                reaction = matching_reactions[0]
                event_yields.setdefault((str(reaction.reaction_id), surface_id), {})[
                    inventory_id
                ] = 1.0
        experimental["wall_inventory"] = {
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
        warnings.append(
            "wall inventory event yields were made explicit from the unique v2 "
            "unit-per-event mapping; review each migrated yield"
        )
    if loaded.mechanism.state_variables or loaded.mechanism.processes:
        experimental["extensions"] = {}
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
    """Write a migrated, self-contained v3 YAML document."""

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


def _stage_case_assets(
    case: CaseSpec,
    staging_directory: Path,
    published_directory: Path,
) -> tuple[CaseSpec, tuple[str, ...]]:
    """Bundle runtime file dependencies and replace their paths in a case."""

    from tools.importers.rate_table import convert_v2_rate_table_h5

    data = case.model_dump(mode="python", exclude_none=True)
    staged: dict[tuple[Path, str], Path] = {}
    warnings: list[str] = []

    def stage(value: Any, label: str, *, electron_table: bool = False) -> Path:
        source = Path(value).resolve()
        if not source.is_file():
            raise MigrationError(f"{label} does not exist: {source}")
        conversion = "electron_table" if electron_table else "copy"
        key = (source, conversion)
        if key in staged:
            return staged[key]
        suffix = source.suffix or ".dat"
        filename = f"{len(staged):02d}_{label.replace('.', '_')}{suffix}"
        temporary_path = staging_directory / filename
        published_path = (published_directory / filename).resolve()
        staging_directory.mkdir(parents=True, exist_ok=True)
        try:
            if electron_table:
                _, dropped = convert_v2_rate_table_h5(source, temporary_path)
                if dropped:
                    warnings.append(
                        f"{label}: dropped unused legacy HDF5 datasets: "
                        f"{', '.join(dropped)}"
                    )
            else:
                shutil.copy2(source, temporary_path)
        except (OSError, ValueError) as exc:
            raise MigrationError(
                f"cannot migrate {label} from {source}: {exc}"
            ) from exc
        staged[key] = published_path
        return published_path

    models = data["models"]
    electrons = models["electrons"]
    if electrons["kind"] == "table":
        electrons["file"] = stage(
            electrons["file"], "models_electrons", electron_table=True
        )
    electron_density = models["electron_density"]
    if electron_density["kind"] == "experimental.prescribed_profile":
        electron_density["file"] = stage(
            electron_density["file"], "models_electron_density"
        )

    for port_index, port in enumerate(data["reactor"]["power_ports"]):
        model = port["model"]
        if model["kind"] == "external_table":
            model["file"] = stage(
                model["file"], f"power_model_{port_index}_{port['port_id']}"
            )

    for step_index, step in enumerate(data["recipe"]["steps"]):
        commands = step["commands"]["power_ports"]
        for port_id, command in commands.items():
            if command["kind"] == "external_table" and "file" in command:
                command["file"] = stage(
                    command["file"],
                    f"power_command_{step_index}_{port_id}",
                )

    try:
        migrated = CaseSpec.model_validate(data)
    except Exception as exc:
        raise MigrationError(f"bundled v3 case failed validation: {exc}") from exc
    return migrated, tuple(warnings)


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
