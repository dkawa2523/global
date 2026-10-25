"""Lower the public input once to explicit configuration used by compilers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from plasma_global.input._resolved_case import ResolvedCase

if TYPE_CHECKING:
    from plasma_global.input.schema import CaseSpec


_POWER_VALUES = {
    "prescribed_power": ("absorbed_power_W", "default_absorbed_power_W"),
    "experimental.rf_envelope": ("absorbed_power_W", "default_absorbed_power_W"),
    "experimental.ccp": ("absorbed_power_W", "default_absorbed_power_W"),
    "experimental.icp": ("delivered_power_W", "default_delivered_power_W"),
}


def absolute_file(value: Any, base: Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("file must be a nonempty path")
    return (base / value).resolve()


def _electron_models(config: dict[str, Any], base: Path) -> dict[str, Any]:
    mode = config["kind"]
    coefficients = config["coefficients"]
    if "file" in coefficients:
        source = {
            "kind": "table",
            "file": absolute_file(coefficients["file"], base),
            "columns": coefficients["columns"],
            "provenance": coefficients["provenance"],
            "lookup": "mean_energy" if mode == "energy_balance" else "local_field",
            "bounds_policy": "error" if coefficients["bounds"] == "error" else "clip",
            "mobility_reference_neutral_density_m3": coefficients.get(
                "mobility_reference_neutral_density_m3"
            ),
        }
    else:
        source = coefficients
    return {
        "electrons": source,
        "electron_closure": {
            "kind": "electron_energy" if mode == "energy_balance" else "local_field"
        },
    }


def _power_model(model: dict[str, Any], base: Path) -> None:
    if any(name.startswith("default_") for name in model):
        raise ValueError("set constant power on the port without a default_ prefix")
    if (
        model["kind"] in {"experimental.rf_envelope", "experimental.ccp"}
        and "voltage_rms_V" in model
    ):
        model["default_voltage_rms_V"] = model.pop("voltage_rms_V")
    pair = _POWER_VALUES.get(model["kind"])
    if pair is not None and pair[0] in model:
        model[pair[1]] = model.pop(pair[0])
    if "file" in model:
        model["file"] = absolute_file(model["file"], base)
    if model["kind"] == "external_table":
        model["bounds_policy"] = model.pop("bounds", "error")
        model.pop("time_reference", None)


def _reactor(config: dict[str, Any], base: Path) -> dict[str, Any]:
    result = {key: config[key] for key in ("chamber_id", "description")}
    zones = []
    for zone_id, raw in config["zones"].items():
        zone = dict(raw)
        initial = zone.pop("initial")
        zone.update(
            {
                "zone_id": zone_id,
                "pressure_Pa": initial["pressure_Pa"],
                "gas_temperature_K": initial["gas_temperature_K"],
                "initial_densities_m3": initial["densities_m3"],
                "initial_mole_fractions": initial["mole_fractions"],
                "initial_seed_densities_m3": initial["seed_densities_m3"],
                "initial_mean_energy_eV": initial["mean_energy_eV"],
            }
        )
        zones.append(zone)
    result["zones"] = zones
    for field, id_key in (
        ("surfaces", "surface_id"),
        ("edges", "edge_id"),
        ("gas_inlets", "inlet_id"),
        ("pumps", "pump_id"),
        ("power_ports", "port_id"),
    ):
        result[field] = [
            dict(value, **{id_key: key}) for key, value in config[field].items()
        ]
    for port in result["power_ports"]:
        model = port["model"]
        _power_model(model, base)
    return result


def _power_clocks(
    values: dict[str, Any],
    kind: str,
    run_start: float,
    step_start: float,
    reference: str,
) -> None:
    if kind == "external_table":
        mode = values.pop("time_reference", reference)
        if mode not in {"run_start", "step_start"}:
            raise ValueError("time_reference must be run_start or step_start")
        offset = values.get("time_offset_s", 0.0)
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise ValueError("time_offset_s must be numeric")
        values["time_offset_s"] = offset + (
            run_start if mode == "run_start" else step_start
        )
    waveform = values.get("waveform", {})
    if waveform.get("kind") == "square_pulse":
        mode = waveform.pop("time_reference", "run_start")
        if mode not in {"run_start", "step_start"} or "origin_s" in waveform:
            raise ValueError("waveform requires time_reference, not internal origin_s")
        waveform["origin_s"] = run_start if mode == "run_start" else None


def _power_commands(
    commands: dict[str, Any],
    port_kinds: dict[str, str],
    base: Path,
    clocks: dict[str, str],
    run_start: float,
    step_start: float,
) -> None:
    for port_id, values in commands.get("power_ports", {}).items():
        if "kind" in values:
            raise ValueError("step power kind is determined by its declared port")
        values["kind"] = port_kinds.get(port_id, "unknown")
        if "file" in values:
            values["file"] = absolute_file(values["file"], base)
        if "bounds" in values:
            values["bounds_policy"] = values.pop("bounds")
        _power_clocks(
            values,
            values["kind"],
            run_start,
            step_start,
            clocks.get(port_id, "run_start"),
        )


def _run(
    config: dict[str, Any],
    ports: list[dict[str, Any]],
    base: Path,
    clocks: dict[str, str],
) -> dict[str, Any]:
    port_kinds = {port["port_id"]: port["model"]["kind"] for port in ports}
    steps = config["steps"] or [{"duration_s": config["duration_s"], "commands": {}}]
    result = []
    step_start = config["start_time_s"]
    for index, raw in enumerate(steps):
        commands = raw.get("commands", {})
        for port_id, kind in port_kinds.items():
            if kind == "external_table":
                commands.setdefault("power_ports", {}).setdefault(port_id, {})
        _power_commands(
            commands, port_kinds, base, clocks, config["start_time_s"], step_start
        )
        result.append(
            {
                "step_id": f"step_{index + 1}",
                "label": raw.get("label"),
                "duration_s": raw["duration_s"],
                "commands": commands,
            }
        )
        step_start += raw["duration_s"]
    return {"recipe_id": "run", "start_time_s": config["start_time_s"], "steps": result}


def _output(output: dict[str, Any], zone_definitions: list[dict[str, Any]]) -> None:
    zones = output.pop("zones") or [zone["zone_id"] for zone in zone_definitions]
    if unknown := set(zones) - {zone["zone_id"] for zone in zone_definitions}:
        raise ValueError(f"output references unknown zones: {sorted(unknown)}")
    output["observables"].extend(
        f"{quantity}[{zone}]" for quantity in output.pop("quantities") for zone in zones
    )


def _validate_local_field_mobility(data: dict[str, Any]) -> None:
    if data["models"]["electrons"]["kind"] == "local_field":
        for port in data["reactor"]["power_ports"]:
            model = port["model"]
            if (
                model["kind"] == "dc_series"
                and model.get("electron_mobility_m2_V_s") is not None
            ):
                raise ValueError(
                    "local_field DC uses table mobility; remove configured mobility"
                )


def resolve_case(case: CaseSpec, base_directory: Path) -> ResolvedCase:
    data = case.model_dump(mode="python", by_alias=True)
    data.pop("schema_version")
    clocks = {
        key: port["model"].get("time_reference", "run_start")
        for key, port in data["reactor"]["power_ports"].items()
    }
    data["reactor"] = _reactor(data["reactor"], base_directory)
    data["recipe"] = _run(
        data.pop("run"), data["reactor"]["power_ports"], base_directory, clocks
    )
    models = data["models"]
    _validate_local_field_mobility(data)
    models.update(_electron_models(models["electrons"], base_directory))
    if models["surface_kinetics"] is not None:
        empty_surface: dict[str, Any] = {}
        models["surface_kinetics"] = empty_surface
    density = models["electron_density"]
    if "file" in density:
        density["file"] = absolute_file(density["file"], base_directory)
        density["hold"] = "error" if density.pop("bounds") == "error" else "edge"
        density["time_offset_s"] += data["recipe"]["start_time_s"]
    chemistry = data["chemistry"]
    if "file" in chemistry:
        if len(chemistry) != 1:
            raise ValueError(
                "chemistry.file cannot be combined with inline definitions"
            )
        data["chemistry"] = {
            "manifest": absolute_file(chemistry["file"], base_directory)
        }
    else:
        data["chemistry"] = {"definition": chemistry, "base_directory": base_directory}
    _output(data["output"], data["reactor"]["zones"])
    return ResolvedCase.model_validate(data)
