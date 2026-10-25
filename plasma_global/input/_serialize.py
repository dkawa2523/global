"""Serialize explicit compilation settings back to the public YAML schema."""

from __future__ import annotations

from typing import Any

from plasma_global.input._resolve import _POWER_VALUES
from plasma_global.input._resolved_case import ResolvedCase


def _public_power_model(model: dict[str, Any]) -> None:
    pair = _POWER_VALUES.get(model["kind"])
    if pair is not None and pair[1] in model:
        model[pair[0]] = model.pop(pair[1])
    if "default_voltage_rms_V" in model:
        model["voltage_rms_V"] = model.pop("default_voltage_rms_V")
    if model["kind"] == "external_table":
        model["bounds"] = model.pop("bounds_policy")


def _reactor_document(reactor: dict[str, Any]) -> dict[str, Any]:
    identities = (
        ("zones", "zone_id"),
        ("surfaces", "surface_id"),
        ("edges", "edge_id"),
        ("gas_inlets", "inlet_id"),
        ("pumps", "pump_id"),
        ("power_ports", "port_id"),
    )
    for field, id_key in identities:
        reactor[field] = {item.pop(id_key): item for item in reactor[field]}
    for zone in reactor["zones"].values():
        initial = {
            key.removeprefix("initial_"): zone.pop(key)
            for key in tuple(zone)
            if key.startswith("initial_")
        }
        for key in ("pressure_Pa", "gas_temperature_K"):
            initial[key] = zone.pop(key)
        zone["initial"] = initial
    for port in reactor["power_ports"].values():
        model = port["model"]
        _public_power_model(model)
    return reactor


def _models_document(models: dict[str, Any], start: float) -> dict[str, Any]:
    source = models["electrons"]
    mode = models.pop("electron_closure")["kind"]
    if source["kind"] == "table":
        coefficients = {
            "file": source["file"],
            "columns": source.get("columns", {}),
            "provenance": source.get("provenance", {}),
            "bounds": "error" if source["bounds_policy"] == "error" else "hold",
        }
        if "mobility_reference_neutral_density_m3" in source:
            coefficients["mobility_reference_neutral_density_m3"] = source[
                "mobility_reference_neutral_density_m3"
            ]
    else:
        coefficients = source
    models["electrons"] = {
        "kind": "energy_balance" if mode == "electron_energy" else "local_field",
        "coefficients": coefficients,
    }
    if "surface_kinetics" in models:
        models["surface_kinetics"] = {"kind": "mass_action"}
    density = models["electron_density"]
    if "hold" in density:
        density["bounds"] = "error" if density.pop("hold") == "error" else "hold"
        density["time_offset_s"] = density.get("time_offset_s", 0.0) - start
    return models


def _commands_document(
    commands: dict[str, Any], external_ports: list[str], start: float
) -> None:
    powers = commands.setdefault("power_ports", {})
    for port in external_ports:
        powers.setdefault(port, {"kind": "external_table", "time_offset_s": 0.0})
    for command in powers.values():
        kind = command.pop("kind")
        if kind == "external_table":
            command["time_offset_s"] = command.get("time_offset_s", 0.0) - start
        if "bounds_policy" in command:
            command["bounds"] = command.pop("bounds_policy")
        waveform = command.get("waveform", {})
        if waveform.get("kind") == "square_pulse":
            origin = waveform.pop("origin_s", None)
            waveform["time_reference"] = "step_start" if origin is None else "run_start"


def _run_document(recipe: dict[str, Any], reactor: dict[str, Any]) -> dict[str, Any]:
    steps = recipe["steps"]
    external_ports = [
        key
        for key, port in reactor["power_ports"].items()
        if port["model"]["kind"] == "external_table"
    ]
    for step in steps:
        identity = step.pop("step_id")
        step.setdefault("label", identity)
        _commands_document(step["commands"], external_ports, recipe["start_time_s"])
    if len(steps) == 1 and not any(steps[0]["commands"].values()):
        run = {"duration_s": steps[0]["duration_s"]}
    else:
        run = {"steps": steps}
    run["start_time_s"] = recipe["start_time_s"]
    return run


def case_document(case: ResolvedCase) -> dict[str, Any]:
    data = case.model_dump(mode="json", exclude_none=True)
    chemistry = data["chemistry"]
    data["chemistry"] = (
        {"file": chemistry["manifest"]}
        if "manifest" in chemistry
        else chemistry["definition"]
    )
    data["reactor"] = _reactor_document(data["reactor"])
    data["models"] = _models_document(data["models"], data["recipe"]["start_time_s"])
    data["run"] = _run_document(data.pop("recipe"), data["reactor"])
    return {"schema_version": 4, **data}
