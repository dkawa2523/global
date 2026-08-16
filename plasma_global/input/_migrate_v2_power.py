"""Coordinate schema-v2 power-port and recipe-command migration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_power_commands import (
    match_experimental_setpoint_to_model,
    power_command,
)
from plasma_global.input._migrate_v2_power_models import port_model


def migrate_power_ports(
    *,
    chamber: Any,
    run: Any,
    recipe: Any,
    resolved: Any,
    gas_fraction: float,
    used_external: set[str],
    unused: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Translate reactor power ports and retain models for command validation."""

    ports: list[dict[str, Any]] = []
    model_by_id: dict[str, Mapping[str, Any]] = {}
    for index, port in enumerate(chamber.power_ports):
        model = port_model(
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
        migrated_port = {
            "port_id": port_id,
            "zone_id": str(port.zone_id),
            "model": model,
        }
        target = str(port.coupling_target)
        if model["kind"] == "experimental.ccp" and target:
            migrated_port["coupling_target"] = target
        elif target:
            unused.add(f"reactor.power_ports[{index}].coupling_target")
        ports.append(migrated_port)
    return ports, model_by_id


def migrate_step_power_commands(
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
    """Translate all power commands from one legacy recipe step."""

    commands: dict[str, Any] = {}
    for raw_port_id, values in step.power_ports.items():
        port_id = str(raw_port_id)
        if port_id not in port_model_by_id:
            raise MigrationError(
                f"recipe.steps[{step_index}] references unknown power port {port_id!r}"
            )
        port_model = port_model_by_id[port_id]
        prefix = f"recipe.steps[{step_index}].power_ports.{port_id}"
        migrated = power_command(
            values=dict(values or {}),
            kind=str(port_model["kind"]),
            base_dir=recipe_dir,
            external_inputs=external_inputs,
            used_external=used_external,
            prefix=prefix,
            unused=unused,
            warnings=warnings,
        )
        commands[port_id] = match_experimental_setpoint_to_model(
            migrated, port_model, prefix
        )
    return commands
