"""Cross-reference checks on the public declaration, before physical compilation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from plasma_global.input._schema_power import parse_power_command

if TYPE_CHECKING:
    from plasma_global.input.schema import (
        CaseSpec,
        InitialConditions,
        ReactorDefinition,
    )


def validate_composition(initial: InitialConditions) -> None:
    if (initial.densities_m3 is None) == (initial.mole_fractions is None):
        raise ValueError("set exactly one of densities_m3 or mole_fractions")
    if initial.densities_m3 is not None:
        if initial.seed_densities_m3:
            raise ValueError("seed_densities_m3 requires mole_fractions")
        if not any(value > 0.0 for value in initial.densities_m3.values()):
            raise ValueError("densities_m3 must contain a positive density")
    elif not initial.mole_fractions or not math.isclose(
        sum(initial.mole_fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("mole_fractions must sum to one")


def validate_topology(reactor: ReactorDefinition) -> None:
    if not reactor.zones:
        raise ValueError("reactor.zones must contain at least one zone")
    zones = set(reactor.zones)
    for name in ("surfaces", "gas_inlets", "pumps", "power_ports"):
        for identity, entity in getattr(reactor, name).items():
            if entity.zone_id not in zones:
                raise ValueError(
                    f"reactor.{name}.{identity}.zone_id references "
                    f"unknown zone {entity.zone_id!r}"
                )
    for identity, edge in reactor.edges.items():
        if unknown := {edge.from_zone, edge.to_zone} - zones:
            raise ValueError(
                f"reactor.edges.{identity}: unknown zones {sorted(unknown)}"
            )


def _validate_commands(case: CaseSpec) -> None:
    for index, step in enumerate(case.run.normalized_steps(), 1):
        for family in ("power_ports", "gas_inlets", "surfaces"):
            commands = getattr(step.commands, family)
            if unknown := set(commands) - set(getattr(case.reactor, family)):
                raise ValueError(
                    f"run.steps.{index}.commands.{family}: "
                    f"unknown IDs {sorted(unknown)}"
                )
        for identity, values in step.commands.power_ports.items():
            try:
                parse_power_command(case.reactor.power_ports[identity].model, values)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"run.steps.{index}.commands.power_ports.{identity}: {exc}"
                ) from exc


def _validate_initial_electrons(case: CaseSpec) -> None:
    energy_balance = case.models.electrons.kind == "energy_balance"
    for identity, zone in case.reactor.zones.items():
        if energy_balance and zone.initial.mean_energy_eV is None:
            raise ValueError(
                f"reactor.zones.{identity}.initial.mean_energy_eV "
                "is required for energy_balance"
            )
        if not energy_balance and zone.initial.mean_energy_eV is not None:
            raise ValueError(
                f"reactor.zones.{identity}.initial.mean_energy_eV "
                "is incompatible with local_field"
            )


def _validate_inventory(case: CaseSpec) -> None:
    config = None if case.experimental is None else case.experimental.wall_inventory
    if config is None:
        return
    if unknown := set(config.initial_by_surface) - set(case.reactor.surfaces):
        raise ValueError(
            f"experimental.wall_inventory: unknown surfaces {sorted(unknown)}"
        )
    for event in config.events:
        if event.surface_id not in case.reactor.surfaces:
            raise ValueError(
                "experimental.wall_inventory.events: "
                f"unknown surface {event.surface_id!r}"
            )
        declared = config.initial_by_surface.get(event.surface_id, {})
        if missing := set(event.inventory_particles_per_event) - set(declared):
            raise ValueError(
                "experimental.wall_inventory.events: "
                f"undeclared inventories {sorted(missing)}"
            )


def _validate_enabled_models(case: CaseSpec) -> None:
    if case.models.surface_kinetics is None:
        for identity, surface in case.reactor.surfaces.items():
            if surface.initial_coverages:
                raise ValueError(
                    f"reactor.surfaces.{identity}.initial_coverages requires "
                    "models.surface_kinetics"
                )
    gas = case.models.gas_energy
    if gas.kind == "evolved":
        unknown = set(gas.wall_energy_relaxation_s_inv_by_zone) - set(
            case.reactor.zones
        )
        if unknown:
            raise ValueError(f"models.gas_energy: unknown zones {sorted(unknown)}")


def _validate_local_field_power(case: CaseSpec) -> None:
    if case.models.electrons.kind != "local_field":
        return
    for identity, port in case.reactor.power_ports.items():
        model = port.model
        if model.kind == "dc_series" and model.electron_mobility_m2_V_s is not None:
            raise ValueError(
                f"reactor.power_ports.{identity}: local_field uses table mobility"
            )
        if model.kind == "experimental.icp" and model.downstream_fraction > 0.0:
            raise ValueError(
                f"reactor.power_ports.{identity}.model.downstream_fraction: "
                "nonzero downstream heating requires models.electrons.kind="
                "'energy_balance'; local_field does not integrate electron energy"
            )


def validate_case(case: CaseSpec) -> None:
    _validate_commands(case)
    _validate_initial_electrons(case)
    _validate_inventory(case)
    _validate_enabled_models(case)
    _validate_local_field_power(case)
    times = case.solver.save_at_s
    if times is not None and (
        times[0] < case.run.start_time_s or times[-1] > case.run.end_time_s
    ):
        raise ValueError("solver.save_at_s must lie within run interval")
