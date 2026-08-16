"""Translate schema-v2 reactor geometry, initial state, and wall transport.

Power model details remain in the power migration modules; this module owns the
reactor aggregate and preserves the legacy traversal order used for diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_common import record_extra_keys
from plasma_global.input._migrate_v2_power import migrate_power_ports

_BOHM_KEYS = (
    "characteristic_length_m",
    "h_factor",
    "min_h_factor",
    "max_h_factor",
    "ion_neutral_cross_section_m2",
)


def _bohm_wall_transport(raw: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    transport: dict[str, Any] = {"kind": "bohm"}
    for key in _BOHM_KEYS:
        if raw.get(key) is None:
            continue
        value = raw[key]
        transport[key] = (
            "auto"
            if key == "h_factor" and str(value).lower() == "auto"
            else float(value)
        )
    return transport, {"ion_loss", *_BOHM_KEYS}


def _prescribed_wall_transport(
    raw: Mapping[str, Any], prefix: str
) -> tuple[dict[str, Any], set[str]]:
    if raw.get("frequency_s") is None:
        raise MigrationError(f"{prefix}: prescribed ion loss needs frequency_s")
    return (
        {
            "kind": "prescribed_frequency",
            "frequency_s_inv": float(raw["frequency_s"]),
        },
        {"ion_loss", "frequency_s"},
    )


def _ambipolar_wall_transport(
    raw: Mapping[str, Any], prefix: str
) -> tuple[dict[str, Any], set[str]]:
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
    return transport, {
        "ion_loss",
        "diffusion_coefficient_m2_s",
        "diffusion_length_m",
    }


def _wall_transport(
    values: Mapping[str, Any], prefix: str, unused: set[str]
) -> dict[str, Any]:
    raw = dict(values or {})
    mode = str(raw.get("ion_loss", "bohm")).strip().lower()
    if mode == "bohm":
        transport, known = _bohm_wall_transport(raw)
    elif mode == "prescribed_loss_frequency":
        transport, known = _prescribed_wall_transport(raw, prefix)
    elif mode == "ambipolar_diffusion":
        transport, known = _ambipolar_wall_transport(raw, prefix)
    elif mode == "off":
        transport, known = {"kind": "off"}, {"ion_loss"}
    else:
        raise MigrationError(f"{prefix}: unsupported ion_loss mode {mode!r}")
    record_extra_keys(raw, known, prefix, unused)
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


def migrate_reactor(
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
    ports, model_by_id = migrate_power_ports(
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
