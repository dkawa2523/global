"""Adapt schema-v3 reactor topology to static compiled components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, assert_never

import numpy as np

from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import Zone
from plasma_global.core.transport import CompiledTransport
from plasma_global.errors import CaseValidationError
from plasma_global.input.schema import (
    AmbipolarWallTransport,
    BohmWallTransport,
    CaseSpec,
    DCSeriesModel,
    EvolvedGasEnergy,
    ExperimentalCCPModel,
    ExperimentalICPModel,
    ExperimentalRFEnvelopeModel,
    ExternalTableCommand,
    ExternalTableModel,
    FixedGasEnergy,
    OffWallTransport,
    PowerPortConfig,
    PrescribedFrequencyWallTransport,
    PrescribedPowerModel,
    SurfaceConfig,
)
from plasma_global.models.external_table import (
    ExternalTableBinding,
    ExternalTableStore,
)
from plasma_global.models.gas_energy import HeavyEnergyClosure
from plasma_global.models.power import (
    DCSeriesPort,
    ExternalTablePowerPort,
    PowerCoordinator,
    PowerPort,
    PrescribedPowerPort,
)
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry
from plasma_global.models.walls import BoundaryReaction, WallBoundary

_BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True, slots=True)
class CompiledReactor:
    zones: tuple[Zone, ...]
    transport: CompiledTransport
    wall_boundaries: tuple[WallBoundary, ...]
    power_coordinator: PowerCoordinator | None
    heavy_energy_closure: HeavyEnergyClosure | None
    surface_model: CompiledSurfaceModel | None


def compile_initial_densities(
    case: CaseSpec, chemistry: CompiledChemistry
) -> Mapping[str, Mapping[str, float]]:
    """Resolve either explicit densities or pressure-normalized composition once."""

    species_index = {
        species_id: index for index, species_id in enumerate(chemistry.species_ids)
    }
    resolved: dict[str, Mapping[str, float]] = {}
    for zone in case.reactor.zones:
        if zone.initial_densities_m3 is not None:
            densities = {
                species_id: float(value)
                for species_id, value in zone.initial_densities_m3.items()
            }
            unknown = set(densities) - set(species_index)
            if unknown:
                raise CaseValidationError(
                    f"zone {zone.zone_id!r} initializes unknown species {sorted(unknown)}"
                )
            pressure_from_state = (
                sum(densities.values()) * _BOLTZMANN_J_K * zone.gas_temperature_K
            )
            if not math.isclose(
                pressure_from_state,
                zone.pressure_Pa,
                rel_tol=1.0e-6,
                abs_tol=1.0e-12,
            ):
                raise CaseValidationError(
                    f"zone {zone.zone_id!r} pressure_Pa={zone.pressure_Pa:g} is "
                    "inconsistent with sum(initial_densities_m3)*kB*T="
                    f"{pressure_from_state:g}"
                )
        else:
            assert zone.initial_mole_fractions is not None
            fractions = {
                species_id: float(value)
                for species_id, value in zone.initial_mole_fractions.items()
            }
            seeds = {
                species_id: float(value)
                for species_id, value in zone.initial_seed_densities_m3.items()
            }
            unknown = (set(fractions) | set(seeds)) - set(species_index)
            if unknown:
                raise CaseValidationError(
                    f"zone {zone.zone_id!r} initializes unknown species {sorted(unknown)}"
                )
            nonneutral_fractions = [
                species_id
                for species_id in fractions
                if chemistry.charges[species_index[species_id]] != 0.0
            ]
            neutral_seeds = [
                species_id
                for species_id in seeds
                if chemistry.charges[species_index[species_id]] == 0.0
            ]
            if nonneutral_fractions or neutral_seeds:
                raise CaseValidationError(
                    f"zone {zone.zone_id!r} mole fractions must be neutral and seed "
                    "densities charged; invalid "
                    f"fractions={sorted(nonneutral_fractions)}, seeds={sorted(neutral_seeds)}"
                )
            total_density = zone.pressure_Pa / (_BOLTZMANN_J_K * zone.gas_temperature_K)
            seed_density = sum(seeds.values())
            if seed_density >= total_density:
                raise CaseValidationError(
                    f"zone {zone.zone_id!r} seed density must be below p/(kB*T)"
                )
            neutral_density = total_density - seed_density
            densities = {
                species_id: fraction * neutral_density
                for species_id, fraction in fractions.items()
            }
            densities.update(seeds)
        positive_charge_density = sum(
            chemistry.charges[species_index[species_id]] * density
            for species_id, density in densities.items()
            if chemistry.charges[species_index[species_id]] > 0.0
        )
        if positive_charge_density <= 0.0:
            raise CaseValidationError(
                f"zone {zone.zone_id!r} requires an explicit positive-ion seed"
            )
        resolved[zone.zone_id] = MappingProxyType(densities)
    return MappingProxyType(resolved)


def validate_surface_initial_conditions(
    case: CaseSpec, chemistry_data: ChemistryData
) -> None:
    """Validate independent adsorbates after chemistry IDs are available."""

    surface_species = [
        item for item in chemistry_data.species if item.phase == "surface"
    ]
    for surface in case.reactor.surfaces:
        applicable = [
            item
            for item in surface_species
            if not item.surfaces or surface.surface_id in item.surfaces
        ]
        free_sites = {item.id for item in applicable if "site" in item.state_tags}
        supplied = set(surface.initial_coverages)
        if explicit_free := supplied & free_sites:
            raise CaseValidationError(
                f"surface {surface.surface_id!r} must not initialize algebraic "
                f"free site(s) {sorted(explicit_free)}"
            )
        independent = {
            item.id
            for item in applicable
            if item.id not in free_sites and "film_fragment" not in item.state_tags
        }
        if unknown := supplied - independent:
            raise CaseValidationError(
                f"surface {surface.surface_id!r} initializes unknown or dependent "
                f"coverage species {sorted(unknown)}"
            )
        occupancy = {
            item.id: float(item.elements.get("site", 1.0)) for item in applicable
        }
        occupied = sum(
            occupancy[species_id] * coverage
            for species_id, coverage in surface.initial_coverages.items()
        )
        if occupied > 1.0 + 1.0e-12:
            raise CaseValidationError(
                f"surface {surface.surface_id!r} initial site occupancy exceeds one"
            )


def compile_external_binding(
    model: ExternalTableModel,
    store: ExternalTableStore,
    command: ExternalTableCommand | None = None,
) -> ExternalTableBinding:
    """Resolve a static external-table model and one optional step override."""

    source = model.file if command is None or command.file is None else command.file
    interpolation = (
        model.interpolation
        if command is None or command.interpolation is None
        else command.interpolation
    )
    bounds_policy = (
        model.bounds_policy
        if command is None or command.bounds_policy is None
        else command.bounds_policy
    )
    power_scale = (
        model.power_scale
        if command is None or command.power_scale is None
        else command.power_scale
    )
    voltage_scale = (
        model.voltage_scale
        if command is None or command.voltage_scale is None
        else command.voltage_scale
    )
    current_scale = (
        model.current_scale
        if command is None or command.current_scale is None
        else command.current_scale
    )
    gap_m = model.gap_m if command is None or command.gap_m is None else command.gap_m
    total_density_m3 = (
        model.total_density_m3
        if command is None or command.total_density_m3 is None
        else command.total_density_m3
    )
    plasma_potential_V = (
        model.plasma_potential_V
        if command is None or command.plasma_potential_V is None
        else command.plasma_potential_V
    )
    return ExternalTableBinding(
        data=store.get(source),
        interpolation=interpolation,
        bounds=bounds_policy,
        power_scale=float(power_scale),
        voltage_scale=float(voltage_scale),
        current_scale=float(current_scale),
        gap_m=gap_m,
        total_density_m3=total_density_m3,
        plasma_potential_V=float(plasma_potential_V),
        time_offset_s=0.0 if command is None else float(command.time_offset_s),
    )


def _compile_zones(case: CaseSpec) -> tuple[Zone, ...]:
    return tuple(
        Zone(zone.zone_id, zone.volume_m3, zone.gas_temperature_K)
        for zone in case.reactor.zones
    )


def _compile_static_transport(
    case: CaseSpec, chemistry: CompiledChemistry
) -> CompiledTransport:
    zones = tuple(case.reactor.zones)
    zone_index = {zone.zone_id: index for index, zone in enumerate(zones)}
    pump_frequency = np.zeros(len(zones))
    for pump in case.reactor.pumps:
        index = zone_index[pump.zone_id]
        pump_frequency[index] += pump.speed_m3_s / zones[index].volume_m3
    return CompiledTransport(
        volumes_m3=np.asarray([zone.volume_m3 for zone in zones]),
        pump_frequency_s_inv=pump_frequency,
        edge_from=np.asarray(
            [zone_index[edge.from_zone] for edge in case.reactor.edges], dtype=int
        ),
        edge_to=np.asarray(
            [zone_index[edge.to_zone] for edge in case.reactor.edges], dtype=int
        ),
        edge_conductance_m3_s=np.asarray(
            [edge.conductance_m3_s for edge in case.reactor.edges]
        ),
        n_species=len(chemistry.species_ids),
    )


def _compile_heavy_energy(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> HeavyEnergyClosure | None:
    config = case.models.gas_energy
    if isinstance(config, FixedGasEnergy):
        return None
    if not isinstance(config, EvolvedGasEnergy):
        assert_never(config)
    species = {item.id: item for item in chemistry_data.species}
    missing_cv = [
        species_id
        for species_id in chemistry.species_ids
        if species[species_id].cv_over_kb is None
    ]
    if missing_cv:
        raise CaseValidationError(
            "evolved gas energy requires cv_over_kb for every heavy gas species; "
            f"missing {missing_cv}"
        )
    cv_over_kb = np.asarray(
        [species[species_id].cv_over_kb for species_id in chemistry.species_ids],
        dtype=float,
    )
    wall_temperatures: list[float] = []
    for zone in case.reactor.zones:
        surfaces = [
            item for item in case.reactor.surfaces if item.zone_id == zone.zone_id
        ]
        total_area = sum(item.area_m2 for item in surfaces)
        wall_temperatures.append(
            sum(item.area_m2 * item.temperature_K for item in surfaces) / total_area
            if total_area > 0.0
            else zone.gas_temperature_K
        )
    return HeavyEnergyClosure(
        cv_over_kb=cv_over_kb,
        wall_temperature_K=np.asarray(wall_temperatures),
        wall_relaxation_s_inv=np.asarray(
            [
                config.wall_energy_relaxation_s_inv_by_zone.get(zone.zone_id, 0.0)
                for zone in case.reactor.zones
            ]
        ),
    )


def _compile_surface_model(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
) -> CompiledSurfaceModel | None:
    if case.models.surface_kinetics is None:
        return None
    surfaces = tuple(
        SurfaceGeometry(
            surface_id=surface.surface_id,
            zone_id=surface.zone_id,
            area_m2=surface.area_m2,
            site_density_m2=surface.site_density_m2,
            temperature_K=surface.temperature_K,
            initial_coverages=surface.initial_coverages,
        )
        for surface in case.reactor.surfaces
    )
    if not surfaces:
        raise CaseValidationError(
            "models.surface_kinetics requires at least one reactor surface"
        )
    return CompiledSurfaceModel(
        chemistry=chemistry_data,
        gas_species_ids=chemistry.species_ids,
        gas_masses_kg=chemistry.masses_kg,
        zone_ids=tuple(zone.zone_id for zone in case.reactor.zones),
        zone_volumes_m3=np.asarray([zone.volume_m3 for zone in case.reactor.zones]),
        surfaces=surfaces,
        domain_atol=case.solver.atol,
    )


def _ccp_geometry(
    case: CaseSpec, port: PowerPortConfig, chemistry: CompiledChemistry
) -> dict[str, float]:
    zone = next(item for item in case.reactor.zones if item.zone_id == port.zone_id)
    surfaces = [item for item in case.reactor.surfaces if item.zone_id == port.zone_id]
    if not surfaces:
        raise CaseValidationError(
            f"experimental CCP port {port.port_id!r} needs reactor surface areas"
        )
    total_area = sum(item.area_m2 for item in surfaces)
    target = next(
        (item for item in surfaces if item.surface_id == port.coupling_target), None
    )
    powered_area = target.area_m2 if target is not None else total_area / 2.0
    grounded_area = max(total_area - powered_area, powered_area)
    positive_masses = chemistry.masses_kg[chemistry.charges > 0.0]
    if positive_masses.size == 0:
        raise CaseValidationError(
            f"experimental CCP port {port.port_id!r} requires a positive ion species"
        )
    return {
        "zone_volume_m3": zone.volume_m3,
        "powered_area_m2": powered_area,
        "grounded_area_m2": grounded_area,
        "electrode_gap_m": zone.volume_m3 / total_area,
        "dominant_ion_mass_kg": float(np.min(positive_masses)),
        "gas_temperature_K": zone.gas_temperature_K,
    }


def _compile_power_ports(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
) -> PowerCoordinator | None:
    compiled: list[PowerPort] = []
    for port in case.reactor.power_ports:
        model = port.model
        if isinstance(model, PrescribedPowerModel):
            physical_port: PowerPort = PrescribedPowerPort(
                port.port_id,
                port.zone_id,
                electron_fraction=model.electron_fraction,
                gas_fraction=model.gas_fraction,
                default_power_W=model.default_absorbed_power_W,
            )
        elif isinstance(model, DCSeriesModel):
            physical_port = DCSeriesPort(
                port.port_id,
                port.zone_id,
                ballast_resistance_ohm=model.ballast_resistance_ohm,
                gap_m=model.gap_m,
                electrode_area_m2=model.electrode_area_m2,
                absorption_fraction=model.power_absorption_fraction,
                configured_mobility_m2_V_s=model.electron_mobility_m2_V_s,
                default_voltage_V=model.source_voltage_V,
            )
        elif isinstance(model, ExternalTableModel):
            default = compile_external_binding(model, external_tables)
            physical_port = ExternalTablePowerPort(port.port_id, port.zone_id, default)
        elif isinstance(model, ExperimentalRFEnvelopeModel):
            from plasma_global.experimental.power import RFEnvelopePort

            physical_port = RFEnvelopePort(
                port_id=port.port_id,
                zone_id=port.zone_id,
                frequency_Hz=model.frequency_Hz,
                role=model.role,
                coupling_efficiency=model.coupling_efficiency,
                effective_impedance_ohm=model.effective_impedance_ohm,
                base_reduced_field_Td=model.base_reduced_field_Td,
                reduced_field_per_sqrt_W_Td=model.reduced_field_per_sqrt_W_Td,
                self_bias_fraction=model.self_bias_fraction,
                plasma_potential_offset_V=model.plasma_potential_offset_V,
                plasma_potential_per_sqrt_W=model.plasma_potential_per_sqrt_W,
                default_power_W=(
                    model.default_absorbed_power_W
                    if model.control == "absorbed_power"
                    else None
                ),
                default_voltage_V=(
                    model.default_voltage_rms_V if model.control == "voltage" else None
                ),
            )
        elif isinstance(model, ExperimentalCCPModel):
            from plasma_global.experimental.power import CCPPowerPort

            physical_port = CCPPowerPort(
                port_id=port.port_id,
                zone_id=port.zone_id,
                frequency_Hz=model.frequency_Hz,
                default_power_W=(
                    model.default_absorbed_power_W
                    if model.control == "absorbed_power"
                    else None
                ),
                default_voltage_V=(
                    model.default_voltage_rms_V if model.control == "voltage" else None
                ),
                **_ccp_geometry(case, port, chemistry),
            )
        elif isinstance(model, ExperimentalICPModel):
            from plasma_global.experimental.power import ICPPowerPort

            zone = next(
                item for item in case.reactor.zones if item.zone_id == port.zone_id
            )
            physical_port = ICPPowerPort(
                port_id=port.port_id,
                zone_id=port.zone_id,
                frequency_Hz=model.frequency_Hz,
                gas_temperature_K=zone.gas_temperature_K,
                default_power_W=model.default_delivered_power_W,
            )
        else:
            assert_never(model)
        compiled.append(physical_port)
    if not compiled:
        return None
    return PowerCoordinator(
        ports=tuple(compiled),
        zone_ids=tuple(zone.zone_id for zone in case.reactor.zones),
    )


def _auto_bohm_factor(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    initial_densities: Mapping[str, Mapping[str, float]],
    surface: SurfaceConfig,
) -> float:
    config = surface.wall_transport
    if not isinstance(config, BohmWallTransport):
        raise TypeError("_auto_bohm_factor requires Bohm wall transport")
    if config.h_factor != "auto":
        return 0.61 * float(config.h_factor)
    zone = next(item for item in case.reactor.zones if item.zone_id == surface.zone_id)
    neutral_species = {
        species_id
        for species_id, charge in zip(chemistry.species_ids, chemistry.charges)
        if charge == 0.0
    }
    neutral_density = sum(
        density
        for species_id, density in initial_densities[zone.zone_id].items()
        if species_id in neutral_species
    )
    length = config.characteristic_length_m or zone.volume_m3 / surface.area_m2
    mean_free_path = 1.0 / max(
        neutral_density * config.ion_neutral_cross_section_m2, 1.0e-300
    )
    factor = 0.86 / math.sqrt(3.0 + length / (2.0 * mean_free_path))
    return 0.61 * min(max(factor, config.min_h_factor), config.max_h_factor)


def _compile_wall_boundary(
    chemistry: CompiledChemistry,
    *,
    zone_id: str,
    surface_id: str,
    area_m2: float,
    transport_kind: Literal["bohm", "prescribed_frequency", "ambipolar", "off"],
    bohm_factor: float = 0.61,
    sheath_energy_eV: float = 0.0,
    prescribed_frequency_s_inv: float | None = None,
    diffusion_coefficient_m2_s: float | None = None,
    diffusion_length_m: float | None = None,
) -> WallBoundary:
    """Bind typed compiled boundary reactions to one reactor surface."""

    reactions = tuple(
        BoundaryReaction(
            reaction_id=reaction.id,
            incident_species=reaction.incident_species,
            products=reaction.products,
            probability=reaction.probability,
        )
        for reaction in chemistry.boundary_reactions
        if (not reaction.zones or zone_id in reaction.zones)
        and (not reaction.surfaces or surface_id in reaction.surfaces)
    )
    return WallBoundary(
        zone_id=zone_id,
        surface_id=surface_id,
        area_m2=area_m2,
        reactions=reactions,
        transport_kind=transport_kind,
        bohm_factor=bohm_factor,
        sheath_energy_eV=sheath_energy_eV,
        prescribed_frequency_s_inv=prescribed_frequency_s_inv,
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        diffusion_length_m=diffusion_length_m,
    )


def _compile_walls(
    case: CaseSpec,
    chemistry: CompiledChemistry,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> tuple[WallBoundary, ...]:
    walls: list[WallBoundary] = []
    for surface in case.reactor.surfaces:
        transport = surface.wall_transport
        if isinstance(transport, OffWallTransport):
            continue
        if isinstance(transport, BohmWallTransport):
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface.surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="bohm",
                bohm_factor=_auto_bohm_factor(
                    case, chemistry, initial_densities, surface
                ),
            )
        elif isinstance(transport, PrescribedFrequencyWallTransport):
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface.surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="prescribed_frequency",
                prescribed_frequency_s_inv=transport.frequency_s_inv,
            )
        elif isinstance(transport, AmbipolarWallTransport):
            zone = next(
                item for item in case.reactor.zones if item.zone_id == surface.zone_id
            )
            boundary = _compile_wall_boundary(
                chemistry,
                zone_id=surface.zone_id,
                area_m2=surface.area_m2,
                surface_id=surface.surface_id,
                sheath_energy_eV=surface.ion_impact_energy_eV,
                transport_kind="ambipolar",
                diffusion_coefficient_m2_s=transport.diffusion_coefficient_m2_s,
                diffusion_length_m=(
                    transport.diffusion_length_m or zone.volume_m3 / surface.area_m2
                ),
            )
        else:
            assert_never(transport)
        walls.append(boundary)
    return tuple(walls)


def compile_reactor(
    case: CaseSpec,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    external_tables: ExternalTableStore,
    initial_densities: Mapping[str, Mapping[str, float]],
) -> CompiledReactor:
    """Compile topology, transport, walls, and typed power ports."""

    validate_surface_initial_conditions(case, chemistry_data)
    return CompiledReactor(
        zones=_compile_zones(case),
        transport=_compile_static_transport(case, chemistry),
        wall_boundaries=_compile_walls(case, chemistry, initial_densities),
        power_coordinator=_compile_power_ports(case, chemistry, external_tables),
        heavy_energy_closure=_compile_heavy_energy(case, chemistry_data, chemistry),
        surface_model=_compile_surface_model(case, chemistry_data, chemistry),
    )


__all__ = [
    "CompiledReactor",
    "compile_external_binding",
    "compile_initial_densities",
    "compile_reactor",
    "validate_surface_initial_conditions",
]
