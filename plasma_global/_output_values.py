"""Physical values obtained from the same evaluation used by the solver."""

from __future__ import annotations

import numpy as np

from plasma_global.core.compiled import CompiledGlobalModel, ModelEvaluation
from plasma_global.core.domain import RecipeSegment
from plasma_global.models.walls import WallBoundary

BOLTZMANN_J_K = 1.380649e-23

LEDGER_QUANTITIES = {
    "electron_power_W_m3": "absorbed_power_J_m3_s",
    "gas_power_W_m3": "gas_power_J_m3_s",
    "reaction_energy_loss_W_m3": "reaction_energy_loss_J_m3_s",
    "wall_energy_loss_W_m3": "wall_energy_loss_J_m3_s",
    "elastic_heating_W_m3": "elastic_heating_J_m3_s",
    "gas_reaction_heating_W_m3": "gas_reaction_heating_J_m3_s",
    "surface_reaction_heating_W_m3": "surface_reaction_heating_J_m3_s",
    "transport_electron_power_W_m3": "transport_electron_energy_J_m3_s",
    "transport_heavy_power_W_m3": "transport_heavy_energy_J_m3_s",
    "inlet_heavy_power_W_m3": "inlet_heavy_energy_J_m3_s",
    "wall_heavy_power_W_m3": "wall_heavy_energy_exchange_J_m3_s",
}

PORT_UNITS = {
    "electron_power_W": "W",
    "gas_power_W": "W",
    "absorbed_power_W": "W",
    "source_voltage_V": "V",
    "plasma_voltage_V": "V",
    "voltage_rms_V": "V",
    "current_A": "A",
    "current_rms_A": "A",
    "dc_self_bias_V": "V",
    "delivered_power_W": "W",
    "source_power_W": "W",
    "commanded_power_W": "W",
    "apparent_power_VA": "VA",
    "reduced_field_Td": "Td",
    "ion_energy_eV": "eV",
    "plasma_potential_V": "V",
    "self_bias_V": "V",
    "frequency_Hz": "Hz",
    "rf_voltage_rms_V": "V",
    "rf_current_rms_A": "A",
    "coil_voltage_rms_V": "V",
    "coil_current_rms_A": "A",
    "downstream_power_W": "W",
    "reflected_power_W": "W",
    "coupling_efficiency": "1",
    "bulk_resistance_ohm": "ohm",
    "sheath_reactance_ohm": "ohm",
    "powered_sheath_thickness_m": "m",
    "grounded_sheath_thickness_m": "m",
    "powered_ion_energy_eV": "eV",
    "grounded_ion_energy_eV": "eV",
}


def zone_values(
    model: CompiledGlobalModel,
    state: np.ndarray,
    evaluated: ModelEvaluation,
    *,
    collect_ledger: bool,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for zone in model.zones:
        name = zone.zone_id
        electrons = evaluated.electron_states[name]
        temperature = evaluated.gas_temperature_K_by_zone[name]
        densities = state[model.layout.density_slices[name]]
        total = float(np.sum(densities))
        scalars = {
            "electron_density_m3": electrons.density_m3,
            "mean_energy_eV": electrons.mean_energy_eV,
            "electron_temperature_eV": electrons.temperature_eV,
            "gas_temperature_K": temperature,
            "charge_residual_m3": evaluated.charge_residual_m3_by_zone[name],
            "heavy_density_m3": total,
            "neutral_density_m3": float(np.sum(densities[model.charges == 0])),
            "pressure_Pa": total * BOLTZMANN_J_K * temperature,
        }
        if electrons.reduced_field_Td is not None:
            scalars["reduced_field_Td"] = electrons.reduced_field_Td
        values.update({f"{key}[{name}]": value for key, value in scalars.items()})
        values.update(
            _species_values(model, name, densities, total, temperature, evaluated)
        )
        if collect_ledger:
            values.update(_ledger_values(model, name, evaluated))
    return values


def _species_values(
    model: CompiledGlobalModel,
    zone: str,
    densities: np.ndarray,
    total: float,
    temperature: float,
    evaluated: ModelEvaluation,
) -> dict[str, float]:
    values: dict[str, float] = {}
    derivative = evaluated.derivative[model.layout.density_slices[zone]]
    for index, species in enumerate(model.species_ids):
        values[f"partial_pressure_Pa[{zone},{species}]"] = (
            float(densities[index]) * BOLTZMANN_J_K * temperature
        )
        values[f"net_production_m3_s[{zone},{species}]"] = float(derivative[index])
        if total > 0:
            values[f"mole_fraction[{zone},{species}]"] = float(densities[index]) / total
    return values


def _ledger_values(
    model: CompiledGlobalModel,
    zone: str,
    evaluated: ModelEvaluation,
) -> dict[str, float]:
    ledger = evaluated.ledger_by_zone[zone]
    volume = next(item.volume_m3 for item in model.zones if item.zone_id == zone)
    values = {
        f"{name}[{zone}]": float(getattr(ledger, field))
        for name, field in LEDGER_QUANTITIES.items()
    }
    values[f"absorbed_power_W[{zone}]"] = (
        ledger.absorbed_power_J_m3_s + ledger.gas_power_J_m3_s
    ) * volume
    return values


def surface_values(
    model: CompiledGlobalModel,
    state: np.ndarray,
    evaluated: ModelEvaluation,
    segment: RecipeSegment,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for wall in model.wall_boundaries:
        if wall.surface_id is not None:
            values.update(_wall_values(model, wall, evaluated))
    if model.surface_model is not None:
        values.update(_surface_coverages(model, state, segment))
    return values


def _wall_values(
    model: CompiledGlobalModel, wall: WallBoundary, evaluated: ModelEvaluation
) -> dict[str, float]:
    records = [
        record
        for record in evaluated.ledger_by_zone[wall.zone_id].wall_fluxes
        if record.surface_id == wall.surface_id
    ]
    values = {}
    if records:
        values[f"wall_sheath_energy_eV[{wall.surface_id}]"] = records[
            0
        ].sheath_energy_eV
    elif wall.transport_kind == "off":
        values[f"wall_sheath_energy_eV[{wall.surface_id}]"] = 0.0
    fluxes = {record.incident_species: record.incident_flux_m2_s for record in records}
    for index in np.flatnonzero(model.charges > 0):
        species = model.species_ids[int(index)]
        values[f"incident_flux_m2_s[{wall.surface_id},{species}]"] = fluxes.get(
            species, 0.0
        )
    return values


def _surface_coverages(
    model: CompiledGlobalModel,
    state: np.ndarray,
    segment: RecipeSegment,
) -> dict[str, float]:
    surface_model = model.surface_model
    if surface_model is None:
        return {}
    coverage = state[model.layout.surface_coverage_slice]
    values: dict[str, float] = {}
    for surface in surface_model.surfaces:
        identity = surface.surface_id
        values[f"surface_temperature_K[{identity}]"] = (
            segment.surface_temperature_K_by_surface.get(
                identity, surface.temperature_K
            )
        )
        species_ids = [
            species
            for region, species in surface_model.layout.state_index
            if region == identity
        ]
        species_ids.append(surface_model.layout.free_species_by_surface[identity])
        for species in species_ids:
            values[f"coverage[{identity},{species}]"] = surface_model.coverage(
                coverage, identity, species
            )
    return values


def port_values(evaluated: ModelEvaluation) -> dict[str, float]:
    values: dict[str, float] = {}
    if evaluated.power_coupling is None:
        return values
    for identity, port in evaluated.power_coupling.port_results.items():
        scalars = dict(port.observables)
        scalars.update(
            electron_power_W=port.electron_power_W, gas_power_W=port.gas_power_W
        )
        if port.reduced_field_Td is not None:
            scalars["reduced_field_Td"] = port.reduced_field_Td
        values.update(
            {
                f"port_{name}[{identity}]": value
                for name, value in scalars.items()
                if name in PORT_UNITS
            }
        )
    return values
