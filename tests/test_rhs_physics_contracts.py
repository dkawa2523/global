from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_global.chemistry.models import E_CHARGE
from plasma_global.physics.electron_energy_relaxation import (
    apply_field_table_energy_relaxation,
    field_table_energy_relaxation_zones,
)
from plasma_global.physics.gas_phase_core import electron_absorbed_power_fraction
from plasma_global.physics.gas_reactions import reaction_mass_action
from plasma_global.physics.surface_rates import evaluate_surface_rate
from plasma_global.physics.types import CompiledGasReaction, CompiledSurfaceReaction, SurfaceRateContext


def _rate_model_system(floor_density: float = 1.0e9):
    return SimpleNamespace(floor_density=floor_density)


def test_gas_mass_action_does_not_use_density_floor_as_reactant() -> None:
    rxn = CompiledGasReaction(
        reaction_id='R_ZERO_REACTANT',
        zones=['plasma'],
        reactant_gas=[(0, 1.0, 'Ar_star')],
        electron_reactant_stoich=0.0,
        delta_gas=[],
        rate_model={'backend': 'constant', 'value': 1.0},
        energy_model=None,
    )

    value = reaction_mass_action(_rate_model_system(), rxn, np.array([0.0]), ne=1.0e15)

    assert value == 0.0


def test_electron_mass_action_does_not_use_density_floor_as_reactant() -> None:
    rxn = CompiledGasReaction(
        reaction_id='R_ZERO_ELECTRON',
        zones=['plasma'],
        reactant_gas=[],
        electron_reactant_stoich=1.0,
        delta_gas=[],
        rate_model={'backend': 'constant', 'value': 1.0},
        energy_model=None,
    )

    system = SimpleNamespace(
        electron_density_closure='quasi_neutral',
        gas_charges=np.array([1.0]),
        floor_density=1.0e9,
    )

    value = reaction_mass_action(system, rxn, np.array([0.0]), ne=system.floor_density)

    assert value == 0.0


def test_prescribed_electron_profile_can_drive_electron_mass_action() -> None:
    rxn = CompiledGasReaction(
        reaction_id='R_PRESCRIBED_ELECTRON',
        zones=['plasma'],
        reactant_gas=[],
        electron_reactant_stoich=1.0,
        delta_gas=[],
        rate_model={'backend': 'constant', 'value': 1.0},
        energy_model=None,
    )
    system = SimpleNamespace(
        electron_density_closure='prescribed_profile',
        gas_charges=np.array([0.0]),
        floor_density=1.0e9,
    )

    value = reaction_mass_action(system, rxn, np.array([0.0]), ne=2.0e15)

    assert value == pytest.approx(2.0e15)


def test_surface_neutral_sticking_does_not_use_density_floor_as_reactant() -> None:
    system = SimpleNamespace(
        floor_density=1.0e9,
        gas_species=[SimpleNamespace(charge=0)],
        gas_masses=np.array([6.63e-26]),
    )
    rxn = CompiledSurfaceReaction(
        reaction_id='S_NEUTRAL_STICKING',
        zone_id='plasma',
        surface_id='wall',
        gas_reactants=[(0, 1.0, 'Ar')],
        surface_reactants=[],
        delta_gas=[],
        delta_surface=[],
        area_over_volume=1.0,
        area_m2=1.0,
        site_density_m2=1.0,
        rate_model={'backend': 'sticking', 'sticking_value': 1.0},
        inventory_idx=None,
        film_factor=0.0,
    )
    context = SurfaceRateContext(
        reaction=rxn,
        gas_row=np.array([0.0]),
        gas_temperature_K=300.0,
        state=np.array([]),
        surface_temperature_K=300.0,
        ion_energy_eV=0.0,
        positive_ion_density_m3=0.0,
        ion_flux_m2_s=0.0,
    )

    assert evaluate_surface_rate(SimpleNamespace(system=system), context) == 0.0


def test_surface_coverage_factor_zero_coverage_gives_zero_rate() -> None:
    system = SimpleNamespace(
        floor_density=1.0,
        gas_species=[SimpleNamespace(charge=1)],
        state_layout=SimpleNamespace(surface_index={'wall': {'wall:F': 0}}),
    )
    rxn = CompiledSurfaceReaction(
        reaction_id='S_ION_YIELD_BLOCKED',
        zone_id='plasma',
        surface_id='wall',
        gas_reactants=[(0, 1.0, 'Ar_plus')],
        surface_reactants=[],
        delta_gas=[],
        delta_surface=[],
        area_over_volume=1.0,
        area_m2=1.0,
        site_density_m2=1.0,
        rate_model={
            'backend': 'ion_assisted',
            'yield_value': 1.0,
            'threshold_eV': 0.0,
            'coverage_factor': {'kind': 'species_power', 'species': 'wall:F', 'exponent': 1.0},
        },
        inventory_idx=None,
        film_factor=0.0,
    )
    context = SurfaceRateContext(
        reaction=rxn,
        gas_row=np.array([1.0e15]),
        gas_temperature_K=300.0,
        state=np.array([0.0]),
        surface_temperature_K=300.0,
        ion_energy_eV=100.0,
        positive_ion_density_m3=1.0e15,
        ion_flux_m2_s=2.0,
    )

    assert evaluate_surface_rate(SimpleNamespace(system=system), context) == 0.0


def test_field_table_energy_relaxation_accumulates_without_overwriting_rhs() -> None:
    system = SimpleNamespace(
        run_config=SimpleNamespace(
            swarm=SimpleNamespace(
                table=SimpleNamespace(
                    electron_energy_mode='table_relaxation',
                    energy_relaxation_time_s=2.0,
                )
            )
        ),
        zone_ids=['plasma'],
        zone_index={'plasma': 0},
        floor_density=1.0,
    )
    current_energy = 3.0 * E_CHARGE
    coupled = SimpleNamespace(
        eedf_by_zone={
            'plasma': SimpleNamespace(
                transport=SimpleNamespace(lookup_mode='field', mean_energy_eV=5.0)
            )
        },
        ne_by_zone={'plasma': 2.0},
        electron_energy=np.array([current_energy]),
    )
    rhs = np.array([7.0])

    apply_field_table_energy_relaxation(system, coupled, rhs)

    target_energy = 2.0 * 5.0 * E_CHARGE
    assert field_table_energy_relaxation_zones(system, coupled) == {'plasma'}
    assert rhs[0] == pytest.approx(7.0 + (target_energy - current_energy) / 2.0)


def test_absorbed_power_is_split_only_when_gas_temperature_is_evolved() -> None:
    with_gas_temperature = SimpleNamespace(
        state_layout=SimpleNamespace(slices={'gas_temperature': object()}),
        run_config=SimpleNamespace(physics=SimpleNamespace(gas_heating_fraction=0.25)),
    )
    without_gas_temperature = SimpleNamespace(
        state_layout=SimpleNamespace(slices={}),
        run_config=SimpleNamespace(physics=SimpleNamespace(gas_heating_fraction=0.25)),
    )

    assert electron_absorbed_power_fraction(with_gas_temperature) == pytest.approx(0.75)
    assert electron_absorbed_power_fraction(without_gas_temperature) == pytest.approx(1.0)
