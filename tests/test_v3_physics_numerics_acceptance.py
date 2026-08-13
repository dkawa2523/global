from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, SolverSettings, Zone
from plasma_global.core.solver import solve_compiled_model
from plasma_global.core.transport import CompiledTransport, SegmentTransport
from plasma_global.models.electrons import (
    ELEMENTARY_CHARGE_C,
    ElectronEnergyClosure,
    LocalFieldClosure,
    TabulatedMeanEnergy,
)
from plasma_global.models.gas_energy import BOLTZMANN_J_K, HeavyEnergyClosure
from plasma_global.models.walls import BoundaryReaction, WallBoundary


@dataclass(frozen=True)
class ChemistryFixture:
    species_ids: tuple[str, ...]
    charges: np.ndarray
    masses_kg: np.ndarray
    reaction_ids: tuple[str, ...]
    stoichiometry: np.ndarray
    reactant_orders: np.ndarray
    electron_orders: np.ndarray
    rate_evaluators: tuple[object, ...]
    energy_loss_eV: np.ndarray
    gas_heating_eV: np.ndarray
    reaction_zones: tuple[tuple[str, ...], ...]
    element_names: tuple[str, ...] = ()
    element_matrix: np.ndarray | None = None

    @property
    def jacobian_species_pattern(self) -> np.ndarray:
        return (
            (self.stoichiometry != 0.0).T.astype(np.int8)
            @ (self.reactant_orders != 0.0).astype(np.int8)
        ) > 0


def _normalized_drift(series: np.ndarray) -> float:
    initial = np.asarray(series[0], dtype=float)
    scale = np.maximum(np.abs(initial), 1.0)
    return float(np.max(np.abs(np.asarray(series) - initial) / scale))


def _inert_pair() -> ChemistryFixture:
    return ChemistryFixture(
        species_ids=("A", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([6.6e-26, 6.6e-26]),
        reaction_ids=(),
        stoichiometry=np.zeros((0, 2)),
        reactant_orders=np.zeros((0, 2)),
        electron_orders=np.zeros(0),
        rate_evaluators=(),
        energy_loss_eV=np.zeros(0),
        gas_heating_eV=np.zeros(0),
        reaction_zones=(),
    )


def test_closed_system_element_and_charge_drift_are_below_1e_8() -> None:
    elements = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    chemistry = ChemistryFixture(
        species_ids=("X", "X_plus", "Y", "Y_plus"),
        charges=np.array([0.0, 1.0, 0.0, 1.0]),
        masses_kg=np.full(4, 5.0e-26),
        reaction_ids=("charge_transfer",),
        stoichiometry=np.array([[1.0, -1.0, -1.0, 1.0]]),
        reactant_orders=np.array([[0.0, 1.0, 1.0, 0.0]]),
        electron_orders=np.array([0.0]),
        rate_evaluators=(1.0e-18,),
        energy_loss_eV=np.array([0.0]),
        gas_heating_eV=np.zeros(1),
        reaction_zones=((),),
        element_names=("X", "Y"),
        element_matrix=elements,
    )
    model = CompiledGlobalModel(
        chemistry=chemistry,
        zones=(Zone("closed", 1.0),),
        segments=(RecipeSegment("closed", 0.0, 2.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={
                "closed": {
                    "X": 2.0e17,
                    "X_plus": 4.0e17,
                    "Y": 6.0e17,
                    "Y_plus": 1.0e17,
                }
            },
            mean_energy_eV_by_zone={"closed": 3.0},
        ),
        SolverSettings(rtol=1.0e-9, atol=1.0e-5, sample_interval_s=0.05),
    )
    density = result.state[:, model.layout.density_slices["closed"]]
    element_inventory = density @ elements.T
    heavy_charge = density @ chemistry.charges

    assert _normalized_drift(element_inventory) <= 1.0e-8
    assert _normalized_drift(heavy_charge[:, None]) <= 1.0e-8


def test_cstr_matches_constant_inlet_and_pump_analytic_solution_to_1e_6() -> None:
    pump_frequency = 0.5
    particle_source = 3.0
    transport = CompiledTransport(
        volumes_m3=np.array([1.0]),
        pump_frequency_s_inv=np.array([pump_frequency]),
        edge_from=np.array([], dtype=int),
        edge_to=np.array([], dtype=int),
        edge_conductance_m3_s=np.array([]),
        n_species=2,
    )
    segment = RecipeSegment(
        "cstr",
        0.0,
        2.0,
        reduced_field_Td_by_zone={"cstr": 1.0},
        transport=SegmentTransport(
            particle_source_m3_s=np.array([[particle_source, 0.0]]),
            inlet_heavy_energy_J_m3_s=np.zeros(1),
        ),
    )
    model = CompiledGlobalModel(
        chemistry=_inert_pair(),
        zones=(Zone("cstr", 1.0),),
        segments=(segment,),
        electron_closure=LocalFieldClosure(TabulatedMeanEnergy((1.0, 2.0), (3.0, 3.0))),
        transport=transport,
    )
    initial_density = 2.0
    result = solve_compiled_model(
        model,
        InitialState(densities_m3_by_zone={"cstr": {"A": initial_density, "ion": 1.0}}),
        SolverSettings(rtol=1.0e-10, atol=1.0e-12, save_at_s=(0.0, 2.0)),
    )
    steady_density = particle_source / pump_frequency
    expected = steady_density + (initial_density - steady_density) * np.exp(
        -pump_frequency * segment.end_s
    )
    error = abs(result.final_value("n[cstr,A]") - expected) / expected

    assert error <= 1.0e-6


def test_adiabatic_monatomic_filling_uses_inlet_enthalpy() -> None:
    cv_over_kb = 1.5
    inlet_temperature_K = 600.0
    initial_temperature_K = 300.0
    initial_neutral_density = 2.0e20
    ion_density = 1.0e15
    particle_source = 1.0e20
    duration_s = 0.5
    closure = HeavyEnergyClosure(
        cv_over_kb=np.array([cv_over_kb, cv_over_kb]),
        wall_temperature_K=np.array([initial_temperature_K]),
        wall_relaxation_s_inv=np.zeros(1),
    )
    transport = CompiledTransport(
        volumes_m3=np.array([1.0]),
        pump_frequency_s_inv=np.zeros(1),
        edge_from=np.array([], dtype=int),
        edge_to=np.array([], dtype=int),
        edge_conductance_m3_s=np.array([]),
        n_species=2,
        heavy_cv_over_kb=closure.cv_over_kb,
    )
    segment = RecipeSegment(
        "fill",
        0.0,
        duration_s,
        reduced_field_Td_by_zone={"vessel": 1.0},
        transport=SegmentTransport(
            particle_source_m3_s=np.array([[particle_source, 0.0]]),
            inlet_heavy_energy_J_m3_s=np.array(
                [
                    particle_source
                    * (cv_over_kb + 1.0)
                    * BOLTZMANN_J_K
                    * inlet_temperature_K
                ]
            ),
        ),
    )
    model = CompiledGlobalModel(
        chemistry=_inert_pair(),
        zones=(Zone("vessel", 1.0),),
        segments=(segment,),
        electron_closure=LocalFieldClosure(TabulatedMeanEnergy((1.0, 2.0), (3.0, 3.0))),
        transport=transport,
        heavy_energy_closure=closure,
    )
    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={
                "vessel": {"A": initial_neutral_density, "ion": ion_density}
            },
            gas_temperature_K_by_zone={"vessel": initial_temperature_K},
        ),
        SolverSettings(rtol=1.0e-10, atol=1.0e-12, save_at_s=(0.0, duration_s)),
    )
    final_density = result.state[-1, model.layout.density_slices["vessel"]]
    final_energy = result.state[-1, model.layout.heavy_energy_indices["vessel"]]
    actual_temperature = closure.temperature_K(
        final_density[None, :], np.array([final_energy])
    )[0]
    initial_total_density = initial_neutral_density + ion_density
    expected_temperature = (
        cv_over_kb * initial_total_density * initial_temperature_K
        + (cv_over_kb + 1.0) * particle_source * duration_s * inlet_temperature_K
    ) / (cv_over_kb * (initial_total_density + particle_source * duration_s))

    assert actual_temperature == pytest.approx(expected_temperature, rel=1.0e-8)


def test_bohm_wall_loss_matches_exponential_decay_to_1e_5() -> None:
    mass_kg = 6.6e-26
    area_m2 = 1.0e-3
    bohm_factor = 0.61
    mean_energy_eV = 3.0
    segment = RecipeSegment(
        "bohm",
        0.0,
        1.0,
        reduced_field_Td_by_zone={"plasma": 1.0},
    )
    wall = WallBoundary(
        zone_id="plasma",
        area_m2=area_m2,
        bohm_factor=bohm_factor,
        reactions=(BoundaryReaction("neutralize", "ion", products={"A": 1.0}),),
    )
    model = CompiledGlobalModel(
        chemistry=_inert_pair(),
        zones=(Zone("plasma", 1.0),),
        segments=(segment,),
        electron_closure=LocalFieldClosure(
            TabulatedMeanEnergy((1.0, 2.0), (mean_energy_eV, mean_energy_eV))
        ),
        wall_boundaries=(wall,),
    )
    initial_ion_density = 1.0e15
    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={"plasma": {"A": 1.0e20, "ion": initial_ion_density}}
        ),
        SolverSettings(rtol=1.0e-9, atol=1.0e-14, save_at_s=(0.0, 1.0)),
    )
    electron_temperature_eV = (2.0 / 3.0) * mean_energy_eV
    loss_frequency = (
        bohm_factor
        * np.sqrt(ELEMENTARY_CHARGE_C * electron_temperature_eV / mass_kg)
        * area_m2
    )
    expected = initial_ion_density * np.exp(-loss_frequency * segment.end_s)
    error = abs(result.final_value("n[plasma,ion]") - expected) / expected

    assert error <= 1.0e-5


def test_halving_solver_tolerances_changes_main_quantities_and_peak_time_within_limits() -> (
    None
):
    k1, k2 = 2.0, 0.5
    chemistry = ChemistryFixture(
        species_ids=("A", "B", "C", "ion"),
        charges=np.array([0.0, 0.0, 0.0, 1.0]),
        masses_kg=np.full(4, 5.0e-26),
        reaction_ids=("A_to_B", "B_to_C"),
        stoichiometry=np.array([[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0]]),
        reactant_orders=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        electron_orders=np.zeros(2),
        rate_evaluators=(k1, k2),
        energy_loss_eV=np.zeros(2),
        gas_heating_eV=np.zeros(2),
        reaction_zones=((), ()),
    )
    model = CompiledGlobalModel(
        chemistry=chemistry,
        zones=(Zone("batch", 1.0),),
        segments=(RecipeSegment("batch", 0.0, 2.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    initial = InitialState(
        densities_m3_by_zone={
            "batch": {"A": 1.0e12, "B": 0.0, "C": 0.0, "ion": 1.0e10}
        },
        mean_energy_eV_by_zone={"batch": 3.0},
    )

    def solve(rtol: float, atol: float):
        return solve_compiled_model(
            model,
            initial,
            SolverSettings(
                rtol=rtol,
                atol=atol,
                sample_interval_s=0.005,
            ),
        )

    coarse = solve(1.0e-5, 1.0e-7)
    fine = solve(5.0e-6, 5.0e-8)
    coarse_b = coarse.series("n[batch,B]")
    fine_b = fine.series("n[batch,B]")
    quantities_coarse = np.array(
        [
            coarse.final_value("n[batch,A]"),
            coarse_b.max(),
            coarse.final_value("n[batch,C]"),
        ]
    )
    quantities_fine = np.array(
        [fine.final_value("n[batch,A]"), fine_b.max(), fine.final_value("n[batch,C]")]
    )
    relative_change = np.abs(quantities_coarse - quantities_fine) / np.maximum(
        np.abs(quantities_fine), 1.0
    )
    coarse_peak_time = float(coarse.time_s[int(np.argmax(coarse_b))])
    fine_peak_time = float(fine.time_s[int(np.argmax(fine_b))])
    analytic_peak_time = np.log(k1 / k2) / (k1 - k2)

    assert np.max(relative_change) <= 0.01
    assert abs(coarse_peak_time - fine_peak_time) / fine_peak_time <= 0.02
    assert abs(fine_peak_time - analytic_peak_time) / analytic_peak_time <= 0.02
