from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import plasma_global.core.compiled as compiled_module
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, SolverSettings, Zone
from plasma_global.core.exceptions import QuasineutralityError, StateDomainError
from plasma_global.core.solver import solve_compiled_model
from plasma_global.models.electrons import (
    ELEMENTARY_CHARGE_C,
    ElectronEnergyClosure,
    LocalFieldClosure,
    TabulatedMeanEnergy,
)
from plasma_global.models.walls import (
    ELECTRON_MASS_KG,
    BoundaryReaction,
    WallBoundary,
)


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

    @property
    def jacobian_species_pattern(self) -> np.ndarray:
        return (
            (self.stoichiometry != 0.0).T.astype(np.int8)
            @ (self.reactant_orders != 0.0).astype(np.int8)
        ) > 0


ARGON_MASS_KG = 6.6335209e-26


def inert_argon() -> ChemistryFixture:
    return ChemistryFixture(
        species_ids=("Ar", "Ar_plus"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([ARGON_MASS_KG, ARGON_MASS_KG]),
        reaction_ids=(),
        stoichiometry=np.zeros((0, 2)),
        reactant_orders=np.zeros((0, 2)),
        electron_orders=np.zeros(0),
        rate_evaluators=(),
        energy_loss_eV=np.zeros(0),
        gas_heating_eV=np.zeros(0),
        reaction_zones=(),
    )


def test_mean_energy_and_electron_temperature_are_distinct() -> None:
    density = 2.0e15
    state = ElectronEnergyClosure().evaluate(
        net_heavy_charge_density_m3=density,
        energy_density_J_m3=density * ELEMENTARY_CHARGE_C * 3.0,
        reduced_field_Td=None,
    )

    assert state.mean_energy_eV == pytest.approx(3.0)
    assert state.temperature_eV == pytest.approx(2.0)


def test_quasineutral_closure_rejects_negative_electron_density_without_floor() -> None:
    with pytest.raises(QuasineutralityError, match="negative electron density"):
        ElectronEnergyClosure().evaluate(
            net_heavy_charge_density_m3=-1.0,
            energy_density_J_m3=0.0,
            reduced_field_Td=None,
        )


def test_local_field_is_algebraic_and_rejects_out_of_table_domain() -> None:
    closure = LocalFieldClosure(
        TabulatedMeanEnergy(
            reduced_field_Td=(10.0, 30.0),
            mean_energy_eV=(1.5, 4.5),
        )
    )
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("plasma", 1.0),),
        segments=(
            RecipeSegment(
                "field",
                0.0,
                1.0,
                reduced_field_Td_by_zone={"plasma": 20.0},
            ),
        ),
        electron_closure=closure,
    )
    initial = InitialState(
        densities_m3_by_zone={"plasma": {"Ar": 1.0e20, "Ar_plus": 1.0e15}}
    )
    state = model.initial_state(initial)
    evaluated = model.evaluate(0.0, state, model.segments[0])

    assert model.layout.evolves_electron_energy is False
    assert not any("electron_energy" in label for label in model.layout.labels)
    assert evaluated.electron_states["plasma"].mean_energy_eV == pytest.approx(3.0)
    assert evaluated.electron_states["plasma"].temperature_eV == pytest.approx(2.0)
    with pytest.raises(StateDomainError, match="outside table bounds"):
        closure.evaluate(
            net_heavy_charge_density_m3=1.0e15,
            energy_density_J_m3=None,
            reduced_field_Td=31.0,
        )


def test_mass_action_bdf_matches_first_order_analytic_solution() -> None:
    rate_s_inv = 2.0
    chemistry = ChemistryFixture(
        species_ids=("A", "B", "ion"),
        charges=np.array([0.0, 0.0, 1.0]),
        masses_kg=np.array([2.0e-26, 2.0e-26, 6.0e-26]),
        reaction_ids=("A_to_B",),
        stoichiometry=np.array([[-1.0, 1.0, 0.0]]),
        reactant_orders=np.array([[1.0, 0.0, 0.0]]),
        electron_orders=np.array([0.0]),
        rate_evaluators=(rate_s_inv,),
        energy_loss_eV=np.array([0.0]),
        gas_heating_eV=np.zeros(1),
        reaction_zones=((),),
    )
    model = CompiledGlobalModel(
        chemistry=chemistry,
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("decay", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    initial_A = 1.0e18
    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={"z": {"A": initial_A, "B": 0.0, "ion": 1.0e14}},
            mean_energy_eV_by_zone={"z": 3.0},
        ),
        SolverSettings(rtol=1.0e-9, atol=1.0e-11, save_at_s=(0.0, 0.5, 1.0)),
    )

    expected_A = initial_A * np.exp(-rate_s_inv)
    assert result.status.success is True
    assert result.state.shape == (3, model.layout.size)
    assert result.final_value("n[z,A]") == pytest.approx(expected_A, rel=2.0e-7)
    assert result.final_value("n[z,A]") + result.final_value("n[z,B]") == pytest.approx(
        initial_A,
        rel=1.0e-10,
    )


def test_recipe_segments_bind_endpoint_forcing_and_integrate_power_balance() -> None:
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("plasma", 2.0),),
        segments=(
            RecipeSegment("low", 0.0, 1.0, absorbed_power_W_by_zone={"plasma": 2.0}),
            RecipeSegment("high", 1.0, 2.0, absorbed_power_W_by_zone={"plasma": 6.0}),
        ),
        electron_closure=ElectronEnergyClosure(),
    )
    initial = InitialState(
        densities_m3_by_zone={"plasma": {"Ar": 1.0e20, "Ar_plus": 1.0e15}},
        mean_energy_eV_by_zone={"plasma": 3.0},
    )
    y0 = model.initial_state(initial)
    energy_index = model.layout.electron_energy_indices["plasma"]

    # At the shared boundary, the old segment remains bound to the old RHS.
    old_endpoint_rhs = model.bind_segment(model.segments[0])(1.0, y0)
    new_start_rhs = model.bind_segment(model.segments[1])(1.0, y0)
    assert old_endpoint_rhs[energy_index] == pytest.approx(1.0)
    assert new_start_rhs[energy_index] == pytest.approx(3.0)

    result = solve_compiled_model(
        model,
        initial,
        SolverSettings(rtol=1.0e-10, atol=1.0e-12, save_at_s=(0.0, 1.0, 2.0)),
    )
    expected_final = y0[energy_index] + 1.0 + 3.0
    assert np.array_equal(result.time_s, np.array([0.0, 1.0, 2.0]))
    assert result.final_value("electron_energy[plasma]") == pytest.approx(
        expected_final, rel=1.0e-10
    )
    assert result.metadata["electron_closure"] == "electron_energy"


def test_bound_rhs_skips_full_diagnostic_result_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("plasma", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"plasma": {"Ar": 1.0e20, "Ar_plus": 1.0e15}},
            mean_energy_eV_by_zone={"plasma": 3.0},
        )
    )

    def reject_diagnostics(_self: object) -> None:
        raise AssertionError("solver RHS built ModelEvaluation")

    monkeypatch.setattr(
        compiled_module.ModelEvaluation, "__post_init__", reject_diagnostics
    )
    derivative = model.bind_segment(model.segments[0])(0.0, state)

    assert derivative.shape == state.shape


def test_rhs_never_clips_negative_density() -> None:
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"Ar": 1.0e20, "Ar_plus": 1.0e15}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    state[model.layout.density_slices["z"].start] = -1.0

    with pytest.raises(StateDomainError, match="Negative density"):
        model.evaluate(0.0, state, model.segments[0])


def test_domain_tolerance_boundary_survives_validation_refactor() -> None:
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        domain_atol=0.2,
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"Ar": 1.0e20, "Ar_plus": 1.0e15}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    neutral_index = model.layout.density_slices["z"].start
    state[neutral_index] = -2.0

    accepted = model.evaluate(0.0, state, model.segments[0])
    assert np.isfinite(accepted.derivative).all()

    state[neutral_index] = np.nextafter(-2.0, -np.inf)
    with pytest.raises(StateDomainError, match=r"-10\*domain_atol"):
        model.evaluate(0.0, state, model.segments[0])


def test_evaluation_ledger_reconstructs_energy_derivative() -> None:
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("plasma", 2.0),),
        segments=(
            RecipeSegment(
                "powered", 0.0, 1.0, absorbed_power_W_by_zone={"plasma": 6.0}
            ),
        ),
        electron_closure=ElectronEnergyClosure(),
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"plasma": {"Ar": 1.0e20, "Ar_plus": 1.0e15}},
            mean_energy_eV_by_zone={"plasma": 3.0},
        )
    )

    evaluated = model.evaluate(0.0, state, model.segments[0])
    ledger = evaluated.ledger_by_zone["plasma"]
    energy_index = model.layout.electron_energy_indices["plasma"]
    reconstructed = (
        ledger.absorbed_power_J_m3_s
        - ledger.reaction_energy_loss_J_m3_s
        - ledger.wall_energy_loss_J_m3_s
        - ledger.elastic_heating_J_m3_s
        + ledger.transport_electron_energy_J_m3_s
    )

    assert evaluated.derivative[energy_index] == pytest.approx(reconstructed)


def test_jacobian_sparsity_uses_chemistry_dependencies_with_no_cross_zone_fill() -> (
    None
):
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("upstream", 1.0), Zone("downstream", 1.0)),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    pattern = model.jac_sparsity.toarray()
    first = slice(0, 3)
    second = slice(3, 6)

    assert not np.any(pattern[first, first][0:2])
    assert np.all(pattern[first, first][2])
    assert not np.any(pattern[second, second][0:2])
    assert np.all(pattern[second, second][2])
    assert not np.any(pattern[first, second])
    assert not np.any(pattern[second, first])


def test_wall_flux_and_boundary_return_share_one_ledger() -> None:
    wall = WallBoundary(
        zone_id="plasma",
        area_m2=2.0,
        bohm_factor=0.61,
        reactions=(
            BoundaryReaction(
                reaction_id="neutralize",
                incident_species="Ar_plus",
                products={"Ar": 1.0},
            ),
        ),
    )
    model = CompiledGlobalModel(
        chemistry=inert_argon(),
        zones=(Zone("plasma", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        wall_boundaries=(wall,),
    )
    ion_density = 2.0e15
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"plasma": {"Ar": 1.0e20, "Ar_plus": ion_density}},
            mean_energy_eV_by_zone={"plasma": 3.0},
        )
    )
    evaluated = model.evaluate(0.0, state, model.segments[0])
    density_slice = model.layout.density_slices["plasma"]
    neutral_rhs, ion_rhs = evaluated.derivative[density_slice]
    record = evaluated.ledger_by_zone["plasma"].wall_fluxes[0]
    expected_speed = 0.61 * np.sqrt(ELEMENTARY_CHARGE_C * 2.0 / ARGON_MASS_KG)
    expected_rate = ion_density * expected_speed * 2.0

    assert record.bohm_speed_m_s == pytest.approx(expected_speed)
    assert record.incident_rate_m3_s == pytest.approx(expected_rate)
    assert record.branch_rates_m3_s["neutralize"] == pytest.approx(expected_rate)
    assert ion_rhs == pytest.approx(-expected_rate)
    assert neutral_rhs == pytest.approx(expected_rate)
    floating_sheath_eV = (
        0.5 * 2.0 * np.log(ARGON_MASS_KG / (2.0 * np.pi * ELECTRON_MASS_KG))
    )
    assert evaluated.ledger_by_zone["plasma"].wall_energy_loss_J_m3_s == pytest.approx(
        expected_rate * (4.0 + floating_sheath_eV) * ELEMENTARY_CHARGE_C
    )
