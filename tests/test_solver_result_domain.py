from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import plasma_global.core.solver as solver_module
from plasma_global.chemistry.data import (
    ChemistryData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.core._solver_domain import (
    _accepted_state_for_result,
    canonicalize_result_state,
)
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, SolverSettings, Zone
from plasma_global.core.solver import solve_compiled_model
from plasma_global.errors import IntegrationError, ModelConfigurationError
from plasma_global.experimental.accumulators import (
    GenericState,
    GenericStateAccumulator,
    LinearRelaxation,
)
from plasma_global.experimental.runtime import ExperimentalRuntimeAccumulator
from plasma_global.models.electrons import (
    ElectronEnergyClosure,
    LocalFieldClosure,
    TabulatedMeanEnergy,
)
from plasma_global.models.gas_energy import BOLTZMANN_J_K, HeavyEnergyClosure
from plasma_global.models.kinetics import TabulatedElectronKinetics
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry
from plasma_global.models.walls import BoundaryReaction, WallBoundary
from plasma_global.output import read_result_h5, write_result_h5
from plasma_global.postprocess import derive_observables


@dataclass(frozen=True)
class _InertChemistry:
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
        count = len(self.species_ids)
        return np.zeros((count, count), dtype=bool)


def _inert_chemistry(
    species_ids: tuple[str, ...], charges: tuple[float, ...]
) -> _InertChemistry:
    count = len(species_ids)
    return _InertChemistry(
        species_ids=species_ids,
        charges=np.asarray(charges),
        masses_kg=np.full(count, 6.0e-26),
        reaction_ids=(),
        stoichiometry=np.zeros((0, count)),
        reactant_orders=np.zeros((0, count)),
        electron_orders=np.zeros(0),
        rate_evaluators=(),
        energy_loss_eV=np.zeros(0),
        gas_heating_eV=np.zeros(0),
        reaction_zones=(),
    )


def _electron_model(
    segments: tuple[RecipeSegment, ...], *, wall_frequency_s_inv: float | None = None
) -> CompiledGlobalModel:
    walls = ()
    if wall_frequency_s_inv is not None:
        walls = (
            WallBoundary(
                "z",
                1.0,
                transport_kind="prescribed_frequency",
                prescribed_frequency_s_inv=wall_frequency_s_inv,
                reactions=(BoundaryReaction("neutralize", "ion", products={"A": 1.0}),),
            ),
        )
    return CompiledGlobalModel(
        chemistry=_inert_chemistry(("A", "ion"), (0.0, 1.0)),
        zones=(Zone("z", 1.0),),
        segments=segments,
        electron_closure=ElectronEnergyClosure(),
        wall_boundaries=walls,
    )


def _electron_row(model: CompiledGlobalModel, energy: float) -> np.ndarray:
    row = np.zeros(model.layout.size)
    row[model.layout.density_slices["z"].start] = 1.0
    row[model.layout.electron_energy_indices["z"]] = energy
    return row


def test_cold_electron_result_is_strictly_reusable_and_hdf_stable(
    tmp_path: Path,
) -> None:
    segment = RecipeSegment("drain", 0.0, 1.0)
    model = _electron_model((segment,), wall_frequency_s_inv=1.0e3)
    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0e20, "ion": 1.0e6}},
            mean_energy_eV_by_zone={"z": 3.0},
        ),
        SolverSettings(
            rtol=1.0e-6,
            atol=1.0e-12,
            first_step_s=1.0,
            save_at_s=(0.0, 1.0),
        ),
    )
    energy_index = model.layout.electron_energy_indices["z"]

    assert result.state[-1, energy_index] == 0.0
    assert (
        result.metadata["provenance"]["accepted_state_zeroed_electron_energy_count"]
        == 1
    )
    observables = derive_observables(
        model, result.time_s, result.state, ("mean_energy_eV[z]",)
    )
    assert observables["mean_energy_eV[z]"][-1] == 0.0

    loaded = read_result_h5(write_result_h5(tmp_path / "result.h5", result))
    np.testing.assert_array_equal(loaded.state, result.state)
    loaded_observables = derive_observables(
        model, loaded.time_s, loaded.state, ("mean_energy_eV[z]",)
    )
    np.testing.assert_array_equal(
        loaded_observables["mean_energy_eV[z]"],
        observables["mean_energy_eV[z]"],
    )


def test_result_canonicalization_uses_the_owner_of_a_segment_boundary() -> None:
    first = RecipeSegment(
        "prescribed",
        0.0,
        0.5,
        prescribed_electron_density_m3_by_zone={"z": 1.0},
    )
    second = RecipeSegment(
        "cold", 0.5, 1.0, prescribed_electron_density_m3_by_zone={"z": 0.0}
    )
    model = _electron_model((first, second))
    times = np.array([0.5, 0.75])
    state = np.vstack([_electron_row(model, 1.0e-13)] * 2)
    tolerance = np.full(model.layout.size, 1.0e-12)

    accepted, _, electron_count = canonicalize_result_state(
        model, state, tolerance, times
    )
    energy_index = model.layout.electron_energy_indices["z"]

    assert accepted[0, energy_index] == 1.0e-13
    assert accepted[1, energy_index] == 0.0
    assert electron_count == 1
    model.evaluate(times[0], accepted[0], first)
    model.evaluate(times[1], accepted[1], second)


def test_material_cold_electron_energy_is_not_reported_as_success() -> None:
    segment = RecipeSegment("cold", 0.0, 1.0)
    model = _electron_model((segment,))
    tolerance = np.full(model.layout.size, 1.0e-12)
    state = _electron_row(model, 11.0e-12)[None, :]

    with pytest.raises(IntegrationError, match=r"cold.*10\*domain_atol"):
        canonicalize_result_state(model, state, tolerance, np.array([1.0]))


def test_nonfinite_electron_mean_energy_is_not_reported_as_success() -> None:
    segment = RecipeSegment("charged", 0.0, 1.0)
    model = _electron_model((segment,))
    state = _electron_row(model, 1.0)
    state[model.layout.density_slices["z"].start + 1] = np.nextafter(0.0, 1.0)

    with pytest.raises(IntegrationError, match="mean energy is not finite"):
        canonicalize_result_state(
            model,
            state[None, :],
            np.zeros(model.layout.size),
            np.array([1.0]),
        )


def test_result_electron_mean_energy_must_fit_its_table_domain() -> None:
    segment = RecipeSegment("charged", 0.0, 1.0)
    table = TabulatedElectronKinetics(
        source=Path("result-domain-table"),
        lookup="mean_energy",
        bounds="error",
        axis=np.array([1.0, 2.0]),
        mean_energy_eV=np.array([1.0, 2.0]),
        mobility_m2_V_s=np.array([1.0, 1.0]),
        effective_field_Td=np.array([1.0, 2.0]),
        rate_tables={},
    )
    model = CompiledGlobalModel(
        chemistry=_inert_chemistry(("A", "ion"), (0.0, 1.0)),
        zones=(Zone("z", 1.0),),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
        electron_kinetics_by_zone={"z": table},
    )
    state = _electron_row(model, 1.0e-12)
    state[model.layout.density_slices["z"].start + 1] = 1.0e-300

    with pytest.raises(IntegrationError, match=r"electron table lookup.*outside"):
        canonicalize_result_state(
            model,
            state[None, :],
            np.full(model.layout.size, 1.0e-12),
            np.array([1.0]),
        )


def _heavy_model() -> CompiledGlobalModel:
    return CompiledGlobalModel(
        chemistry=_inert_chemistry(("A",), (0.0,)),
        zones=(Zone("z", 1.0),),
        segments=(
            RecipeSegment("heavy", 0.0, 1.0, reduced_field_Td_by_zone={"z": 1.0}),
        ),
        electron_closure=LocalFieldClosure(TabulatedMeanEnergy((1.0, 2.0), (3.0, 3.0))),
        heavy_energy_closure=HeavyEnergyClosure(
            cv_over_kb=np.array([1.5]),
            wall_temperature_K=np.array([300.0]),
            wall_relaxation_s_inv=np.zeros(1),
        ),
    )


def test_solver_rejects_an_accepted_empty_heavy_mixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _heavy_model()
    initial = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0}},
            gas_temperature_K_by_zone={"z": 300.0},
        )
    )

    def empty_mixture_solution(**_options: Any) -> Any:
        return type(
            "Solution",
            (),
            {
                "success": True,
                "message": "ok",
                "t": np.array([0.0, 1.0]),
                "y": np.column_stack((initial, np.array([-1.0e-13, 1.0e-23]))),
                "t_events": (),
                "nfev": 1,
                "njev": 0,
                "nlu": 0,
            },
        )()

    monkeypatch.setattr(solver_module, "solve_ivp", empty_mixture_solution)

    with pytest.raises(IntegrationError, match="empty mixture"):
        solve_compiled_model(
            model,
            initial,
            SolverSettings(atol=1.0e-12, save_at_s=(0.0, 1.0)),
        )


def test_zero_heavy_energy_for_a_nonempty_mixture_is_rejected() -> None:
    model = _heavy_model()
    state = np.array([[1.0, -1.0e-13]])

    with pytest.raises(IntegrationError, match="gas internal energy is nonpositive"):
        canonicalize_result_state(
            model,
            state,
            np.full(model.layout.size, 1.0e-12),
            np.array([1.0]),
        )


def test_nonfinite_gas_temperature_is_not_reported_as_success() -> None:
    model = _heavy_model()
    state = np.array([[1.0e-300, 1.0]])

    with pytest.raises(IntegrationError, match="temperature is not finite"):
        canonicalize_result_state(
            model,
            state,
            np.full(model.layout.size, 1.0e-12),
            np.array([1.0]),
        )


def test_finite_gas_temperature_remains_valid_at_very_low_density() -> None:
    model = _heavy_model()
    density = 1.0e-200
    energy = BOLTZMANN_J_K * 1.5 * density * 300.0
    state = np.array([[density, energy]])

    accepted, _, _ = canonicalize_result_state(
        model,
        state,
        np.full(model.layout.size, 1.0e-250),
        np.array([1.0]),
    )

    evaluation = model.evaluate(1.0, accepted[0], model.segments[0])
    assert evaluation.gas_temperature_K_by_zone["z"] == pytest.approx(300.0)
    assert np.all(np.isfinite(evaluation.derivative))


def test_result_snaps_only_tolerance_scale_positive_lower_bound_undershoot() -> None:
    accepted, zeroed_count = _accepted_state_for_result(
        np.array([[4.5, -0.5]]),
        np.array([0.1, 0.1]),
        lower_bounds=np.array([5.0, 0.0]),
    )

    np.testing.assert_array_equal(accepted, np.array([[5.0, 0.0]]))
    assert zeroed_count == 1

    with pytest.raises(IntegrationError, match="below lower_bound"):
        _accepted_state_for_result(
            np.array([[3.99]]),
            np.array([0.1]),
            lower_bounds=np.array([5.0]),
        )


def _bounded_extension_model() -> CompiledGlobalModel:
    segment = RecipeSegment("bounded-extension", 0.0, 1.0)
    generic = GenericStateAccumulator(
        states=(
            GenericState(
                "marker",
                owners=("global",),
                initial=4.0,
                lower_bound=1.0,
                upper_bound=5.0,
            ),
        ),
        processes=(LinearRelaxation("marker", equilibrium=0.0, time_constant_s=1.0),),
    )
    return CompiledGlobalModel(
        chemistry=_inert_chemistry(("A", "ion"), (0.0, 1.0)),
        zones=(Zone("z", 1.0),),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
        extension_accumulator=ExperimentalRuntimeAccumulator(generic=generic),
    )


def test_extension_upper_bound_is_compiled_and_continues_solver_trials() -> None:
    model = _bounded_extension_model()
    extension_index = model.layout.extension_slice.start
    assert model.state_upper_bounds[extension_index] == 5.0
    assert not model.state_upper_bounds.flags.writeable

    initial = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    trial = initial.copy()
    trial[extension_index] = 6.0
    scaled_rhs = solver_module._ScaledSegmentRHS(
        model.bind_segment(model.segments[0], domain_atol=np.zeros(model.layout.size)),
        np.ones(model.layout.size),
        model.state_lower_bounds,
        model.state_upper_bounds,
    )

    derivative = scaled_rhs(0.0, trial)

    assert derivative[extension_index] == pytest.approx(-5.0)
    assert trial[extension_index] == 6.0


def test_result_snaps_only_tolerance_scale_upper_bound_overshoot() -> None:
    model = _bounded_extension_model()
    extension_index = model.layout.extension_slice.start
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    tolerance = np.full(model.layout.size, 0.1)
    state[extension_index] = 5.5

    accepted, _, _ = canonicalize_result_state(
        model, state[None, :], tolerance, np.array([1.0])
    )

    assert accepted[0, extension_index] == 5.0
    model.evaluate(1.0, accepted[0], model.segments[0])

    state[extension_index] = 6.01
    with pytest.raises(
        IntegrationError,
        match=r"upper_bound\+10\*domain_atol.*bounded-extension",
    ):
        canonicalize_result_state(model, state[None, :], tolerance, np.array([1.0]))


def test_initial_extension_state_must_respect_its_upper_bound() -> None:
    model = _bounded_extension_model()
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    state[model.layout.extension_slice.start] = 5.1

    with pytest.raises(ModelConfigurationError, match="compiled upper bound"):
        solve_compiled_model(
            model,
            state,
            SolverSettings(save_at_s=(0.0, 1.0)),
        )


def _surface_result_model(surface_ids: tuple[str, ...]) -> CompiledGlobalModel:
    chemistry = ChemistryData(
        source=Path("surface-result.yaml"),
        species=(
            SpeciesData("A", "gas", 0, 36.0, {"X": 1.0}),
            SpeciesData("ion", "gas", 1, 36.0, {"X": 1.0}),
            SpeciesData(
                "site",
                "surface",
                0,
                1.0,
                {"site": 1.0},
                state_tags=frozenset({"site"}),
            ),
            SpeciesData("Ads", "surface", 0, 36.0, {"X": 1.0, "site": 1.0}),
        ),
        gas_reactions=(),
        boundary_reactions=(),
        surface_reactions=(
            ReactionData(
                "stick",
                {"A": 1.0, "site": 1.0},
                {"Ads": 1.0},
                "stick",
                None,
            ),
        ),
        rate_models={"stick": RateModelData("stick", "sticking", {"value": 0.2})},
        cross_sections={},
    )
    surface = CompiledSurfaceModel(
        chemistry=chemistry,
        gas_species_ids=("A", "ion"),
        gas_masses_kg=np.array([6.0e-26, 6.0e-26]),
        zone_ids=("z",),
        zone_volumes_m3=np.array([1.0]),
        surfaces=tuple(
            SurfaceGeometry(surface_id, "z", 1.0, 1.0e19, 300.0)
            for surface_id in surface_ids
        ),
    )
    return CompiledGlobalModel(
        chemistry=_inert_chemistry(("A", "ion"), (0.0, 1.0)),
        zones=(Zone("z", 1.0, 400.0),),
        segments=(RecipeSegment("surface", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        surface_model=surface,
    )


def _surface_result_state(model: CompiledGlobalModel) -> np.ndarray:
    return model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0e20, "ion": 1.0e15}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )


@pytest.mark.parametrize("invalid_coverage", [1.01, 2.0, 100.0])
def test_surface_result_maps_only_tolerance_scale_excess(
    invalid_coverage: float,
) -> None:
    model = _surface_result_model(("wall",))
    state = _surface_result_state(model)
    coverage_index = model.layout.surface_coverage_indices[("wall", "Ads")]
    tolerance = np.full(model.layout.size, 1.0e-10)
    state[coverage_index] = 1.0 + 5.0e-10

    accepted, _, _ = canonicalize_result_state(
        model, state[None, :], tolerance, np.array([1.0])
    )
    assert accepted[0, coverage_index] == 1.0

    state[coverage_index] = invalid_coverage
    with pytest.raises(IntegrationError, match=r"surface.*negative algebraic"):
        canonicalize_result_state(model, state[None, :], tolerance, np.array([1.0]))


def test_surface_result_checks_each_site_simplex() -> None:
    model = _surface_result_model(("wall_a", "wall_b"))
    state = _surface_result_state(model)
    wall_a = model.layout.surface_coverage_indices[("wall_a", "Ads")]
    wall_b = model.layout.surface_coverage_indices[("wall_b", "Ads")]
    tolerance = np.full(model.layout.size, 1.0e-10)
    state[wall_a] = 1.0 + 5.0e-10
    state[wall_b] = 0.5

    accepted, _, _ = canonicalize_result_state(
        model, state[None, :], tolerance, np.array([1.0])
    )
    assert accepted[0, wall_a] == 1.0
    assert accepted[0, wall_b] == 0.5

    state[wall_b] = 1.01
    with pytest.raises(IntegrationError, match=r"surface.*negative algebraic"):
        canonicalize_result_state(model, state[None, :], tolerance, np.array([1.0]))
