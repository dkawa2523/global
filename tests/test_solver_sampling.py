from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import plasma_global.core.solver as solver_module
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import (
    InitialState,
    RecipeSegment,
    SolverSettings,
    Zone,
)
from plasma_global.core.solver import solve_compiled_model
from plasma_global.errors import (
    IntegrationError,
    ModelConfigurationError,
    ModelDomainError,
)
from plasma_global.models.electrons import ElectronEnergyClosure


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


def first_order_chemistry(rate_s_inv: object = 2.0) -> ChemistryFixture:
    return ChemistryFixture(
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


def first_order_decay_chemistry(rate_s_inv: float = 1.0) -> ChemistryFixture:
    """B -> A isolates a trace-species decay beside an inert bulk density."""

    return ChemistryFixture(
        species_ids=("A", "B", "ion"),
        charges=np.array([0.0, 0.0, 1.0]),
        masses_kg=np.array([2.0e-26, 2.0e-26, 6.0e-26]),
        reaction_ids=("B_to_A",),
        stoichiometry=np.array([[1.0, -1.0, 0.0]]),
        reactant_orders=np.array([[0.0, 1.0, 0.0]]),
        electron_orders=np.array([0.0]),
        rate_evaluators=(rate_s_inv,),
        energy_loss_eV=np.array([0.0]),
        gas_heating_eV=np.zeros(1),
        reaction_zones=((),),
    )


def compiled_model(
    *segments: RecipeSegment, chemistry: ChemistryFixture | None = None
) -> CompiledGlobalModel:
    return CompiledGlobalModel(
        chemistry=chemistry or first_order_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=segments,
        electron_closure=ElectronEnergyClosure(),
    )


def initial_state() -> InitialState:
    return InitialState(
        densities_m3_by_zone={"z": {"A": 1.0e18, "B": 0.0, "ion": 1.0e14}},
        mean_energy_eV_by_zone={"z": 3.0},
    )


def test_native_sampling_returns_bdf_points_and_analytic_solution() -> None:
    model = compiled_model(RecipeSegment("native", 0.0, 1.0))

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(rtol=1.0e-9, atol=1.0e-11),
    )

    assert result.metadata["sampling_mode"] == "native"
    assert result.time_s[0] == 0.0
    assert result.time_s[-1] == 1.0
    assert result.time_s.size > 2
    assert result.final_value("n[z,A]") == pytest.approx(
        1.0e18 * np.exp(-2.0), rel=2.0e-7
    )


def test_sample_interval_is_global_and_includes_every_forcing_boundary() -> None:
    model = compiled_model(
        RecipeSegment("first", 0.1, 0.55),
        RecipeSegment("second", 0.55, 1.05),
    )

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(sample_interval_s=0.3),
    )

    assert result.metadata["sampling_mode"] == "sample_interval"
    assert np.allclose(result.time_s, [0.1, 0.4, 0.55, 0.7, 1.0, 1.05])


def test_save_at_is_global_and_includes_every_forcing_boundary() -> None:
    model = compiled_model(
        RecipeSegment("first", 0.1, 0.55),
        RecipeSegment("second", 0.55, 1.05),
    )

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(save_at_s=(0.2, 0.9)),
    )

    assert result.metadata["sampling_mode"] == "save_at"
    assert np.allclose(result.time_s, [0.1, 0.2, 0.55, 0.9, 1.05])


def test_save_at_preserves_distinct_subpicosecond_forcing_boundaries() -> None:
    end_s = 1.0e-15
    model = compiled_model(RecipeSegment("short", 0.0, end_s))

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(save_at_s=(0.0, end_s)),
    )

    assert np.array_equal(result.time_s, [0.0, end_s])


def test_result_contract_records_segment_statistics_and_scaling() -> None:
    model = compiled_model(
        RecipeSegment("first", 0.0, 0.4),
        RecipeSegment("second", 0.4, 1.0),
        chemistry=first_order_chemistry(rate_s_inv=0.0),
    )

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(save_at_s=(0.0, 0.25, 0.75, 1.0)),
    )

    assert np.array_equal(result.time_s, [0.0, 0.25, 0.4, 0.75, 1.0])
    assert result.solver_stats.keys() == {"nfev", "njev", "nlu", "segments_completed"}
    assert result.solver_stats["segments_completed"] == 2
    assert result.solver_stats["nfev"] > 0
    assert result.metadata["sampling_mode"] == "save_at"
    assert len(result.metadata["state_scale"]) == model.layout.size
    assert len(result.metadata["domain_atol"]) == model.layout.size


def test_sampling_modes_are_exclusive_and_save_times_are_bounded() -> None:
    with pytest.raises(ModelConfigurationError, match="mutually exclusive"):
        SolverSettings(sample_interval_s=0.1, save_at_s=(0.0, 1.0))

    model = compiled_model(RecipeSegment("only", 0.1, 1.0))
    with pytest.raises(ModelConfigurationError, match="recipe interval"):
        solve_compiled_model(model, initial_state(), SolverSettings(save_at_s=(0.0,)))


def test_sampling_has_a_fixed_limit_before_allocating_the_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = compiled_model(RecipeSegment("only", 0.0, 1.0))
    monkeypatch.setattr(
        solver_module.np,
        "arange",
        lambda *args, **kwargs: pytest.fail("sampling grid was allocated"),
    )

    with pytest.raises(ModelConfigurationError, match="saved-point limit"):
        solve_compiled_model(
            model,
            initial_state(),
            SolverSettings(sample_interval_s=1.0e-12),
        )


def test_trace_decay_accuracy_does_not_depend_on_unrelated_bulk_density() -> None:
    duration_s = 10.0
    trace_density_m3 = 1.0e6
    model = compiled_model(
        RecipeSegment("trace-decay", 0.0, duration_s),
        chemistry=first_order_decay_chemistry(),
    )
    controls = SolverSettings(
        rtol=1.0e-8,
        atol=1.0e-12,
        save_at_s=(0.0, duration_s),
    )

    final_trace = []
    for bulk_density_m3 in (1.0e12, 1.0e20):
        result = solve_compiled_model(
            model,
            InitialState(
                densities_m3_by_zone={
                    "z": {"A": bulk_density_m3, "B": trace_density_m3, "ion": 1.0}
                },
                mean_energy_eV_by_zone={"z": 3.0},
            ),
            controls,
        )
        final_trace.append(result.final_value("n[z,B]"))

    expected = trace_density_m3 * np.exp(-duration_s)
    assert final_trace == pytest.approx([expected, expected], rel=2.0e-6)
    assert final_trace[0] == pytest.approx(final_trace[1], rel=1.0e-12)


def test_zero_seeded_product_uses_its_own_scale_and_preserves_particles() -> None:
    model = compiled_model(
        RecipeSegment("production", 0.0, 1.0),
        chemistry=first_order_chemistry(rate_s_inv=1.0),
    )
    initial_a = 1.0e20

    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={"z": {"A": initial_a, "B": 0.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        ),
        SolverSettings(rtol=1.0e-8, atol=1.0e-12, save_at_s=(0.0, 1.0)),
    )

    density_slice = model.layout.density_slices["z"]
    state_scale = np.asarray(result.metadata["state_scale"], dtype=float)
    assert state_scale[density_slice.start + 1] == 1.0
    assert result.final_value("n[z,A]") == pytest.approx(
        initial_a * np.exp(-1.0), rel=2.0e-6
    )
    assert result.final_value("n[z,A]") + result.final_value("n[z,B]") == pytest.approx(
        initial_a, rel=2.0e-12
    )


def test_bdf_can_reject_a_negative_newton_trial_without_domain_failure() -> None:
    model = compiled_model(
        RecipeSegment("stiff-decay", 0.0, 1.0),
        chemistry=first_order_decay_chemistry(rate_s_inv=2.0),
    )
    initial_b = 1.0e12

    result = solve_compiled_model(
        model,
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0e20, "B": initial_b, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        ),
        SolverSettings(
            rtol=1.0e-8,
            atol=1.0e-12,
            first_step_s=1.0,
            save_at_s=(0.0, 1.0),
        ),
    )

    assert result.final_value("n[z,B]") == pytest.approx(
        initial_b * np.exp(-2.0), rel=2.0e-6
    )
    assert result.final_value("n[z,B]") >= 0.0


def test_solver_control_validation_is_fail_fast() -> None:
    with pytest.raises(ModelConfigurationError, match=r"rtol < 1"):
        SolverSettings(rtol=1.0)
    with pytest.raises(ModelConfigurationError, match=r"<= rtol"):
        SolverSettings(rtol=1.0e-20)
    with pytest.raises(ModelConfigurationError, match="must not exceed max_step_s"):
        SolverSettings(first_step_s=0.2, max_step_s=0.1)

    model = compiled_model(RecipeSegment("short", 0.0, 0.1))
    with pytest.raises(ModelConfigurationError, match="segment duration"):
        solve_compiled_model(
            model,
            initial_state(),
            SolverSettings(first_step_s=0.2),
        )


def test_invalid_initial_state_fails_before_segment_integration() -> None:
    model = compiled_model(RecipeSegment("bad-domain", 0.0, 1.0))
    bad_state = model.initial_state(initial_state())
    bad_state[model.layout.density_slices["z"].start] = -1.0e8

    with pytest.raises(ModelConfigurationError, match="compiled lower bound"):
        solve_compiled_model(model, bad_state)


def test_domain_failure_reports_segment_and_time() -> None:
    def fail_rate(context: object) -> float:
        del context
        raise ModelDomainError("invalid reaction state")

    model = compiled_model(
        RecipeSegment("bad-domain", 0.0, 1.0),
        chemistry=first_order_chemistry(rate_s_inv=fail_rate),
    )

    with pytest.raises(IntegrationError, match=r"segment 'bad-domain' at t=.* s"):
        solve_compiled_model(model, initial_state())


def test_initial_vector_rejects_wrong_shape_and_nonfinite_values() -> None:
    model = compiled_model(RecipeSegment("initial", 0.0, 1.0))

    with pytest.raises(ValueError, match=r"Initial state has shape .* expected"):
        solve_compiled_model(model, np.zeros(model.layout.size + 1))

    nonfinite_state = model.initial_state(initial_state())
    nonfinite_state[0] = np.nan
    with pytest.raises(ModelConfigurationError, match="only finite values"):
        solve_compiled_model(model, nonfinite_state)
