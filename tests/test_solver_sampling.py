from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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


def autocatalytic_chemistry(rate_m3_s: float) -> ChemistryFixture:
    """A + B -> 2 B exposes growth hidden by an initially small RHS."""

    return ChemistryFixture(
        species_ids=("A", "B", "ion"),
        charges=np.array([0.0, 0.0, 1.0]),
        masses_kg=np.array([2.0e-26, 2.0e-26, 6.0e-26]),
        reaction_ids=("A_plus_B_to_2B",),
        stoichiometry=np.array([[-1.0, 1.0, 0.0]]),
        reactant_orders=np.array([[1.0, 1.0, 0.0]]),
        electron_orders=np.array([0.0]),
        rate_evaluators=(rate_m3_s,),
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
    assert result.metadata["provenance"]["accepted_state_zeroed_negative_count"] == 0


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


def test_first_and_max_step_are_forwarded_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = compiled_model(
        RecipeSegment("first", 0.0, 0.5),
        RecipeSegment("second", 0.5, 1.0),
    )
    real_solve_ivp = solver_module.solve_ivp
    calls: list[dict[str, object]] = []

    def recording_solve_ivp(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_solve_ivp(**kwargs)

    monkeypatch.setattr(solver_module, "solve_ivp", recording_solve_ivp)
    solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(
            first_step_s=0.01,
            max_step_s=0.1,
            save_at_s=(0.0, 1.0),
        ),
    )
    assert [call["first_step"] for call in calls] == [0.01, 0.01]
    assert [call["max_step"] for call in calls] == [0.1, 0.1]

    calls.clear()
    solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(save_at_s=(0.0, 1.0)),
    )
    assert all("first_step" not in call and "max_step" not in call for call in calls)


def test_bdf_integrates_scaled_state_with_dimensionless_scalar_atol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = compiled_model(RecipeSegment("scaled", 0.0, 0.1))
    real_solve_ivp = solver_module.solve_ivp
    calls: list[dict[str, object]] = []

    def recording_solve_ivp(**kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_solve_ivp(**kwargs)

    monkeypatch.setattr(solver_module, "solve_ivp", recording_solve_ivp)
    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(rtol=1.0e-8, atol=1.0e-10),
    )

    call = calls[0]
    scaled_initial = np.asarray(call["y0"], dtype=float)
    density_slice = model.layout.density_slices["z"]
    assert np.array_equal(scaled_initial[density_slice], [1.0, 0.0, 1.0])
    assert call["atol"] == 1.0e-10
    state_scale = np.asarray(result.metadata["state_scale"], dtype=float)
    physical_atol = np.asarray(result.metadata["domain_atol"], dtype=float)
    assert np.array_equal(state_scale[density_slice], [1.0e18, 1.0, 1.0e14])
    assert np.array_equal(physical_atol, 1.0e-10 * state_scale)


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


def test_experimental_quasi_steady_holds_autonomous_segment_to_output_end() -> None:
    inert = first_order_chemistry(rate_s_inv=0.0)
    model = compiled_model(RecipeSegment("hold", 0.0, 1.0), chemistry=inert)

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(
            sample_interval_s=0.2,
            experimental_quasi_steady_threshold_s_inv=1.0e-12,
            experimental_quasi_steady_min_time_s=0.25,
        ),
    )

    assert result.solver_stats["quasi_steady_events"] == 1
    assert result.metadata["experimental_stop_policy"] == (
        "experimental.stop_when_quasi_steady"
    )
    assert (
        result.metadata["experimental_quasi_steady_event_mode"]
        == "nonterminal_observation_v1"
    )
    assert np.allclose(result.time_s, np.linspace(0.0, 1.0, 6))
    assert np.all(result.state == result.state[0])


def test_experimental_quasi_steady_cannot_freeze_slow_unfinished_dynamics() -> None:
    duration_s = 1.0e6
    rate_s_inv = 1.0e-6
    model = compiled_model(
        RecipeSegment("slow", 0.0, duration_s),
        chemistry=first_order_chemistry(rate_s_inv=rate_s_inv),
    )

    result = solve_compiled_model(
        model,
        initial_state(),
        SolverSettings(
            rtol=1.0e-8,
            atol=1.0e-10,
            save_at_s=(0.0, duration_s),
            experimental_quasi_steady_threshold_s_inv=1.0e-3,
            experimental_quasi_steady_min_time_s=1.0,
        ),
    )

    expected_a = 1.0e18 * np.exp(-rate_s_inv * duration_s)
    assert result.solver_stats["quasi_steady_events"] == 1
    assert result.final_value("n[z,A]") == pytest.approx(expected_a, rel=2.0e-6)


def test_experimental_quasi_steady_observation_never_freezes_future_growth() -> None:
    segment = RecipeSegment("autocatalytic", 0.0, 1_000.0)
    model = compiled_model(
        segment,
        chemistry=autocatalytic_chemistry(rate_m3_s=3.0e-20),
    )
    state = InitialState(
        densities_m3_by_zone={"z": {"A": 1.0e18, "B": 1.0, "ion": 1.0e14}},
        mean_energy_eV_by_zone={"z": 3.0},
    )
    integration_controls = {
        "rtol": 1.0e-8,
        "atol": 1.0e-12,
        "max_step_s": 0.1,
        "save_at_s": (0.0, segment.end_s),
    }

    reference = solve_compiled_model(
        model,
        state,
        SolverSettings(**integration_controls),
    )
    observed = solve_compiled_model(
        model,
        state,
        SolverSettings(
            **integration_controls,
            experimental_quasi_steady_threshold_s_inv=1.0e-3,
        ),
    )

    assert observed.final_value("n[z,B]") == pytest.approx(
        reference.final_value("n[z,B]"), rel=1.0e-10
    )
    assert observed.final_value("n[z,B]") > 1.0e12
    assert observed.time_s[-1] == segment.end_s


def test_experimental_quasi_steady_rejects_time_dependent_rate() -> None:
    class DynamicRate:
        time_dependent = True

        def __call__(self, context: object) -> float:
            del context
            return 0.0

    model = compiled_model(
        RecipeSegment("dynamic", 0.0, 1.0),
        chemistry=first_order_chemistry(rate_s_inv=DynamicRate()),
    )

    with pytest.raises(ModelConfigurationError, match="time-invariant"):
        solve_compiled_model(
            model,
            initial_state(),
            SolverSettings(experimental_quasi_steady_threshold_s_inv=1.0e-3),
        )


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


def test_solver_setup_failure_reports_segment_and_start_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = compiled_model(RecipeSegment("bad-setup", 0.25, 1.0))

    def fail_setup(**kwargs: object) -> object:
        del kwargs
        raise ValueError("invalid solver option")

    monkeypatch.setattr(solver_module, "solve_ivp", fail_setup)

    with pytest.raises(
        IntegrationError,
        match=(
            r"Integration setup failed in segment 'bad-setup' "
            r"at t=0\.25 s: invalid solver option"
        ),
    ):
        solve_compiled_model(model, initial_state())


def test_initial_vector_rejects_wrong_shape_and_nonfinite_values() -> None:
    model = compiled_model(RecipeSegment("initial", 0.0, 1.0))

    with pytest.raises(ValueError, match=r"Initial state has shape .* expected"):
        solve_compiled_model(model, np.zeros(model.layout.size + 1))

    nonfinite_state = model.initial_state(initial_state())
    nonfinite_state[0] = np.nan
    with pytest.raises(ModelConfigurationError, match="only finite values"):
        solve_compiled_model(model, nonfinite_state)


@pytest.mark.parametrize(
    ("solver_times", "failure_time"),
    [
        (np.array([0.25, 0.5]), "0.5"),
        (np.empty(0, dtype=float), "0.25"),
    ],
)
def test_unsuccessful_solver_response_reports_last_available_time(
    monkeypatch: pytest.MonkeyPatch,
    solver_times: np.ndarray,
    failure_time: str,
) -> None:
    model = compiled_model(RecipeSegment("failed", 0.25, 1.0))
    failed_solution = SimpleNamespace(
        success=False,
        t=solver_times,
        message="step size failed",
    )

    def return_failed_solution(**kwargs: object) -> object:
        del kwargs
        return failed_solution

    monkeypatch.setattr(solver_module, "solve_ivp", return_failed_solution)

    with pytest.raises(
        IntegrationError,
        match=(
            rf"Integration failed in segment 'failed' at t={failure_time} s: "
            "step size failed"
        ),
    ):
        solve_compiled_model(model, initial_state())


def test_solver_success_without_state_is_reported_as_integration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = compiled_model(RecipeSegment("no-state", 0.25, 1.0))
    empty_solution = SimpleNamespace(
        success=True,
        t=np.empty(0, dtype=float),
        y=np.empty((model.layout.size, 0), dtype=float),
        nfev=0,
        njev=0,
        nlu=0,
    )

    def return_empty_solution(**kwargs: object) -> object:
        del kwargs
        return empty_solution

    monkeypatch.setattr(solver_module, "solve_ivp", return_empty_solution)

    with pytest.raises(
        IntegrationError,
        match=(
            r"Integration failed in segment 'no-state' "
            r"at t=0\.25 s: solver returned no state"
        ),
    ):
        solve_compiled_model(model, initial_state())
