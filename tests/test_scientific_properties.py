from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from plasma_global import load_case
from plasma_global.chemistry.compile import compile_chemistry
from plasma_global.chemistry.data import (
    ChemistryData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, Zone
from plasma_global.core.result import SimulationResult
from plasma_global.core.transport import CompiledTransport, SegmentTransport
from plasma_global.errors import ModelConfigurationError
from plasma_global.input.migrate_v2 import migrate_v2, write_v3_case
from plasma_global.models.electrons import ElectronEnergyClosure
from plasma_global.models.rates import ConstantRate
from plasma_global.output import read_result_h5, write_result_h5

ROOT = Path(__file__).parents[1]
V2_CONFIGS = ROOT / "tests" / "fixtures" / "v2" / "configs"
V2_CHEMISTRY = (
    ROOT
    / "tests"
    / "fixtures"
    / "v2"
    / "chemistry_crane_two_reaction_argon"
    / "chemistry_manifest.yaml"
)


def _conversion_model(rate_s_inv: float) -> tuple[CompiledGlobalModel, np.ndarray]:
    rate = RateModelData("convert", "first_order", {"rate_s_inv": rate_s_inv})
    chemistry = ChemistryData(
        source=Path("property-chemistry.yaml"),
        species=(
            SpeciesData("e", "gas", -1, 0.00054858, {}),
            SpeciesData("A", "gas", 0, 10.0, {"X": 1.0}),
            SpeciesData("B", "gas", 0, 10.0, {"X": 1.0}),
            SpeciesData("X_plus", "gas", 1, 10.0, {"X": 1.0}),
        ),
        gas_reactions=(ReactionData("convert", {"A": 1.0}, {"B": 1.0}, rate.id, 0.0),),
        boundary_reactions=(),
        surface_reactions=(),
        rate_models={rate.id: rate},
        cross_sections={},
    )
    compiled = compile_chemistry(chemistry)
    segment = RecipeSegment("property", 0.0, 1.0)
    model = CompiledGlobalModel(
        chemistry=compiled,
        zones=(Zone("z", 1.0),),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
    )
    return model, compiled.element_matrix


@pytest.mark.property
@given(
    invalid=st.one_of(
        st.sampled_from((-math.inf, math.inf, math.nan)),
        st.integers(min_value=-1_000_000, max_value=-1).map(float),
    )
)
def test_constant_rate_rejects_every_generated_invalid_boundary(invalid: float) -> None:
    with pytest.raises(
        ModelConfigurationError,
        match="must be finite and non-negative",
    ):
        ConstantRate(invalid)


@pytest.mark.property
@given(
    rate_units=st.integers(min_value=0, max_value=10_000),
    density_a=st.integers(min_value=1, max_value=1_000_000),
    density_b=st.integers(min_value=0, max_value=1_000_000),
)
def test_first_order_rate_law_preserves_elements(
    rate_units: int,
    density_a: int,
    density_b: int,
) -> None:
    rate_s_inv = rate_units * 1.0e-4
    model, element_matrix = _conversion_model(rate_s_inv)
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={
                "z": {
                    "A": float(density_a),
                    "B": float(density_b),
                    "X_plus": 1.0,
                }
            },
            mean_energy_eV_by_zone={"z": 2.0},
        )
    )

    evaluated = model.evaluate(0.0, state, model.segments[0])
    density_rhs = evaluated.derivative[model.layout.density_slices["z"]]
    expected_rate = rate_s_inv * density_a

    assert density_rhs.shape == (3,)
    assert density_rhs.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        density_rhs,
        [-expected_rate, expected_rate, 0.0],
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        element_matrix @ density_rhs,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


@st.composite
def _transport_case(
    draw: st.DrawFn,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, bool]:
    species_count = draw(st.integers(min_value=1, max_value=4))
    volumes = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=20),
                min_size=2,
                max_size=2,
            )
        ),
        dtype=np.float64,
    )
    density = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=1_000_000),
                min_size=2 * species_count,
                max_size=2 * species_count,
            )
        ),
        dtype=np.float64,
    ).reshape(2, species_count)
    electron_energy = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=1_000_000),
                min_size=2,
                max_size=2,
            )
        ),
        dtype=np.float64,
    )
    heavy_energy = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=1_000_000),
                min_size=2,
                max_size=2,
            )
        ),
        dtype=np.float64,
    )
    conductance = draw(st.integers(min_value=0, max_value=100)) * 0.1
    return (
        volumes,
        conductance,
        density,
        electron_energy,
        heavy_energy,
        draw(st.booleans()),
    )


@pytest.mark.property
@given(case=_transport_case())
def test_transport_preserves_shape_dtype_and_volume_integrals(
    case: tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, bool],
) -> None:
    volumes, conductance, density, electron, heavy, reverse = case
    source, target = (1, 0) if reverse else (0, 1)
    transport = CompiledTransport(
        volumes_m3=volumes,
        pump_frequency_s_inv=np.zeros(2),
        edge_from=np.array([source]),
        edge_to=np.array([target]),
        edge_conductance_m3_s=np.array([conductance]),
        n_species=density.shape[1],
        heavy_cv_over_kb=np.ones(density.shape[1]),
    )
    forcing = SegmentTransport.zeros(2, density.shape[1])

    density_rhs, electron_rhs, heavy_rhs = transport.evaluate(
        density,
        forcing,
        electron_energy_J_m3=electron,
        heavy_energy_J_m3=heavy,
    )

    assert electron_rhs is not None
    assert heavy_rhs is not None
    assert density_rhs.shape == density.shape
    assert density_rhs.dtype == np.dtype(np.float64)
    assert electron_rhs.shape == electron.shape
    assert electron_rhs.dtype == np.dtype(np.float64)
    assert heavy_rhs.shape == heavy.shape
    assert heavy_rhs.dtype == np.dtype(np.float64)
    scale = max(
        1.0,
        float(np.max(np.abs(volumes[:, None] * density_rhs))),
        float(np.max(np.abs(volumes * electron_rhs))),
        float(np.max(np.abs(volumes * heavy_rhs))),
    )
    tolerance = 16.0 * np.finfo(np.float64).eps * scale
    np.testing.assert_allclose(
        volumes @ density_rhs,
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        volumes @ electron_rhs,
        0.0,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        volumes @ heavy_rhs,
        0.0,
        rtol=0.0,
        atol=tolerance,
    )


@st.composite
def _result_arrays(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_count = draw(st.integers(min_value=1, max_value=5))
    state_count = draw(st.integers(min_value=1, max_value=4))
    increments = draw(
        st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=time_count - 1,
            max_size=time_count - 1,
        )
    )
    time_s = np.concatenate(
        (np.zeros(1), np.cumsum(np.asarray(increments, dtype=np.float64)) * 1.0e-9)
    )
    state = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=-1_000_000, max_value=1_000_000),
                min_size=time_count * state_count,
                max_size=time_count * state_count,
            )
        ),
        dtype=np.float32,
    ).reshape(time_count, state_count)
    observable = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=-1_000_000, max_value=1_000_000),
                min_size=time_count,
                max_size=time_count,
            )
        ),
        dtype=np.float32,
    )
    return time_s, state, observable


@pytest.mark.property
@given(arrays=_result_arrays())
def test_hdf5_serialization_round_trip_preserves_numeric_contract(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    time_s, state, observable = arrays
    labels = tuple(f"state_{index}" for index in range(state.shape[1]))
    result = SimulationResult(
        time_s=time_s,
        state=state,
        state_labels=labels,
        observables={"signal": observable},
    )

    with tempfile.TemporaryDirectory() as directory:
        restored = read_result_h5(
            write_result_h5(Path(directory) / "result.h5", result)
        )

    assert restored.state.shape == state.shape
    assert restored.state.dtype == np.dtype(np.float64)
    assert restored.series("signal").shape == observable.shape
    assert restored.series("signal").dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(restored.time_s, result.time_s)
    np.testing.assert_array_equal(restored.state, result.state)
    np.testing.assert_array_equal(restored.series("signal"), result.series("signal"))


def _yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"fixture {path} must contain a mapping")
    return cast(dict[str, Any], value)


@pytest.mark.property
@given(
    ion_density=st.integers(min_value=1, max_value=1_000_000_000_000),
    duration_ns=st.integers(min_value=10, max_value=1_000),
)
def test_v2_to_v3_migration_preserves_generated_physical_values(
    ion_density: int,
    duration_ns: int,
) -> None:
    source = _yaml_mapping(V2_CONFIGS / "case_crane_two_reaction_argon.yaml")
    chamber = _yaml_mapping(V2_CONFIGS / "chamber_crane_two_reaction_argon.yaml")
    recipe = _yaml_mapping(V2_CONFIGS / "recipe_crane_two_reaction_argon.yaml")
    duration_s = duration_ns * 1.0e-9
    chamber["zones"][0]["initial_densities_m3"]["Ar_plus"] = ion_density
    recipe["steps"][0]["t_end_s"] = duration_s

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        chamber_path = temporary / "chamber.yaml"
        recipe_path = temporary / "recipe.yaml"
        chamber_path.write_text(yaml.safe_dump(chamber), encoding="utf-8")
        recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
        source["include"] = str((V2_CONFIGS / "base_case.yaml").resolve())
        source["files"]["chamber"] = str(chamber_path)
        source["files"]["recipe"] = str(recipe_path)
        source["files"]["chemistry"] = {"manifest": str(V2_CHEMISTRY.resolve())}
        source_path = temporary / "case-v2.yaml"
        source_path.write_text(yaml.safe_dump(source), encoding="utf-8")

        migrated = migrate_v2(source_path)
        reloaded = load_case(write_v3_case(migrated, temporary / "case-v3.yaml"))

    migrated_densities = migrated.case.reactor.zones[0].initial_densities_m3
    assert migrated_densities is not None
    assert migrated_densities["Ar_plus"] == float(ion_density)
    assert migrated.case.recipe.steps[0].duration_s == duration_s
    assert reloaded.model_dump(
        mode="json", exclude_none=True
    ) == migrated.case.model_dump(
        mode="json",
        exclude_none=True,
    )
