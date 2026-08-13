from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from plasma_global.chemistry.data import (
    ChemistryData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.errors import CaseValidationError, ModelDomainError
from plasma_global.models.surface import (
    BOLTZMANN_J_K,
    E_CHARGE,
    CompiledSurfaceModel,
    SurfaceGeometry,
)

AMU_TO_KG = 1.66053906660e-27


def _species(
    *additional_surface_species: SpeciesData,
) -> tuple[SpeciesData, ...]:
    return (
        SpeciesData("A", "gas", 0, 40.0, {"X": 1.0}),
        SpeciesData("ion", "gas", 1, 40.0, {"X": 1.0}),
        SpeciesData(
            "site",
            "surface",
            0,
            0.0,
            {"site": 1.0},
            state_tags=frozenset({"site"}),
        ),
        SpeciesData("Ads", "surface", 0, 40.0, {"X": 1.0, "site": 1.0}),
        *additional_surface_species,
    )


def _chemistry(
    *,
    species: tuple[SpeciesData, ...],
    reactions: tuple[ReactionData, ...] = (),
    rate_models: tuple[RateModelData, ...] = (),
) -> ChemistryData:
    return ChemistryData(
        source=Path("surface-contracts.yaml"),
        species=species,
        gas_reactions=(),
        boundary_reactions=(),
        surface_reactions=reactions,
        rate_models={model.id: model for model in rate_models},
        cross_sections={},
    )


def _model(
    chemistry: ChemistryData,
    surfaces: tuple[SurfaceGeometry, ...],
    *,
    zone_ids: tuple[str, ...] = ("z",),
    zone_volumes_m3: np.ndarray | None = None,
) -> CompiledSurfaceModel:
    return CompiledSurfaceModel(
        chemistry=chemistry,
        gas_species_ids=("A", "ion"),
        gas_masses_kg=np.array([40.0, 40.0]) * AMU_TO_KG,
        zone_ids=zone_ids,
        zone_volumes_m3=(
            np.ones(len(zone_ids)) if zone_volumes_m3 is None else zone_volumes_m3
        ),
        surfaces=surfaces,
    )


def test_layout_preserves_surface_and_species_order_and_multisite_balance() -> None:
    pair = SpeciesData("Pair", "surface", 0, 80.0, {"X": 2.0, "site": 2.0})
    film = SpeciesData(
        "Film",
        "surface",
        0,
        40.0,
        {"X": 1.0, "site": 1.0},
        state_tags=frozenset({"film_fragment"}),
    )
    chemistry = _chemistry(species=_species(pair, film))
    surfaces = (
        SurfaceGeometry(
            "wall_b",
            "z",
            1.0,
            1.0e19,
            300.0,
            {"Ads": 0.2, "Pair": 0.1},
        ),
        SurfaceGeometry(
            "wall_a",
            "z",
            1.0,
            1.0e19,
            300.0,
            {"Ads": 0.3, "Pair": 0.05},
        ),
    )

    model = _model(chemistry, surfaces)
    state = model.initial_state()

    assert model.layout.labels == (
        "coverage[wall_b,Ads]",
        "coverage[wall_b,Pair]",
        "coverage[wall_a,Ads]",
        "coverage[wall_a,Pair]",
    )
    np.testing.assert_array_equal(state, [0.2, 0.1, 0.3, 0.05])
    assert state.dtype == np.dtype(float)
    assert model.coverage(state, "wall_b", "site") == pytest.approx(0.6)
    assert model.coverage(state, "wall_a", "site") == pytest.approx(0.6)


def test_reactions_aggregate_particle_event_heat_and_coverage_rates() -> None:
    stick_model = RateModelData(
        "stick",
        "sticking",
        {
            "value": 0.2,
            "coverage": {
                "kind": "site_blocking",
                "site_species": "site",
                "exponent": 1.0,
            },
        },
    )
    ion_model = RateModelData(
        "ion",
        "ion_assisted",
        {
            "yield": 2.0,
            "threshold_eV": 2.0,
            "reference_energy_eV": 10.0,
            "exponent": 1.0,
        },
    )
    reactions = (
        ReactionData(
            "stick",
            {"A": 1.0, "site": 1.0},
            {"Ads": 1.0},
            "stick",
            None,
            gas_heating_eV=0.25,
            surfaces=("wall",),
        ),
        ReactionData(
            "etch",
            {"ion": 1.0, "Ads": 1.0},
            {"A": 1.0, "site": 1.0},
            "ion",
            None,
            gas_heating_eV=2.0,
            surfaces=("wall",),
        ),
    )
    chemistry = _chemistry(
        species=_species(),
        reactions=reactions,
        rate_models=(stick_model, ion_model),
    )
    surface = SurfaceGeometry("wall", "z", 2.0, 1.0e19, 300.0, {"Ads": 0.25})
    model = _model(chemistry, (surface,), zone_volumes_m3=np.array([4.0]))

    result = model.evaluate(
        model.initial_state(),
        np.array([[1.0e20, 1.0e15]]),
        np.array([400.0]),
        ion_flux_m2_s={("wall", "ion"): 10.0},
        ion_energy_eV={"wall": 6.0},
    )

    thermal_speed = math.sqrt(
        8.0 * BOLTZMANN_J_K * 400.0 / (math.pi * 40.0 * AMU_TO_KG)
    )
    sticking_rate = 0.25 * 0.2 * thermal_speed * 1.0e20 * 0.75
    ion_rate = 10.0 * 2.0 * ((6.0 - 2.0) / (10.0 - 2.0)) * 0.25
    assert result.rates_m2_s["stick@wall"] == pytest.approx(sticking_rate)
    assert result.rates_m2_s["etch@wall"] == pytest.approx(ion_rate)
    np.testing.assert_allclose(
        result.gas_derivative_m3_s,
        [[0.5 * (-sticking_rate + ion_rate), 0.0]],
    )
    np.testing.assert_allclose(
        result.coverage_derivative_s_inv,
        [(sticking_rate - ion_rate) / 1.0e19],
    )
    assert result.gas_heating_J_m3_s == pytest.approx(
        [0.5 * (0.25 * sticking_rate + 2.0 * ion_rate) * E_CHARGE]
    )
    assert result.gas_derivative_m3_s.dtype == np.dtype(float)
    assert result.coverage_derivative_s_inv.shape == (1,)


@pytest.mark.parametrize(
    ("kind", "reactants", "products"),
    [
        ("sticking", {"A": 2.0, "site": 2.0}, {"Ads": 2.0}),
        (
            "sticking",
            {"A": 1.0, "ion": 1.0, "site": 1.0},
            {"Ads": 1.0, "ion": 1.0},
        ),
        (
            "ion_assisted",
            {"ion": 1.0, "Ads": 1.0, "A": 1.0},
            {"ion": 1.0, "A": 2.0, "site": 1.0},
        ),
        (
            "ion_assisted",
            {"ion": 2.0, "Ads": 1.0},
            {"ion": 2.0, "A": 1.0, "site": 1.0},
        ),
    ],
)
def test_flux_driven_surface_rates_reject_unsupported_gas_reactants(
    kind: str,
    reactants: dict[str, float],
    products: dict[str, float],
) -> None:
    parameters = (
        {"value": 0.2}
        if kind == "sticking"
        else {
            "yield": 1.0,
            "threshold_eV": 0.0,
            "reference_energy_eV": 10.0,
            "exponent": 1.0,
        }
    )
    rate_model = RateModelData("rate", kind, parameters)
    reaction = ReactionData(
        "unsupported",
        reactants,
        products,
        rate_model.id,
        None,
    )
    chemistry = _chemistry(
        species=_species(), reactions=(reaction,), rate_models=(rate_model,)
    )
    surface = SurfaceGeometry("wall", "z", 1.0, 1.0e19, 300.0, {"Ads": 0.2})

    with pytest.raises(
        CaseValidationError,
        match="must have exactly one unit",
    ):
        _model(chemistry, (surface,))


def test_sticking_rate_is_independent_of_reactant_order() -> None:
    rate_model = RateModelData("stick", "sticking", {"value": 0.2})
    rates: list[float] = []
    for reactants in ({"A": 1.0, "site": 1.0}, {"site": 1.0, "A": 1.0}):
        reaction = ReactionData("stick", reactants, {"Ads": 1.0}, rate_model.id, None)
        chemistry = _chemistry(
            species=_species(), reactions=(reaction,), rate_models=(rate_model,)
        )
        surface = SurfaceGeometry("wall", "z", 1.0, 1.0e19, 300.0, {"Ads": 0.2})
        model = _model(chemistry, (surface,))
        evaluated = model.evaluate(
            model.initial_state(),
            np.array([[1.0e20, 1.0e15]]),
            np.array([400.0]),
        )
        rates.append(evaluated.rates_m2_s["stick@wall"])

    assert rates[0] == pytest.approx(rates[1])


def test_ion_assisted_rate_is_independent_of_reactant_order() -> None:
    rate_model = RateModelData(
        "etch",
        "ion_assisted",
        {
            "yield": 2.0,
            "threshold_eV": 2.0,
            "reference_energy_eV": 10.0,
            "exponent": 1.0,
        },
    )
    rates: list[float] = []
    for reactants in ({"ion": 1.0, "Ads": 1.0}, {"Ads": 1.0, "ion": 1.0}):
        reaction = ReactionData(
            "etch",
            reactants,
            {"ion": 1.0, "A": 1.0, "site": 1.0},
            rate_model.id,
            None,
        )
        chemistry = _chemistry(
            species=_species(), reactions=(reaction,), rate_models=(rate_model,)
        )
        surface = SurfaceGeometry("wall", "z", 1.0, 1.0e19, 300.0, {"Ads": 0.2})
        model = _model(chemistry, (surface,))
        evaluated = model.evaluate(
            model.initial_state(),
            np.array([[1.0e20, 1.0e15]]),
            np.array([400.0]),
            ion_flux_m2_s={("wall", "ion"): 10.0},
            ion_energy_eV={"wall": 6.0},
        )
        rates.append(evaluated.rates_m2_s["etch@wall"])

    assert rates[0] == pytest.approx(rates[1])


def test_surface_temperature_context_is_resolved_per_surface() -> None:
    rate_model = RateModelData(
        "desorb",
        "desorption",
        {"frequency_s_inv": 3.0, "activation_eV": 0.05},
    )
    reaction = ReactionData(
        "desorb",
        {"Ads": 1.0},
        {"A": 1.0, "site": 1.0},
        "desorb",
        None,
    )
    chemistry = _chemistry(
        species=_species(), reactions=(reaction,), rate_models=(rate_model,)
    )
    surfaces = (
        SurfaceGeometry("cold", "z", 1.0, 1.0e19, 300.0, {"Ads": 0.4}),
        SurfaceGeometry("hot", "z", 1.0, 1.0e19, 600.0, {"Ads": 0.4}),
    )
    model = _model(chemistry, surfaces)
    state = model.initial_state()
    gas = np.array([[0.0, 0.0]])
    gas_temperature = np.array([300.0])

    defaults = model.evaluate(state, gas, gas_temperature)
    overridden = model.evaluate(
        state,
        gas,
        gas_temperature,
        surface_temperature_K={"cold": 600.0},
    )

    activation_factor = math.exp(-0.05 * E_CHARGE / (BOLTZMANN_J_K * 600.0))
    expected_hot_rate = 3.0 * 1.0e19 * activation_factor * 0.4
    assert defaults.rates_m2_s["desorb@cold"] < defaults.rates_m2_s["desorb@hot"]
    assert overridden.rates_m2_s["desorb@cold"] == pytest.approx(expected_hot_rate)
    assert overridden.rates_m2_s["desorb@hot"] == pytest.approx(expected_hot_rate)
    np.testing.assert_allclose(
        overridden.coverage_derivative_s_inv,
        [-expected_hot_rate / 1.0e19, -expected_hot_rate / 1.0e19],
    )


def test_thermal_surface_rates_scale_with_their_physical_site_order() -> None:
    desorption = RateModelData(
        "desorb",
        "desorption",
        {"frequency_s_inv": 3.0, "activation_eV": 0.0},
    )
    langmuir_hinshelwood = RateModelData(
        "recombine",
        "langmuir_hinshelwood",
        {"A_m2_s_inv": 5.0, "activation_eV": 0.0},
    )
    reactions = (
        ReactionData(
            "desorb",
            {"Ads": 1.0},
            {"A": 1.0, "site": 1.0},
            desorption.id,
            None,
        ),
        ReactionData(
            "recombine",
            {"Ads": 2.0},
            {"A": 2.0, "site": 2.0},
            langmuir_hinshelwood.id,
            None,
        ),
    )
    chemistry = _chemistry(
        species=_species(),
        reactions=reactions,
        rate_models=(desorption, langmuir_hinshelwood),
    )
    low_density = 2.0e18
    high_density = 2.0 * low_density
    coverage = 0.25
    surfaces = (
        SurfaceGeometry("low", "z", 1.0, low_density, 300.0, {"Ads": coverage}),
        SurfaceGeometry("high", "z", 1.0, high_density, 300.0, {"Ads": coverage}),
    )

    evaluated = _model(chemistry, surfaces).evaluate(
        np.array([coverage, coverage]),
        np.zeros((1, 2)),
        np.array([300.0]),
    )

    assert evaluated.rates_m2_s["desorb@low"] == pytest.approx(
        3.0 * low_density * coverage
    )
    assert evaluated.rates_m2_s["desorb@high"] == pytest.approx(
        2.0 * evaluated.rates_m2_s["desorb@low"]
    )
    assert evaluated.rates_m2_s["recombine@low"] == pytest.approx(
        5.0 * low_density**2 * coverage**2
    )
    assert evaluated.rates_m2_s["recombine@high"] == pytest.approx(
        4.0 * evaluated.rates_m2_s["recombine@low"]
    )


@pytest.mark.parametrize(
    ("kind", "reactants"),
    [
        ("desorption", {"A": 1.0, "Ads": 1.0}),
        ("desorption", {"Ads": 2.0}),
        ("desorption", {"site": 1.0}),
        ("langmuir_hinshelwood", {"A": 1.0, "Ads": 2.0}),
        ("langmuir_hinshelwood", {"Ads": 1.0}),
        ("langmuir_hinshelwood", {"Ads": 3.0}),
        ("langmuir_hinshelwood", {"Ads": 1.0, "site": 1.0}),
    ],
)
def test_thermal_surface_rates_reject_unsupported_reactant_shapes(
    kind: str,
    reactants: dict[str, float],
) -> None:
    parameters = (
        {"frequency_s_inv": 1.0, "activation_eV": 0.0}
        if kind == "desorption"
        else {"A_m2_s_inv": 1.0, "activation_eV": 0.0}
    )
    rate_model = RateModelData("thermal", kind, parameters)
    reaction = ReactionData(
        "unsupported",
        reactants,
        {"A": 1.0, "site": 1.0},
        rate_model.id,
        None,
    )
    chemistry = _chemistry(
        species=_species(), reactions=(reaction,), rate_models=(rate_model,)
    )
    surface = SurfaceGeometry("wall", "z", 1.0, 1.0e19, 300.0, {"Ads": 0.2})

    with pytest.raises(CaseValidationError, match="surface reactant"):
        _model(chemistry, (surface,))


def test_evaluation_rejects_shape_and_domain_contract_violations() -> None:
    chemistry = _chemistry(species=_species())
    surface = SurfaceGeometry("wall", "z", 1.0, 1.0e19, 300.0, {"Ads": 0.2})
    model = _model(chemistry, (surface,))
    state = model.initial_state()
    gas = np.array([[1.0, 1.0]])
    gas_temperature = np.array([300.0])

    with pytest.raises(ValueError, match="coverage state has the wrong shape"):
        model.evaluate(np.zeros(2), gas, gas_temperature)
    with pytest.raises(ValueError, match="gas arrays have the wrong shape"):
        model.evaluate(state, np.ones((1, 1)), gas_temperature)
    with pytest.raises(ValueError, match="gas arrays have the wrong shape"):
        model.evaluate(state, gas, np.ones(2))
    with pytest.raises(ValueError, match="domain_atol has the wrong shape"):
        model.evaluate(state, gas, gas_temperature, domain_atol=np.ones(2))
    with pytest.raises(ValueError, match="domain_atol has the wrong shape"):
        model.evaluate(state, gas, gas_temperature, domain_atol=np.array([-1.0]))

    overfilled = state.copy()
    overfilled[0] = 1.1
    with pytest.raises(ModelDomainError, match="negative algebraic free-site"):
        model.evaluate(overfilled, gas, gas_temperature)
    with pytest.raises(ModelDomainError, match="temperature must be finite"):
        model.evaluate(
            state,
            gas,
            gas_temperature,
            surface_temperature_K={"wall": float("nan")},
        )
