from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from plasma_global.chemistry.data import (
    ChemistryData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, SolverSettings, Zone
from plasma_global.core.exceptions import ModelConfigurationError, StateDomainError
from plasma_global.core.solver import solve_compiled_model
from plasma_global.core.transport import CompiledTransport, SegmentTransport
from plasma_global.errors import ModelDomainError
from plasma_global.models.electrons import ElectronEnergyClosure, ElectronState
from plasma_global.models.gas_energy import BOLTZMANN_J_K, HeavyEnergyClosure
from plasma_global.models.power import (
    CompiledPowerCommand,
    PowerCoordinator,
    PrescribedPowerPort,
)
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry
from plasma_global.models.walls import (
    BoundaryReaction,
    CompiledWallBoundary,
    WallBoundary,
    WallEvaluation,
    compile_wall_boundary,
    evaluate_compiled_wall_boundary,
)

E_CHARGE = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31


def test_compiled_wall_boundary_keeps_legacy_positional_constructor() -> None:
    compiled = CompiledWallBoundary(
        WallBoundary("z", 1.0),
        np.array([1]),
        np.array([1.0]),
        np.array([6.6e-26]),
        ((),),
        True,
    )

    assert compiled.neutral_indices.size == 0


def _evaluate_compiled_wall(
    *,
    boundary: WallBoundary,
    volume_m3: float,
    species_ids: tuple[str, ...],
    charges: np.ndarray,
    masses_kg: np.ndarray,
    densities_m3: np.ndarray,
    electrons: ElectronState,
) -> WallEvaluation:
    compiled = compile_wall_boundary(
        boundary=boundary,
        species_ids=species_ids,
        charges=charges,
        masses_kg=masses_kg,
    )
    return evaluate_compiled_wall_boundary(
        compiled=compiled,
        volume_m3=volume_m3,
        species_ids=species_ids,
        densities_m3=densities_m3,
        electrons=electrons,
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
    reaction_zones: tuple[tuple[str, ...], ...]
    gas_heating_eV: np.ndarray

    @property
    def jacobian_species_pattern(self) -> np.ndarray:
        return (
            (self.stoichiometry != 0.0).T.astype(np.int8)
            @ (self.reactant_orders != 0.0).astype(np.int8)
        ) > 0


def inert_chemistry() -> ChemistryFixture:
    return ChemistryFixture(
        species_ids=("A", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26]),
        reaction_ids=(),
        stoichiometry=np.zeros((0, 2)),
        reactant_orders=np.zeros((0, 2)),
        electron_orders=np.zeros(0),
        rate_evaluators=(),
        energy_loss_eV=np.zeros(0),
        reaction_zones=(),
        gas_heating_eV=np.zeros(0),
    )


def initial_state(
    *, gas_temperature_K: float | None = None, ads_coverage: float | None = None
) -> InitialState:
    return InitialState(
        densities_m3_by_zone={"z": {"A": 1.0e20, "ion": 1.0e15}},
        mean_energy_eV_by_zone={"z": 3.0},
        gas_temperature_K_by_zone=(
            {} if gas_temperature_K is None else {"z": gas_temperature_K}
        ),
        surface_coverages=(
            {} if ads_coverage is None else {"wall": {"Ads": ads_coverage}}
        ),
    )


def test_evolved_gas_internal_energy_matches_wall_relaxation_analytic() -> None:
    closure = HeavyEnergyClosure(
        cv_over_kb=np.array([1.5, 1.5]),
        wall_temperature_K=np.array([300.0]),
        wall_relaxation_s_inv=np.array([2.0]),
    )
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0, 300.0),),
        segments=(RecipeSegment("cool", 0.0, 0.5),),
        electron_closure=ElectronEnergyClosure(),
        heavy_energy_closure=closure,
    )
    initial = initial_state(gas_temperature_K=600.0)
    state0 = model.initial_state(initial)

    assert model.layout.labels == (
        "n[z,A]",
        "n[z,ion]",
        "electron_energy[z]",
        "gas_internal_energy[z]",
    )
    evaluated = model.evaluate(0.0, state0, model.segments[0])
    initial_energy = state0[model.layout.heavy_energy_indices["z"]]
    equilibrium = closure.energy_J_m3(np.array([[1.0e20, 1.0e15]]), np.array([300.0]))[
        0
    ]
    assert evaluated.derivative[
        model.layout.heavy_energy_indices["z"]
    ] == pytest.approx(-2.0 * (initial_energy - equilibrium))

    result = solve_compiled_model(
        model,
        initial,
        SolverSettings(
            rtol=1.0e-10,
            atol=1.0e-14,
            save_at_s=(0.0, 0.5),
        ),
    )
    expected_energy = equilibrium + (initial_energy - equilibrium) * np.exp(-1.0)
    assert result.final_value("gas_internal_energy[z]") == pytest.approx(
        expected_energy, rel=2.0e-8
    )


def test_evolved_gas_rejects_mismatched_transport_heat_capacities() -> None:
    closure = HeavyEnergyClosure(
        cv_over_kb=np.array([1.5, 1.5]),
        wall_temperature_K=np.array([300.0]),
        wall_relaxation_s_inv=np.zeros(1),
    )
    transport = CompiledTransport(
        volumes_m3=np.array([1.0]),
        pump_frequency_s_inv=np.zeros(1),
        edge_from=np.array([], dtype=int),
        edge_to=np.array([], dtype=int),
        edge_conductance_m3_s=np.array([]),
        n_species=2,
        heavy_cv_over_kb=np.array([2.5, 2.5]),
    )

    with pytest.raises(ModelConfigurationError, match="heat capacities to match"):
        CompiledGlobalModel(
            chemistry=inert_chemistry(),
            zones=(Zone("z", 1.0),),
            segments=(RecipeSegment("flow", 0.0, 1.0),),
            electron_closure=ElectronEnergyClosure(),
            transport=transport,
            heavy_energy_closure=closure,
        )


def test_heavy_energy_ledger_composes_power_reaction_flow_and_elastic_terms() -> None:
    chemistry = ChemistryFixture(
        species_ids=("A", "B", "ion"),
        charges=np.array([0.0, 0.0, 1.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26, 6.0e-26]),
        reaction_ids=("heat",),
        stoichiometry=np.array([[-1.0, 1.0, 0.0]]),
        reactant_orders=np.array([[1.0, 0.0, 0.0]]),
        electron_orders=np.array([0.0]),
        rate_evaluators=(2.0,),
        energy_loss_eV=np.array([0.0]),
        gas_heating_eV=np.array([0.5]),
        reaction_zones=((),),
    )
    closure = HeavyEnergyClosure(
        cv_over_kb=np.array([1.5, 1.5, 1.5]),
        wall_temperature_K=np.array([300.0]),
        wall_relaxation_s_inv=np.array([0.0]),
    )
    forcing = SegmentTransport(
        particle_source_m3_s=np.zeros((1, 3)),
        inlet_heavy_energy_J_m3_s=np.array([4.0]),
    )
    segment = RecipeSegment(
        "balance",
        0.0,
        1.0,
        transport=forcing,
        port_commands={"heater": CompiledPowerCommand(kind="power", power_W=5.0)},
    )
    coordinator = PowerCoordinator(
        ports=(
            PrescribedPowerPort("heater", "z", electron_fraction=0.0, gas_fraction=1.0),
        ),
        zone_ids=("z",),
    )
    model = CompiledGlobalModel(
        chemistry=chemistry,
        zones=(Zone("z", 1.0),),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
        transport=CompiledTransport(
            volumes_m3=np.array([1.0]),
            pump_frequency_s_inv=np.array([0.0]),
            edge_from=np.zeros(0, dtype=int),
            edge_to=np.zeros(0, dtype=int),
            edge_conductance_m3_s=np.zeros(0),
            n_species=3,
            heavy_cv_over_kb=closure.cv_over_kb,
        ),
        power_coordinator=coordinator,
        heavy_energy_closure=closure,
        elastic_heating_evaluator=(
            lambda _zone, _electrons, _density, _temperature, _kinetics: 3.0
        ),
    )
    initial = InitialState(
        densities_m3_by_zone={"z": {"A": 10.0, "B": 0.0, "ion": 1.0}},
        mean_energy_eV_by_zone={"z": 3.0},
        gas_temperature_K_by_zone={"z": 300.0},
    )
    state = model.initial_state(initial)
    evaluated = model.evaluate(0.0, state, segment)
    ledger = evaluated.ledger_by_zone["z"]
    reaction_heating = 0.5 * E_CHARGE * 20.0
    gas_index = model.layout.heavy_energy_indices["z"]
    electron_index = model.layout.electron_energy_indices["z"]

    assert ledger.gas_power_J_m3_s == 5.0
    assert ledger.gas_reaction_heating_J_m3_s == pytest.approx(reaction_heating)
    assert ledger.transport_heavy_energy_J_m3_s == 4.0
    assert ledger.elastic_heating_J_m3_s == 3.0
    assert evaluated.derivative[gas_index] == pytest.approx(
        5.0 + reaction_heating + 4.0 + 3.0
    )
    assert evaluated.derivative[electron_index] == pytest.approx(-3.0)


def surface_chemistry(
    *, ion_assisted: bool, gas_heating_eV: float = 0.25
) -> ChemistryData:
    species = (
        SpeciesData("A", "gas", 0, 36.0, {"X": 1.0}),
        SpeciesData("ion", "gas", 1, 36.0, {"X": 1.0}),
        SpeciesData(
            "site",
            "surface",
            0,
            1.0,
            {"site": 1.0},
            state_tags=frozenset({"site"}),
            surfaces=("wall",),
        ),
        SpeciesData(
            "Ads",
            "surface",
            0,
            36.0,
            {"X": 1.0, "site": 1.0},
            surfaces=("wall",),
        ),
    )
    if ion_assisted:
        reaction = ReactionData(
            id="etch",
            reactants={"ion": 1.0, "Ads": 1.0},
            products={"A": 1.0, "site": 1.0},
            rate_model="ion",
            energy_loss_eV=None,
            surfaces=("wall",),
        )
        model = RateModelData(
            "ion",
            "ion_assisted",
            {
                "yield": 1.0,
                "threshold_eV": 0.0,
                "reference_energy_eV": 10.0,
                "exponent": 1.0,
            },
        )
    else:
        reaction = ReactionData(
            id="stick",
            reactants={"A": 1.0, "site": 1.0},
            products={"Ads": 1.0},
            rate_model="stick",
            energy_loss_eV=None,
            gas_heating_eV=gas_heating_eV,
            surfaces=("wall",),
        )
        model = RateModelData("stick", "sticking", {"value": 0.2})
    return ChemistryData(
        source=Path("surface.yaml"),
        species=species,
        gas_reactions=(),
        boundary_reactions=(),
        surface_reactions=(reaction,),
        rate_models={model.id: model},
        cross_sections={},
    )


def compiled_surface(
    *,
    ion_assisted: bool,
    initial_coverage: float,
    gas_heating_eV: float = 0.25,
) -> CompiledSurfaceModel:
    return CompiledSurfaceModel(
        chemistry=surface_chemistry(
            ion_assisted=ion_assisted, gas_heating_eV=gas_heating_eV
        ),
        gas_species_ids=("A", "ion"),
        gas_masses_kg=np.array([6.0e-26, 6.0e-26]),
        zone_ids=("z",),
        zone_volumes_m3=np.array([1.0]),
        surfaces=(
            SurfaceGeometry(
                "wall",
                "z",
                area_m2=2.0,
                site_density_m2=1.0e19,
                temperature_K=300.0,
                initial_coverages={"Ads": initial_coverage},
            ),
        ),
    )


def test_wall_and_surface_particle_fluxes_carry_species_internal_energy() -> None:
    gas_temperature_K = 400.0
    closure = HeavyEnergyClosure(
        cv_over_kb=np.array([2.5, 1.5]),
        wall_temperature_K=np.array([gas_temperature_K]),
        wall_relaxation_s_inv=np.zeros(1),
    )
    surface = compiled_surface(
        ion_assisted=False,
        initial_coverage=0.2,
        gas_heating_eV=0.0,
    )
    wall = WallBoundary(
        zone_id="z",
        area_m2=2.0,
        surface_id="wall",
        transport_kind="prescribed_frequency",
        prescribed_frequency_s_inv=0.5,
        reactions=(BoundaryReaction("neutralize", "ion", products={"A": 1.0}),),
    )
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0, gas_temperature_K),),
        segments=(RecipeSegment("particle-energy", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        heavy_energy_closure=closure,
        wall_boundaries=(wall,),
        surface_model=surface,
    )
    state = model.initial_state(
        initial_state(gas_temperature_K=gas_temperature_K, ads_coverage=0.25)
    )

    evaluated = model.evaluate(0.0, state, model.segments[0])
    ledger = evaluated.ledger_by_zone["z"]
    density_slice = model.layout.density_slices["z"]
    energy_index = model.layout.heavy_energy_indices["z"]
    incident_rate = ledger.wall_fluxes[0].incident_rate_m3_s
    surface_rate = ledger.surface_rates_m2_s["stick@wall"]
    expected_wall_energy = (
        BOLTZMANN_J_K
        * gas_temperature_K
        * (closure.cv_over_kb[0] - closure.cv_over_kb[1])
        * incident_rate
    )
    expected_surface_energy = (
        -BOLTZMANN_J_K * gas_temperature_K * closure.cv_over_kb[0] * 2.0 * surface_rate
    )

    assert ledger.wall_species_energy_J_m3_s == pytest.approx(expected_wall_energy)
    assert ledger.surface_species_energy_J_m3_s == pytest.approx(
        expected_surface_energy
    )
    assert evaluated.derivative[energy_index] == pytest.approx(
        expected_wall_energy + expected_surface_energy
    )

    # With no explicit heating, advancing density and its carried internal energy
    # by the same flux must leave the algebraic closure temperature unchanged.
    dt_s = 1.0e-8
    next_density = state[density_slice] + dt_s * evaluated.derivative[density_slice]
    next_energy = state[energy_index] + dt_s * evaluated.derivative[energy_index]
    next_temperature = closure.temperature_K(
        next_density.reshape(1, -1), np.array([next_energy])
    )[0]
    assert next_temperature == pytest.approx(gas_temperature_K)


def test_surface_sticking_adds_independent_coverage_and_conserves_event_rate() -> None:
    surface = compiled_surface(ion_assisted=False, initial_coverage=0.2)
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0, 400.0),),
        segments=(RecipeSegment("surface", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        surface_model=surface,
    )
    state = model.initial_state(initial_state(ads_coverage=0.25))
    original = state.copy()
    evaluated = model.evaluate(0.0, state, model.segments[0])

    coverage_index = model.layout.surface_coverage_indices[("wall", "Ads")]
    rate = evaluated.ledger_by_zone["z"].surface_rates_m2_s["stick@wall"]
    assert state[coverage_index] == 0.25
    assert np.array_equal(state, original)
    assert evaluated.derivative[
        model.layout.density_slices["z"].start
    ] == pytest.approx(-2.0 * rate)
    assert evaluated.derivative[coverage_index] == pytest.approx(rate / 1.0e19)
    assert evaluated.ledger_by_zone[
        "z"
    ].surface_reaction_heating_J_m3_s == pytest.approx(2.0 * rate * 0.25 * E_CHARGE)


def test_wall_flux_is_shared_with_ion_assisted_surface_without_double_ion_loss() -> (
    None
):
    surface = compiled_surface(ion_assisted=True, initial_coverage=0.5)
    wall = WallBoundary(
        zone_id="z",
        area_m2=2.0,
        sheath_energy_eV=10.0,
        surface_id="wall",
        transport_kind="prescribed_frequency",
        prescribed_frequency_s_inv=0.5,
        reactions=(BoundaryReaction("neutralize", "ion", products={"A": 1.0}),),
    )
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("etch", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        wall_boundaries=(wall,),
        surface_model=surface,
    )
    state = model.initial_state(initial_state())
    evaluated = model.evaluate(0.0, state, model.segments[0])
    ledger = evaluated.ledger_by_zone["z"]
    record = ledger.wall_fluxes[0]
    event_rate_m2_s = ledger.surface_rates_m2_s["etch@wall"]
    ion_index = model.layout.density_slices["z"].start + 1
    neutral_index = model.layout.density_slices["z"].start
    incident_rate_m3_s = 1.0e15 * 0.5

    assert record.surface_id == "wall"
    assert record.incident_flux_m2_s == pytest.approx(incident_rate_m3_s / 2.0)
    assert event_rate_m2_s == pytest.approx(record.incident_flux_m2_s * 0.5)
    assert evaluated.derivative[ion_index] == pytest.approx(-incident_rate_m3_s)
    assert evaluated.derivative[neutral_index] == pytest.approx(
        incident_rate_m3_s + 2.0 * event_rate_m2_s
    )


@pytest.mark.parametrize(
    ("boundary", "expected_frequency"),
    [
        (
            WallBoundary(
                "z",
                2.0,
                transport_kind="prescribed_frequency",
                prescribed_frequency_s_inv=3.0,
            ),
            3.0,
        ),
        (
            WallBoundary(
                "z",
                2.0,
                transport_kind="ambipolar",
                diffusion_coefficient_m2_s=0.8,
                diffusion_length_m=0.4,
            ),
            5.0,
        ),
    ],
)
def test_frequency_wall_transports_have_explicit_volume_loss(
    boundary: WallBoundary, expected_frequency: float
) -> None:
    wall = _evaluate_compiled_wall(
        boundary=boundary,
        volume_m3=4.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26]),
        densities_m3=np.array([1.0e20, 2.0e15]),
        electrons=ElectronState(2.0e15, 3.0, 2.0, 1.0e-3, None),
    )

    assert wall.records[0].incident_rate_m3_s == pytest.approx(
        2.0e15 * expected_frequency
    )


def test_off_wall_transport_has_no_flux_or_loss() -> None:
    wall = _evaluate_compiled_wall(
        boundary=WallBoundary("z", 2.0, transport_kind="off"),
        volume_m3=1.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26]),
        densities_m3=np.array([1.0e20, 2.0e15]),
        electrons=ElectronState(2.0e15, 3.0, 2.0, 1.0e-3, None),
    )
    assert not wall.records
    assert np.all(wall.species_derivative_m3_s == 0.0)


def test_single_positive_ion_bohm_wall_computes_floating_sheath_energy() -> None:
    ion_mass_kg = 6.6e-26
    electron_temperature_eV = 2.0
    wall = _evaluate_compiled_wall(
        boundary=WallBoundary("z", 2.0, bohm_factor=0.61),
        volume_m3=1.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([ion_mass_kg, ion_mass_kg]),
        densities_m3=np.array([1.0e20, 2.0e15]),
        electrons=ElectronState(
            2.0e15,
            3.0,
            electron_temperature_eV,
            1.0e-3,
            None,
        ),
    )
    expected_sheath_eV = (
        0.5
        * electron_temperature_eV
        * np.log(ion_mass_kg / (2.0 * np.pi * ELECTRON_MASS_KG))
    )
    expected_loss = (
        wall.records[0].incident_rate_m3_s
        * (2.0 * electron_temperature_eV + expected_sheath_eV)
        * E_CHARGE
    )

    assert wall.sheath_energy_eV == pytest.approx(expected_sheath_eV)
    assert wall.electron_energy_loss_J_m3_s == pytest.approx(expected_loss)


def test_explicit_or_nonstandard_wall_energy_keeps_legacy_value() -> None:
    electrons = ElectronState(2.0e15, 3.0, 2.0, 1.0e-3, None)
    explicit = _evaluate_compiled_wall(
        boundary=WallBoundary("z", 2.0, sheath_energy_eV=7.5),
        volume_m3=1.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([6.6e-26, 6.6e-26]),
        densities_m3=np.array([1.0e20, 2.0e15]),
        electrons=electrons,
    )
    multiply_charged = _evaluate_compiled_wall(
        boundary=WallBoundary("z", 2.0),
        volume_m3=1.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 2.0]),
        masses_kg=np.array([6.6e-26, 6.6e-26]),
        densities_m3=np.array([1.0e20, 2.0e15]),
        electrons=electrons,
    )

    assert explicit.sheath_energy_eV == 7.5
    assert multiply_charged.sheath_energy_eV == 0.0


def test_wall_uses_nonnegative_ion_view_and_charge_weighted_bohm_speed() -> None:
    boundary = WallBoundary("z", 2.0, bohm_factor=0.61)
    electron_state = ElectronState(2.0e15, 3.0, 2.0, 1.0e-3, None)
    negative = _evaluate_compiled_wall(
        boundary=boundary,
        volume_m3=1.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 2.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26]),
        densities_m3=np.array([1.0e20, -1.0e-6]),
        electrons=electron_state,
    )
    positive = _evaluate_compiled_wall(
        boundary=boundary,
        volume_m3=1.0,
        species_ids=("A", "ion"),
        charges=np.array([0.0, 2.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26]),
        densities_m3=np.array([1.0e20, 2.0e15]),
        electrons=electron_state,
    )

    assert negative.records[0].incident_flux_m2_s == 0.0
    assert negative.records[0].incident_rate_m3_s == 0.0
    expected_speed = 0.61 * np.sqrt(2.0 * E_CHARGE * 2.0 / 6.0e-26)
    assert positive.records[0].bohm_speed_m_s == pytest.approx(expected_speed)


def test_compiled_wall_receives_roundoff_clipped_reaction_density() -> None:
    wall = WallBoundary(
        "z",
        2.0,
        reactions=(BoundaryReaction("neutralize", "ion", {"A": 1.0}),),
    )
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        electron_density_provider=lambda _time, _zone: 1.0e15,
        wall_boundaries=(wall,),
        domain_atol=1.0,
    )
    state = model.initial_state(initial_state())
    density_slice = model.layout.density_slices["z"]
    state[density_slice.start + 1] = -5.0

    evaluated = model.evaluate(0.0, state, model.segments[0])

    np.testing.assert_array_equal(evaluated.derivative[density_slice], [0.0, 0.0])
    assert evaluated.ledger_by_zone["z"].wall_energy_loss_J_m3_s == 0.0


def test_coverage_domain_uses_component_tolerance_without_projection() -> None:
    surface = compiled_surface(ion_assisted=False, initial_coverage=0.0)
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("surface", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        surface_model=surface,
        domain_atol=1.0e-3,
    )
    state = model.initial_state(initial_state())
    coverage_index = model.layout.surface_coverage_indices[("wall", "Ads")]
    state[coverage_index] = -5.0e-3
    original = state.copy()

    model.evaluate(0.0, state, model.segments[0])
    assert np.array_equal(state, original)
    state[coverage_index] = -1.01e-2
    with pytest.raises(ModelDomainError, match="nonnegative domain"):
        model.evaluate(0.0, state, model.segments[0])


def test_prescribed_electron_density_reports_charge_residual() -> None:
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("profile", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        electron_density_provider=lambda _time_s, _zone_id: 4.0e14,
    )
    state = model.initial_state(initial_state())
    evaluated = model.evaluate(0.0, state, model.segments[0])

    assert evaluated.electron_states["z"].density_m3 == 4.0e14
    assert evaluated.charge_residual_m3_by_zone["z"] == pytest.approx(6.0e14)

    invalid = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("profile", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        electron_density_provider=lambda _time_s, _zone_id: -1.0,
    )
    with pytest.raises(StateDomainError, match="Prescribed electron density"):
        invalid.initial_state(initial_state())


def test_fixed_gas_rejects_an_initial_gas_energy_state() -> None:
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("fixed", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
    )
    with pytest.raises(ModelConfigurationError, match="Fixed gas energy"):
        model.initial_state(initial_state(gas_temperature_K=300.0))
