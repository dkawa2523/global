from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import InitialState, RecipeSegment, SolverSettings, Zone
from plasma_global.core.exceptions import StateDomainError
from plasma_global.core.solver import _accepted_state_for_result, solve_compiled_model
from plasma_global.core.transport import CompiledTransport, SegmentTransport
from plasma_global.errors import CouplingConvergenceError, IntegrationError
from plasma_global.models.electrons import (
    ElectronEnergyClosure,
    LocalFieldClosure,
)
from plasma_global.models.external_table import (
    ExternalTableBinding,
    load_external_table,
)
from plasma_global.models.kinetics import (
    TabulatedElectronKinetics,
)
from plasma_global.models.power import (
    CompiledPowerCommand,
    ExternalTablePowerPort,
    PowerCoordinator,
    PowerPortResult,
    PowerState,
    PrescribedPowerPort,
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


def inert_chemistry() -> ChemistryFixture:
    return ChemistryFixture(
        species_ids=("tracer", "ion"),
        charges=np.array([0.0, 1.0]),
        masses_kg=np.array([6.0e-26, 6.0e-26]),
        reaction_ids=(),
        stoichiometry=np.zeros((0, 2)),
        reactant_orders=np.zeros((0, 2)),
        electron_orders=np.zeros(0),
        rate_evaluators=(),
        energy_loss_eV=np.zeros(0),
        gas_heating_eV=np.zeros(0),
        reaction_zones=(),
    )


def reactive_chemistry(rate_evaluator: object) -> ChemistryFixture:
    return ChemistryFixture(
        species_ids=("A", "B", "ion"),
        charges=np.array([0.0, 0.0, 1.0]),
        masses_kg=np.array([2.0e-26, 2.0e-26, 6.0e-26]),
        reaction_ids=("convert",),
        stoichiometry=np.array([[-1.0, 1.0, 0.0]]),
        reactant_orders=np.array([[1.0, 0.0, 0.0]]),
        electron_orders=np.array([0.0]),
        rate_evaluators=(rate_evaluator,),
        energy_loss_eV=np.array([0.0]),
        gas_heating_eV=np.zeros(1),
        reaction_zones=((),),
    )


def test_multizone_edge_solution_and_volume_integrals() -> None:
    volumes = np.array([1.0, 2.0])
    conductance = 0.4
    transport = CompiledTransport(
        volumes_m3=volumes,
        pump_frequency_s_inv=np.zeros(2),
        edge_from=np.array([0]),
        edge_to=np.array([1]),
        edge_conductance_m3_s=np.array([conductance]),
        n_species=2,
    )
    segment = RecipeSegment(
        "transfer",
        0.0,
        0.5,
        transport=SegmentTransport.zeros(2, 2),
    )
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("source", 1.0), Zone("receiver", 2.0)),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
        transport=transport,
    )
    initial = InitialState(
        densities_m3_by_zone={
            "source": {"tracer": 10.0, "ion": 2.0},
            "receiver": {"tracer": 1.0, "ion": 1.0},
        },
        mean_energy_eV_by_zone={"source": 3.0, "receiver": 3.0},
    )
    y0 = model.initial_state(initial)
    result = solve_compiled_model(
        model,
        initial,
        SolverSettings(rtol=1.0e-10, atol=1.0e-13, save_at_s=(0.0, 0.25, 0.5)),
    )

    decay = np.exp(-conductance * segment.end_s / volumes[0])
    expected_source = 10.0 * decay
    expected_receiver = 1.0 + volumes[0] / volumes[1] * 10.0 * (1.0 - decay)
    assert result.final_value("n[source,tracer]") == pytest.approx(
        expected_source, rel=2.0e-8
    )
    assert result.final_value("n[receiver,tracer]") == pytest.approx(
        expected_receiver, rel=2.0e-8
    )
    initial_particles = volumes @ np.array([10.0, 1.0])
    final_particles = volumes @ np.array(
        [
            result.final_value("n[source,tracer]"),
            result.final_value("n[receiver,tracer]"),
        ]
    )
    assert final_particles == pytest.approx(initial_particles, rel=1.0e-12)
    initial_energy = volumes @ np.array(
        [
            y0[model.layout.electron_energy_indices["source"]],
            y0[model.layout.electron_energy_indices["receiver"]],
        ]
    )
    final_energy = volumes @ np.array(
        [
            result.final_value("electron_energy[source]"),
            result.final_value("electron_energy[receiver]"),
        ]
    )
    assert final_energy == pytest.approx(initial_energy, rel=1.0e-12)


def test_inlet_pump_and_electron_energy_transport_enter_core_ledger() -> None:
    transport = CompiledTransport(
        volumes_m3=np.array([2.0]),
        pump_frequency_s_inv=np.array([0.5]),
        edge_from=np.array([], dtype=int),
        edge_to=np.array([], dtype=int),
        edge_conductance_m3_s=np.array([]),
        n_species=2,
    )
    forcing = SegmentTransport(
        particle_source_m3_s=np.array([[3.0, 0.0]]),
        inlet_heavy_energy_J_m3_s=np.array([4.0]),
    )
    segment = RecipeSegment("flow", 0.0, 1.0, transport=forcing)
    model = CompiledGlobalModel(
        chemistry=inert_chemistry(),
        zones=(Zone("z", 2.0),),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
        transport=transport,
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"tracer": 2.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    evaluation = model.evaluate(0.0, state, segment)
    ledger = evaluation.ledger_by_zone["z"]

    assert evaluation.derivative[model.layout.density_slices["z"]][0] == pytest.approx(
        2.0
    )
    assert ledger.transport_species_source_m3_s["tracer"] == pytest.approx(2.0)
    assert ledger.transport_electron_energy_J_m3_s == pytest.approx(
        -0.5 * state[model.layout.electron_energy_indices["z"]]
    )
    assert ledger.inlet_heavy_energy_J_m3_s == 4.0


def _mean_energy_table(path: Path, *, lookup: str) -> TabulatedElectronKinetics:
    axis = np.array([2.0, 4.0]) if lookup == "mean_energy" else np.array([1.0, 2.0])
    return TabulatedElectronKinetics(
        source=path,
        lookup=lookup,
        bounds="error",
        axis=axis,
        mean_energy_eV=np.array([2.0, 4.0]),
        mobility_m2_V_s=np.array([0.5, 0.4]),
        effective_field_Td=np.array([1.0, 2.0]),
        rate_tables={"convert": np.array([1.0, 3.0])},
    )


def test_table_rates_and_mobility_feed_reactions_and_power_ports(
    tmp_path: Path,
) -> None:
    def table_rate(context: object) -> float:
        assert context.electron_mobility_m2_V_s == pytest.approx(0.45)
        return float(context.rate_coefficients["convert"])

    table = _mean_energy_table(tmp_path / "prepared.h5", lookup="mean_energy")
    coordinator = PowerCoordinator(
        ports=(
            PrescribedPowerPort(
                "source_a", "z", electron_fraction=0.75, gas_fraction=0.25
            ),
            PrescribedPowerPort("source_b", "z", electron_fraction=1.0),
        ),
        zone_ids=("z",),
    )
    segment = RecipeSegment(
        "powered",
        0.0,
        1.0,
        port_commands={
            "source_a": CompiledPowerCommand(kind="power", power_W=40.0),
            "source_b": CompiledPowerCommand(kind="power", power_W=10.0),
        },
    )
    model = CompiledGlobalModel(
        chemistry=reactive_chemistry(table_rate),
        zones=(Zone("z", 2.0),),
        segments=(segment,),
        electron_closure=ElectronEnergyClosure(),
        power_coordinator=coordinator,
        electron_kinetics_by_zone={"z": table},
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 5.0, "B": 0.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    evaluation = model.evaluate(0.0, state, segment)
    ledger = evaluation.ledger_by_zone["z"]

    assert ledger.reaction_rates_m3_s["convert"] == pytest.approx(10.0)
    assert ledger.absorbed_power_J_m3_s == pytest.approx(20.0)
    assert ledger.gas_power_J_m3_s == pytest.approx(5.0)
    assert evaluation.kinetics_by_zone["z"].mobility_m2_V_s == pytest.approx(0.45)


@dataclass(frozen=True)
class FixedFieldPort:
    time_dependent = False
    produces_reduced_field = True

    port_id: str
    zone_id: str
    field_Td: float

    def evaluate(
        self,
        _time_s: float,
        state: PowerState,
        _command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        mobility = float(state.electron_mobility_m2_V_s)
        return PowerPortResult(
            port_id=self.port_id,
            zone_id=self.zone_id,
            electron_power_W=mobility,
            reduced_field_Td=self.field_Td,
        )


def test_local_field_port_coupling_converges_and_drives_table_rates(
    tmp_path: Path,
) -> None:
    table = _mean_energy_table(tmp_path / "local.h5", lookup="local_field")
    coordinator = PowerCoordinator(
        ports=(FixedFieldPort("field", "z", 1.5),),
        zone_ids=("z",),
    )
    segment = RecipeSegment("field", 0.0, 1.0)
    model = CompiledGlobalModel(
        chemistry=reactive_chemistry(
            lambda context: float(context.rate_coefficients["convert"])
        ),
        zones=(Zone("z", 1.0),),
        segments=(segment,),
        electron_closure=LocalFieldClosure(table.mean_energy_from_field),
        power_coordinator=coordinator,
        electron_kinetics_by_zone={"z": table},
    )
    state = model.initial_state(
        InitialState(densities_m3_by_zone={"z": {"A": 5.0, "B": 0.0, "ion": 1.0}})
    )
    evaluation = model.evaluate(0.0, state, segment)

    assert evaluation.power_coupling is not None
    assert evaluation.power_coupling.iterations_by_zone["z"] == 3
    assert evaluation.electron_states["z"].mean_energy_eV == pytest.approx(3.0)
    assert evaluation.electron_states["z"].temperature_eV == pytest.approx(2.0)
    assert evaluation.ledger_by_zone["z"].reaction_rates_m3_s[
        "convert"
    ] == pytest.approx(10.0)


@dataclass(frozen=True)
class OscillatingFieldPort:
    time_dependent = False
    produces_reduced_field = True

    port_id: str = "oscillating"
    zone_id: str = "z"

    def evaluate(
        self,
        _time_s: float,
        state: PowerState,
        _command: CompiledPowerCommand | None,
    ) -> PowerPortResult:
        mobility = float(state.electron_mobility_m2_V_s)
        field = 2.0 if mobility < 1.5 else 1.0
        return PowerPortResult(self.port_id, self.zone_id, 0.0, reduced_field_Td=field)


def test_power_kinetics_coupling_fails_after_twelve_iterations(tmp_path: Path) -> None:
    table = TabulatedElectronKinetics(
        source=tmp_path / "oscillating.h5",
        lookup="local_field",
        bounds="error",
        axis=np.array([1.0, 2.0]),
        mean_energy_eV=np.array([1.0, 2.0]),
        mobility_m2_V_s=np.array([1.0, 2.0]),
        effective_field_Td=np.array([1.0, 2.0]),
        rate_tables={},
    )
    coordinator = PowerCoordinator((OscillatingFieldPort(),), ("z",))

    with pytest.raises(CouplingConvergenceError, match="12 iterations"):
        coordinator.evaluate(
            time_s=0.0,
            commands={},
            electron_density_m3_by_zone={"z": 1.0},
            neutral_density_m3_by_zone={"z": 1.0},
            mean_energy_eV_by_zone={"z": None},
            kinetics_by_zone={"z": table},
        )


def test_external_power_and_hdf5_kinetics_tables_are_loaded_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "power.csv"
    csv_path.write_text(
        "time_s,electron_power_W,gas_power_W,reduced_field_Td\n0,1,0,1\n1,3,0,2\n",
        encoding="utf-8",
    )
    port = ExternalTablePowerPort(
        "external",
        "z",
        ExternalTableBinding(
            data=load_external_table(csv_path),
            interpolation="linear",
            bounds="error",
            power_scale=1.0,
            voltage_scale=1.0,
            current_scale=1.0,
            gap_m=None,
            total_density_m3=None,
            plasma_potential_V=0.0,
        ),
    )
    coordinator = PowerCoordinator((port,), ("z",))

    h5_path = tmp_path / "rates.h5"
    with h5py.File(h5_path, "w") as handle:
        handle["effective_field_Td"] = np.array([1.0, 2.0])
        handle["mean_energy_eV"] = np.array([2.0, 4.0])
        handle["mobility_m2_V_s"] = np.array([0.5, 0.4])
        rates = handle.create_group("rate_coefficients")
        rates["convert"] = np.array([1.0, 3.0])
    table = TabulatedElectronKinetics.from_hdf5(h5_path, lookup="local_field")

    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CSV reopened")),
    )
    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HDF5 reopened")
        ),
    )
    first = coordinator.evaluate(
        time_s=0.25,
        commands={},
        electron_density_m3_by_zone={"z": 1.0},
        neutral_density_m3_by_zone={"z": 1.0},
        mean_energy_eV_by_zone={"z": None},
        kinetics_by_zone={"z": table},
    )
    second = coordinator.evaluate(
        time_s=0.75,
        commands={},
        electron_density_m3_by_zone={"z": 1.0},
        neutral_density_m3_by_zone={"z": 1.0},
        mean_energy_eV_by_zone={"z": None},
        kinetics_by_zone={"z": table},
    )

    assert first.electron_power_W_by_zone["z"] == pytest.approx(1.5)
    assert second.electron_power_W_by_zone["z"] == pytest.approx(2.5)
    assert first.kinetics_by_zone["z"].rate_coefficients["convert"] == pytest.approx(
        1.5
    )
    assert second.kinetics_by_zone["z"].rate_coefficients["convert"] == pytest.approx(
        2.5
    )


def test_roundoff_negative_density_is_zero_only_for_mass_action() -> None:
    model = CompiledGlobalModel(
        chemistry=reactive_chemistry(2.0),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        domain_atol=np.array([0.1, 0.1, 0.1, 1.0e-30]),
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0, "B": 2.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    a_index = model.layout.density_slices["z"].start
    state[a_index] = -0.5
    evaluation = model.evaluate(0.0, state, model.segments[0])

    assert state[a_index] == -0.5
    assert evaluation.ledger_by_zone["z"].reaction_rates_m3_s["convert"] == 0.0
    assert evaluation.derivative[a_index] == 0.0
    state[a_index] = -1.01
    with pytest.raises(StateDomainError, match=r"-10\*domain_atol"):
        model.evaluate(0.0, state, model.segments[0])


def test_quasineutral_charge_roundoff_uses_the_density_domain_scale() -> None:
    model = CompiledGlobalModel(
        chemistry=reactive_chemistry(2.0),
        zones=(Zone("z", 1.0),),
        segments=(RecipeSegment("step", 0.0, 1.0),),
        electron_closure=ElectronEnergyClosure(),
        domain_atol=np.array([0.1, 0.1, 0.1, 0.1]),
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0, "B": 2.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    state[model.layout.density_slices["z"].stop - 1] = -0.5
    state[model.layout.electron_energy_indices["z"]] = 0.5

    evaluation = model.evaluate(0.0, state, model.segments[0])

    assert evaluation.electron_states["z"].density_m3 == 0.0
    assert evaluation.electron_states["z"].energy_density_J_m3 == 0.0
    assert evaluation.charge_residual_m3_by_zone["z"] == -0.5


def test_roundoff_negative_electron_energy_is_zero_only_for_closure_and_rates() -> None:
    transport = CompiledTransport(
        volumes_m3=np.array([1.0]),
        pump_frequency_s_inv=np.array([2.0]),
        edge_from=np.array([], dtype=int),
        edge_to=np.array([], dtype=int),
        edge_conductance_m3_s=np.array([]),
        n_species=3,
    )
    model = CompiledGlobalModel(
        chemistry=reactive_chemistry(lambda context: context.mean_energy_eV),
        zones=(Zone("z", 1.0),),
        segments=(
            RecipeSegment(
                "step",
                0.0,
                1.0,
                transport=SegmentTransport.zeros(1, 3),
            ),
        ),
        electron_closure=ElectronEnergyClosure(),
        transport=transport,
        domain_atol=np.array([0.1, 0.1, 0.1, 0.1]),
    )
    state = model.initial_state(
        InitialState(
            densities_m3_by_zone={"z": {"A": 1.0, "B": 2.0, "ion": 1.0}},
            mean_energy_eV_by_zone={"z": 3.0},
        )
    )
    energy_index = model.layout.electron_energy_indices["z"]
    state[energy_index] = -0.5
    evaluation = model.evaluate(0.0, state, model.segments[0])

    assert state[energy_index] == -0.5
    assert evaluation.electron_states["z"].energy_density_J_m3 == 0.0
    assert evaluation.electron_states["z"].mean_energy_eV == 0.0
    assert evaluation.ledger_by_zone["z"].reaction_rates_m3_s["convert"] == 0.0
    assert evaluation.derivative[energy_index] == 1.0
    state[energy_index] = -1.01
    with pytest.raises(
        StateDomainError, match=r"Electron energy below -10\*domain_atol"
    ):
        model.evaluate(0.0, state, model.segments[0])


def test_only_result_copy_zeroes_roundoff_negative_components() -> None:
    raw = np.array([[1.0, -0.5, -1.0e-4]])
    accepted, count = _accepted_state_for_result(
        raw,
        np.array([0.0, 0.1, 1.0e-4]),
    )

    np.testing.assert_array_equal(raw, [[1.0, -0.5, -1.0e-4]])
    np.testing.assert_array_equal(accepted, [[1.0, 0.0, 0.0]])
    assert count == 2
    with pytest.raises(IntegrationError, match=r"-10\*domain_atol"):
        _accepted_state_for_result(np.array([[1.0, -1.01]]), np.array([0.0, 0.1]))
