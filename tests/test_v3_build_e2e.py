from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pytest

import plasma_global.build as build_module
import plasma_global.models.external_table as external_table_module
from plasma_global.build import CompiledCase, compile_case, simulate_case
from plasma_global.errors import CaseValidationError
from plasma_global.input.load import load_case
from plasma_global.input.schema import CaseSpec
from plasma_global.models.electrons import ELEMENTARY_CHARGE_C
from plasma_global.models.gas_energy import elastic_electron_heating_J_m3_s
from plasma_global.models.power import CompiledPowerCommand

FIXTURE = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"


def _updated_case(case: CaseSpec, update: object) -> CaseSpec:
    data = case.model_dump(mode="python")
    update(data)
    return CaseSpec.model_validate(data)


def _write_momentum_chemistry(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "species.csv").write_text(
        "id,phase,charge,mass_amu,elements,state_tags,cv_over_kb\n"
        "e,gas,-1,0.00054858,,electron,\n"
        "Ar,gas,0,39.948,Ar:1,,1.5\n"
        "Ar_plus,gas,1,39.948,Ar:1,ion,1.5\n",
        encoding="utf-8",
    )
    (directory / "gas.csv").write_text("id,equation,rate_model\n", encoding="utf-8")
    (directory / "boundary.csv").write_text(
        "id,equation,surfaces\nneutralize,Ar_plus -> Ar,wall\n",
        encoding="utf-8",
    )
    (directory / "rates.yaml").write_text("rate_models: {}\n", encoding="utf-8")
    (directory / "ar-momentum.csv").write_text(
        "energy_eV,sigma_m2\n0.0,1.0e-20\n1.0,1.2e-20\n10.0,2.0e-20\n100.0,1.0e-20\n",
        encoding="utf-8",
    )
    (directory / "cross-sections.yaml").write_text(
        "cross_sections:\n"
        "  - id: Ar_momentum\n"
        "    kind: momentum_transfer\n"
        "    target: Ar\n"
        "    threshold_eV: 0.0\n"
        "    energy_loss_eV: 0.0\n"
        "    file: ar-momentum.csv\n",
        encoding="utf-8",
    )
    manifest = directory / "chemistry.yaml"
    manifest.write_text(
        "schema_version: 3\n"
        "species: species.csv\n"
        "gas_reactions: gas.csv\n"
        "boundary_reactions: boundary.csv\n"
        "rate_models: rates.yaml\n"
        "cross_sections: cross-sections.yaml\n",
        encoding="utf-8",
    )
    return manifest


def _write_surface_chemistry(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "species.csv").write_text(
        "id,phase,charge,mass_amu,elements,state_tags,surfaces\n"
        "e,gas,-1,0.00054858,,electron,\n"
        "A,gas,0,40,A:1,,\n"
        "A_plus,gas,1,40,A:1,ion,\n"
        "wall:*,surface,0,0,site:1,site,wall\n"
        "wall:A*,surface,0,40,A:1;site:1,adsorbate,wall\n",
        encoding="utf-8",
    )
    (directory / "gas.csv").write_text("id,equation,rate_model\n", encoding="utf-8")
    (directory / "surface.csv").write_text(
        "id,equation,rate_model,zones,surfaces\n"
        "stick,A + wall:* -> wall:A*,stick,plasma,wall\n",
        encoding="utf-8",
    )
    (directory / "rates.yaml").write_text(
        "rate_models:\n"
        "  stick:\n"
        "    kind: sticking\n"
        "    value: 0.1\n"
        "    coverage:\n"
        "      kind: site_blocking\n"
        "      site_species: wall:*\n"
        "      exponent: 1.0\n",
        encoding="utf-8",
    )
    manifest = directory / "chemistry.yaml"
    manifest.write_text(
        "schema_version: 3\n"
        "species: species.csv\n"
        "gas_reactions: gas.csv\n"
        "surface_reactions: surface.csv\n"
        "rate_models: rates.yaml\n",
        encoding="utf-8",
    )
    return manifest


def test_canonical_v3_case_loads_compiles_and_simulates() -> None:
    case = load_case(FIXTURE)
    compiled = compile_case(case)
    result = simulate_case(case)

    assert isinstance(compiled, CompiledCase)
    assert compiled.chemistry.species_ids == ("Ar", "Ar_plus")
    assert compiled.model.wall_boundaries[0].reactions[0].reaction_id == "neutralize"
    assert result.status.success
    assert result.time_s[0] == pytest.approx(case.recipe.start_time_s)
    assert result.time_s[-1] == pytest.approx(case.recipe.end_time_s)
    assert result.state.shape == (result.time_s.size, 3)
    assert np.isfinite(result.state).all()
    assert "schema_version: 3" in result.metadata["effective_case_yaml"]
    assert result.metadata["model_ids"]["electron_closure"] == "electron_energy"
    assert result.metadata["provenance"]["chemistry"]["fixture"] == (
        "canonical-minimal-v3"
    )


def test_compile_case_revalidates_and_detaches_the_case_snapshot() -> None:
    case = load_case(FIXTURE)
    compiled = compile_case(case)

    assert compiled.case is not case
    assert compiled.case.source_path == case.source_path
    assert compiled.case.reactor.zones is not case.reactor.zones
    case.output.observables.append("density:Ar")
    assert compiled.case.output.observables == []

    invalid = load_case(FIXTURE)
    densities = invalid.reactor.zones[0].initial_densities_m3
    assert densities is not None
    cast(dict[str, Any], densities)["Ar"] = "not-a-density"
    with pytest.raises(CaseValidationError, match="changes made after validation"):
        compile_case(invalid)


def test_pressure_composition_and_explicit_ion_seed_compile_to_densities() -> None:
    def update(data: dict[str, object]) -> None:
        zone = data["reactor"]["zones"][0]
        zone.pop("initial_densities_m3")
        zone["initial_mole_fractions"] = {"Ar": 1.0}
        zone["initial_seed_densities_m3"] = {"Ar_plus": 1.0e12}

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))

    assert compiled.initial_state.densities_m3_by_zone["plasma"] == pytest.approx(
        {"Ar": 2.0e20, "Ar_plus": 1.0e12}
    )


def test_fixed_gas_energy_is_the_schema_default() -> None:
    def update(data: dict[str, object]) -> None:
        data["models"].pop("gas_energy")

    case = _updated_case(load_case(FIXTURE), update)
    compiled = compile_case(case)

    assert case.models.gas_energy.kind == "fixed"
    assert not compiled.model.layout.evolves_heavy_energy


def test_every_zone_requires_an_explicit_positive_ion_seed() -> None:
    def update(data: dict[str, object]) -> None:
        zone = data["reactor"]["zones"][0]
        zone["initial_densities_m3"] = {"Ar": 2.00000001e20}

    with pytest.raises(CaseValidationError, match="positive-ion seed"):
        compile_case(_updated_case(load_case(FIXTURE), update))


def test_square_pulse_is_split_at_every_switch_boundary() -> None:
    def update(data: dict[str, object]) -> None:
        step = data["recipe"]["steps"][0]
        step["duration_s"] = 1.0
        step["commands"]["power_ports"]["source"]["absorbed_power_W"] = 1.0
        step["commands"]["power_ports"]["source"]["waveform"] = {
            "kind": "square_pulse",
            "duty_cycle": 0.25,
            "repetition_Hz": 2.0,
            "phase_s": 0.0,
        }
        data["reactor"]["surfaces"][0]["wall_transport"] = {"kind": "off"}
        data["solver"]["max_step_s"] = 0.05

    case = _updated_case(load_case(FIXTURE), update)
    compiled = compile_case(case)
    boundaries = [compiled.segments[0].start_s]
    boundaries.extend(segment.end_s for segment in compiled.segments)

    assert boundaries == pytest.approx([0.0, 0.125, 0.5, 0.625, 1.0])
    commands = [segment.port_commands["source"] for segment in compiled.segments]
    assert all(isinstance(command, CompiledPowerCommand) for command in commands)
    assert [command.kind for command in commands] == ["power", "off", "power", "off"]
    initial = compiled.model.initial_state(compiled.initial_state)
    powers = [
        compiled.model.evaluate(
            segment.start_s + 0.5 * (segment.end_s - segment.start_s),
            initial,
            segment,
        )
        .ledger_by_zone["plasma"]
        .absorbed_power_J_m3_s
        for segment in compiled.segments
    ]
    assert powers == pytest.approx([100.0, 0.0, 100.0, 0.0])

    energy_index = compiled.model.layout.electron_energy_indices["plasma"]
    expected_energy = initial[energy_index] + sum(
        power * (segment.end_s - segment.start_s)
        for power, segment in zip(powers, compiled.segments, strict=True)
    )
    result = simulate_case(case)
    relative_error = abs(result.state[-1, energy_index] - expected_energy) / abs(
        expected_energy
    )
    assert relative_error <= 1.0e-6


@pytest.mark.parametrize(
    ("transport", "expected_kind", "attribute", "expected_value"),
    [
        ({"kind": "bohm"}, "bohm", "sheath_energy_eV", 7.5),
        (
            {"kind": "prescribed_frequency", "frequency_s_inv": 12.0},
            "prescribed_frequency",
            "prescribed_frequency_s_inv",
            12.0,
        ),
        (
            {
                "kind": "ambipolar",
                "diffusion_coefficient_m2_s": 0.2,
                "diffusion_length_m": 0.05,
            },
            "ambipolar",
            "diffusion_length_m",
            0.05,
        ),
    ],
)
def test_typed_wall_transport_variants_and_impact_energy_compile(
    transport: dict[str, object],
    expected_kind: str,
    attribute: str,
    expected_value: float,
) -> None:
    def update(data: dict[str, object]) -> None:
        surface = data["reactor"]["surfaces"][0]
        surface["wall_transport"] = transport
        surface["ion_impact_energy_eV"] = 7.5

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    wall = compiled.model.wall_boundaries[0]

    assert wall.transport_kind == expected_kind
    assert wall.sheath_energy_eV == pytest.approx(7.5)
    assert getattr(wall, attribute) == pytest.approx(expected_value)


def test_off_wall_transport_compiles_no_boundary() -> None:
    def update(data: dict[str, object]) -> None:
        data["reactor"]["surfaces"][0]["wall_transport"] = {"kind": "off"}

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    assert compiled.model.wall_boundaries == ()


@pytest.mark.parametrize(
    ("h_factor", "expected_mode"),
    [(0.61, "numeric"), ("auto", "auto")],
)
def test_bohm_h_factor_contract_is_recorded_without_changing_model_id(
    h_factor: float | str,
    expected_mode: str,
) -> None:
    def update(data: dict[str, object]) -> None:
        data["reactor"]["surfaces"][0]["wall_transport"] = {
            "kind": "bohm",
            "h_factor": h_factor,
        }

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))

    assert compiled.metadata["model_ids"]["wall_transport"] == {"wall": "bohm"}
    assert compiled.metadata["provenance"]["wall_transport_closure"] == {
        "bohm_h_factor": {
            "version": "direct-multiplier-v2",
            "surface_modes": {"wall": expected_mode},
        }
    }


def test_schema_bohm_h_factor_is_the_complete_edge_density_factor() -> None:
    def update(data: dict[str, object]) -> None:
        data["reactor"]["surfaces"][0]["wall_transport"] = {
            "kind": "bohm",
            "h_factor": 1.0,
        }

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    state = compiled.model.initial_state(compiled.initial_state)
    evaluated = compiled.model.evaluate(0.0, state, compiled.segments[0])
    electrons = evaluated.electron_states["plasma"]
    ion_index = compiled.chemistry.species_ids.index("Ar_plus")
    sound_speed = np.sqrt(
        ELEMENTARY_CHARGE_C
        * electrons.temperature_eV
        / compiled.chemistry.masses_kg[ion_index]
    )

    assert evaluated.ledger_by_zone["plasma"].wall_fluxes[
        0
    ].bohm_speed_m_s == pytest.approx(sound_speed)


def test_auto_bohm_h_factor_tracks_current_neutral_density_and_limits() -> None:
    length_m = 0.1
    cross_section_m2 = 1.0e-18

    def update(data: dict[str, object]) -> None:
        data["reactor"]["surfaces"][0]["wall_transport"] = {
            "kind": "bohm",
            "h_factor": "auto",
            "characteristic_length_m": length_m,
            "ion_neutral_cross_section_m2": cross_section_m2,
            "min_h_factor": 0.02,
            "max_h_factor": 1.0,
        }

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    initial = compiled.model.initial_state(compiled.initial_state)
    neutral_index = compiled.model.layout.density_slices["plasma"].start

    def evaluated_speed(neutral_density_m3: float) -> float:
        state = initial.copy()
        state[neutral_index] = neutral_density_m3
        evaluated = compiled.model.evaluate(0.0, state, compiled.segments[0])
        return evaluated.ledger_by_zone["plasma"].wall_fluxes[0].bohm_speed_m_s

    low_density_speed = evaluated_speed(0.0)
    initial_density_speed = evaluated_speed(2.0e20)
    high_density_speed = evaluated_speed(1.0e25)
    electrons = compiled.model.evaluate(
        0.0, initial, compiled.segments[0]
    ).electron_states["plasma"]
    ion_index = compiled.chemistry.species_ids.index("Ar_plus")
    sound_speed = np.sqrt(
        ELEMENTARY_CHARGE_C
        * electrons.temperature_eV
        / compiled.chemistry.masses_kg[ion_index]
    )
    low_density_limit = 0.86 / np.sqrt(3.0)
    expected_initial = 0.86 / np.sqrt(3.0 + length_m * 2.0e20 * cross_section_m2 / 2.0)

    assert low_density_speed == pytest.approx(low_density_limit * sound_speed)
    assert initial_density_speed == pytest.approx(expected_initial * sound_speed)
    assert high_density_speed == pytest.approx(0.02 * sound_speed)
    assert low_density_speed > initial_density_speed > high_density_speed


def test_evolved_gas_inlet_compiles_particle_enthalpy_source(tmp_path: Path) -> None:
    manifest = _write_momentum_chemistry(tmp_path / "chemistry")

    def update(data: dict[str, object]) -> None:
        data["chemistry"]["manifest"] = manifest
        data["models"]["gas_energy"] = {"kind": "evolved"}
        data["reactor"]["gas_inlets"] = [
            {
                "inlet_id": "feed",
                "zone_id": "plasma",
                "flow_sccm": {"Ar": 1.0},
                "temperature_K": 450.0,
            }
        ]

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    source = compiled.segments[0].transport.inlet_heavy_energy_J_m3_s
    particles_m3_s = 4.477962e17 / 0.01
    expected = particles_m3_s * (1.5 + 1.0) * 1.380649e-23 * 450.0

    np.testing.assert_allclose(source, [expected])


def test_compile_reads_each_shared_external_model_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    power_path = tmp_path / "power.csv"
    power_path.write_text(
        "time_s,electron_power_W,reduced_field_Td\n0.0,1.0e-6,10.0\n1.0,2.0e-6,20.0\n",
        encoding="utf-8",
    )
    kinetics_path = tmp_path / "kinetics.h5"
    with h5py.File(kinetics_path, "w") as handle:
        handle["mean_energy_eV"] = np.array([1.0, 10.0])
        handle["mobility_m2_V_s"] = np.array([0.5, 0.1])
        handle["effective_field_Td"] = np.array([10.0, 100.0])
        handle.create_group("rate_coefficients")

    def update(data: dict[str, object]) -> None:
        data["reactor"]["power_ports"][0]["model"] = {
            "kind": "external_table",
            "file": power_path,
        }
        command = data["recipe"]["steps"][0]["commands"]["power_ports"]["source"]
        data["recipe"]["steps"][0]["commands"]["power_ports"]["source"] = {
            "kind": "external_table",
            "file": power_path,
        }
        assert command
        data["models"]["electrons"] = {
            "kind": "table",
            "file": kinetics_path,
            "lookup": "mean_energy",
        }

    case = _updated_case(load_case(FIXTURE), update)
    counts = {"chemistry": 0, "power": 0, "kinetics": 0}
    original_chemistry = build_module.load_chemistry
    original_power = external_table_module.load_external_table
    original_kinetics = build_module.TabulatedElectronKinetics.from_hdf5

    def counted_chemistry(path: Path) -> object:
        counts["chemistry"] += 1
        return original_chemistry(path)

    def counted_power(path: Path) -> object:
        counts["power"] += 1
        return original_power(path)

    def counted_kinetics(*args: object, **kwargs: object) -> object:
        counts["kinetics"] += 1
        return original_kinetics(*args, **kwargs)

    monkeypatch.setattr(build_module, "load_chemistry", counted_chemistry)
    monkeypatch.setattr(external_table_module, "load_external_table", counted_power)
    monkeypatch.setattr(
        build_module.TabulatedElectronKinetics,
        "from_hdf5",
        counted_kinetics,
    )
    compile_case(case)

    assert counts == {"chemistry": 1, "power": 1, "kinetics": 1}


def test_previous_external_table_knots_bind_each_forcing_interval(
    tmp_path: Path,
) -> None:
    power_path = tmp_path / "piecewise-power.csv"
    power_path.write_text(
        "time_s,electron_power_W\n-0.1,1.0e-6\n0.15,2.0e-6\n0.55,3.0e-6\n0.9,4.0e-6\n",
        encoding="utf-8",
    )

    def update(data: dict[str, object]) -> None:
        data["reactor"]["power_ports"][0]["model"] = {
            "kind": "external_table",
            "file": power_path,
            "interpolation": "previous",
        }
        step = data["recipe"]["steps"][0]
        step["duration_s"] = 1.0
        step["commands"]["power_ports"]["source"] = {
            "kind": "external_table",
            "file": power_path,
            "interpolation": "previous",
            "time_offset_s": 0.1,
        }
        data["solver"]["max_step_s"] = 0.05

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    boundaries = [compiled.segments[0].start_s]
    boundaries.extend(segment.end_s for segment in compiled.segments)
    bindings = [segment.port_commands["source"] for segment in compiled.segments]

    assert boundaries == pytest.approx([0.0, 0.25, 0.65, 1.0])
    assert all(
        isinstance(binding, CompiledPowerCommand)
        and binding.kind == "external_table"
        and not binding.time_dependent
        for binding in bindings
    )
    assert len({id(binding) for binding in bindings}) == len(bindings)
    initial = compiled.model.initial_state(compiled.initial_state)
    powers = [
        compiled.model.evaluate(
            segment.start_s + 0.5 * (segment.end_s - segment.start_s),
            initial,
            segment,
        )
        .ledger_by_zone["plasma"]
        .absorbed_power_J_m3_s
        for segment in compiled.segments
    ]
    assert powers == pytest.approx([1.0e-4, 2.0e-4, 3.0e-4])
    left_limits = [
        compiled.model.evaluate(segment.end_s, initial, segment)
        .ledger_by_zone["plasma"]
        .absorbed_power_J_m3_s
        for segment in compiled.segments[:-1]
    ]
    right_values = [
        compiled.model.evaluate(segment.start_s, initial, segment)
        .ledger_by_zone["plasma"]
        .absorbed_power_J_m3_s
        for segment in compiled.segments[1:]
    ]
    assert left_limits == pytest.approx([1.0e-4, 2.0e-4])
    assert right_values == pytest.approx([2.0e-4, 3.0e-4])


def test_linear_external_table_does_not_create_knot_segments(tmp_path: Path) -> None:
    power_path = tmp_path / "linear-power.csv"
    power_path.write_text(
        "time_s,electron_power_W\n0.0,1.0e-6\n0.5,2.0e-6\n1.0,3.0e-6\n",
        encoding="utf-8",
    )

    def update(data: dict[str, object]) -> None:
        data["reactor"]["power_ports"][0]["model"] = {
            "kind": "external_table",
            "file": power_path,
            "interpolation": "linear",
        }
        step = data["recipe"]["steps"][0]
        step["duration_s"] = 1.0
        step["commands"]["power_ports"]["source"] = {"kind": "external_table"}

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    assert len(compiled.segments) == 1
    initial = compiled.model.initial_state(compiled.initial_state)
    segment = compiled.segments[0]
    compiled.model.evaluate(segment.start_s, initial, segment)
    compiled.model.evaluate(segment.end_s, initial, segment)


def test_local_field_table_and_dc_series_are_bound_to_the_core(
    tmp_path: Path,
) -> None:
    kinetics_path = tmp_path / "local-field.h5"
    with h5py.File(kinetics_path, "w") as handle:
        handle["mean_energy_eV"] = np.array([1.0, 5.0])
        handle["mobility_m2_V_s"] = np.array([0.1, 0.1])
        handle["effective_field_Td"] = np.array([10.0, 1000.0])
        handle.create_group("rate_coefficients")

    def update(data: dict[str, object]) -> None:
        zone = data["reactor"]["zones"][0]
        zone.pop("initial_mean_energy_eV")
        data["reactor"]["power_ports"][0]["model"] = {
            "kind": "dc_series",
            "ballast_resistance_ohm": 1.0e6,
            "gap_m": 0.1,
            "electrode_area_m2": 0.01,
        }
        data["recipe"]["steps"][0]["commands"]["power_ports"]["source"] = {
            "kind": "dc_series",
            "source_voltage_V": 10.0,
        }
        data["models"]["electrons"] = {
            "kind": "table",
            "file": kinetics_path,
            "lookup": "local_field",
        }
        data["models"]["electron_closure"] = {"kind": "local_field"}

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    state = compiled.model.initial_state(compiled.initial_state)
    segment = compiled.segments[0]
    evaluated = compiled.model.evaluate(segment.start_s, state, segment)

    assert compiled.model.layout.evolves_electron_energy is False
    assert evaluated.electron_states["plasma"].reduced_field_Td == pytest.approx(
        499.2, rel=1.0e-2
    )
    assert evaluated.ledger_by_zone["plasma"].absorbed_power_J_m3_s > 0.0


def test_requested_observables_are_postprocessed_without_output_side_effects() -> None:
    names = [
        "electron_density_m3[plasma]",
        "mean_energy_eV[plasma]",
        "electron_temperature_eV[plasma]",
        "gas_temperature_K[plasma]",
        "charge_residual_m3[plasma]",
        "absorbed_power_W[plasma]",
    ]

    def update(data: dict[str, object]) -> None:
        model = data["reactor"]["power_ports"][0]["model"]
        model["electron_fraction"] = 0.75
        model["gas_fraction"] = 0.25
        data["output"]["observables"] = names
        data["output"]["summary_series"] = [
            "n[plasma,Ar_plus]",
            "mean_energy_eV[plasma]",
        ]

    case = _updated_case(load_case(FIXTURE), update)
    result = simulate_case(case)

    assert list(result.observables) == names
    assert result.observables["electron_density_m3[plasma]"] == pytest.approx(
        result.series("n[plasma,Ar_plus]")
    )
    assert result.observables["electron_temperature_eV[plasma]"] == pytest.approx(
        (2.0 / 3.0) * result.observables["mean_energy_eV[plasma]"]
    )
    assert result.observables["gas_temperature_K[plasma]"] == pytest.approx(300.0)
    assert result.observables["charge_residual_m3[plasma]"] == pytest.approx(0.0)
    assert result.observables["absorbed_power_W[plasma]"] == pytest.approx(1.0e-6)
    assert result.metadata["summary_series"] == (
        "n[plasma,Ar_plus]",
        "mean_energy_eV[plasma]",
    )


def test_unknown_observable_is_a_compile_error() -> None:
    def update(data: dict[str, object]) -> None:
        data["output"]["observables"] = ["legacy_everything"]

    with pytest.raises(ValueError, match="unknown output observables"):
        compile_case(_updated_case(load_case(FIXTURE), update))


def test_unknown_summary_series_is_a_compile_error() -> None:
    def update(data: dict[str, object]) -> None:
        data["output"]["summary_series"] = ["n[plasma,legacy_species]"]

    with pytest.raises(ValueError, match="unknown output summary series"):
        compile_case(_updated_case(load_case(FIXTURE), update))


def test_prescribed_electron_density_profile_is_bound_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / "electron-density.csv"
    profile_path.write_text("time_s,ne\n0.0,2.0e12\n1.0e-7,4.0e12\n", encoding="utf-8")

    def update(data: dict[str, object]) -> None:
        data["models"]["electron_density"] = {
            "kind": "experimental.prescribed_profile",
            "file": profile_path,
            "zone_columns": {"plasma": "ne"},
            "interpolation": "linear",
            "hold": "error",
        }
        data["output"]["observables"] = [
            "electron_density_m3[plasma]",
            "charge_residual_m3[plasma]",
        ]

    from plasma_global.experimental import profile as profile_module

    count = 0
    original = profile_module.PrescribedElectronProfile.from_csv.__func__

    def counted(cls: object, *args: object, **kwargs: object) -> object:
        nonlocal count
        count += 1
        return original(cls, *args, **kwargs)

    monkeypatch.setattr(
        profile_module.PrescribedElectronProfile,
        "from_csv",
        classmethod(counted),
    )
    result = simulate_case(_updated_case(load_case(FIXTURE), update))

    assert count == 1
    assert result.observables["electron_density_m3[plasma]"][[0, -1]] == pytest.approx(
        [2.0e12, 4.0e12]
    )
    assert result.observables["charge_residual_m3[plasma]"][0] == pytest.approx(-1.0e12)


def test_previous_electron_profile_is_bound_on_each_side_of_a_knot(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "electron-density.csv"
    profile_path.write_text(
        "time_s,ne\n0.0,2.0e12\n5.0e-8,4.0e12\n1.0e-7,6.0e12\n",
        encoding="utf-8",
    )

    def update(data: dict[str, object]) -> None:
        data["models"]["electron_density"] = {
            "kind": "experimental.prescribed_profile",
            "file": profile_path,
            "zone_columns": {"plasma": "ne"},
            "interpolation": "previous",
            "hold": "error",
        }

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    assert [segment.end_s for segment in compiled.segments] == pytest.approx(
        [5.0e-8, 1.0e-7]
    )
    state = compiled.model.initial_state(compiled.initial_state)
    knot = compiled.segments[0].end_s
    left = compiled.model.evaluate(knot, state, compiled.segments[0])
    right = compiled.model.evaluate(knot, state, compiled.segments[1])
    assert left.electron_states["plasma"].density_m3 == pytest.approx(2.0e12)
    assert right.electron_states["plasma"].density_m3 == pytest.approx(4.0e12)


def test_momentum_cross_section_compiles_elastic_heating_into_both_ledgers(
    tmp_path: Path,
) -> None:
    manifest = _write_momentum_chemistry(tmp_path / "chemistry")

    def update(data: dict[str, object]) -> None:
        data["chemistry"]["manifest"] = manifest
        data["models"]["gas_energy"] = {
            "kind": "evolved",
            "wall_energy_relaxation_s_inv_by_zone": {"plasma": 0.0},
        }

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    state = compiled.model.initial_state(compiled.initial_state)
    evaluated = compiled.model.evaluate(0.0, state, compiled.segments[0])
    ledger = evaluated.ledger_by_zone["plasma"]
    electron_index = compiled.model.layout.electron_energy_indices["plasma"]
    heavy_index = compiled.model.layout.heavy_energy_indices["plasma"]

    assert ledger.elastic_heating_J_m3_s > 0.0
    assert evaluated.derivative[heavy_index] == pytest.approx(
        ledger.elastic_heating_J_m3_s
    )
    assert evaluated.derivative[electron_index] == pytest.approx(
        ledger.absorbed_power_J_m3_s
        - ledger.wall_energy_loss_J_m3_s
        - ledger.elastic_heating_J_m3_s
    )
    assert compiled.metadata["model_ids"]["elastic_heating"] == (
        "momentum_cross_sections"
    )
    assert (
        compiled.metadata["provenance"]["missing_momentum_cross_section_targets"] == ()
    )


def test_table_eedf_momentum_rate_overrides_maxwellian_fallback(tmp_path: Path) -> None:
    manifest = _write_momentum_chemistry(tmp_path / "chemistry")
    kinetics_path = tmp_path / "kinetics.h5"
    with h5py.File(kinetics_path, "w") as handle:
        handle["mean_energy_eV"] = np.array([1.0, 10.0])
        handle["mobility_m2_V_s"] = np.array([0.5, 0.1])
        handle["effective_field_Td"] = np.array([10.0, 100.0])
        rates = handle.create_group("rate_coefficients")
        rates["Ar_momentum"] = np.array([1.0e-13, 1.0e-13])

    def update(data: dict[str, object]) -> None:
        data["chemistry"]["manifest"] = manifest
        data["models"]["electrons"] = {
            "kind": "table",
            "file": kinetics_path,
            "lookup": "mean_energy",
        }
        data["models"]["gas_energy"] = {"kind": "evolved"}

    compiled = compile_case(_updated_case(load_case(FIXTURE), update))
    state = compiled.model.initial_state(compiled.initial_state)
    evaluated = compiled.model.evaluate(0.0, state, compiled.segments[0])
    expected = elastic_electron_heating_J_m3_s(
        electron_density_m3=1.0e12,
        electron_temperature_eV=2.0,
        gas_temperature_K=300.0,
        neutral_densities_m3=np.array([2.0e20]),
        neutral_masses_kg=np.array([compiled.chemistry.masses_kg[0]]),
        momentum_rate_coefficients_m3_s=np.array([1.0e-13]),
    )
    assert evaluated.ledger_by_zone["plasma"].elastic_heating_J_m3_s == pytest.approx(
        expected
    )


def test_surface_state_reaction_temperature_override_and_free_site_observable(
    tmp_path: Path,
) -> None:
    manifest = _write_surface_chemistry(tmp_path / "surface-chemistry")

    def update(data: dict[str, object]) -> None:
        data["chemistry"]["manifest"] = manifest
        zone = data["reactor"]["zones"][0]
        zone["initial_densities_m3"] = {"A": 1.0e20, "A_plus": 1.0e12}
        zone["pressure_Pa"] = 1.00000001e20 * 1.380649e-23 * 300.0
        surface = data["reactor"]["surfaces"][0]
        surface["site_density_m2"] = 1.0e19
        surface["initial_coverages"] = {"wall:A*": 0.0}
        surface["wall_transport"] = {"kind": "off"}
        data["models"]["surface_kinetics"] = {}
        commands = data["recipe"]["steps"][0]["commands"]
        commands["surfaces"] = {"wall": {"temperature_K": 450.0}}
        data["output"]["observables"] = ["coverage[wall,wall:*]"]

    case = _updated_case(load_case(FIXTURE), update)
    compiled = compile_case(case)
    initial = compiled.model.initial_state(compiled.initial_state)
    evaluated = compiled.model.evaluate(0.0, initial, compiled.segments[0])
    coverage_index = compiled.model.layout.surface_coverage_indices[("wall", "wall:A*")]
    gas_index = compiled.model.layout.density_slices["plasma"].start

    assert compiled.segments[0].surface_temperature_K_by_surface == {"wall": 450.0}
    assert evaluated.derivative[coverage_index] > 0.0
    assert evaluated.derivative[gas_index] < 0.0
    result = simulate_case(case)
    assert result.observables["coverage[wall,wall:*]"][0] == pytest.approx(1.0)
    assert result.observables["coverage[wall,wall:*]"][-1] < 1.0
