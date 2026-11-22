from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from _model_fixtures import constant_field_port

from plasma_global import load_case, simulate, write_result
from plasma_global._result_network import network_data, network_graph
from plasma_global._result_plot import composition_data
from plasma_global.build import compile_case
from plasma_global.cli import main as cli
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import RecipeSegment
from plasma_global.core.result import series_unit
from plasma_global.errors import CaseValidationError
from plasma_global.input._schema_output import NetworkConfig, PlotConfig
from plasma_global.models.power import CompiledPowerCommand, PowerCoordinator
from plasma_global.output import (
    build_summary,
    network_result_h5,
    plot_result_h5,
    read_result_h5,
)
from plasma_global.postprocess import derive_observables
from tools.generate_schema_docs import minimal_case_document


def _document() -> dict:
    document = minimal_case_document()
    document["chemistry"]["species"].append(
        {
            "id": "Ar_star",
            "phase": "gas",
            "charge": 0,
            "mass_amu": 39.948,
            "elements": {"Ar": 1.0},
        }
    )
    document["chemistry"]["gas_reactions"] = [
        {
            "id": "relax",
            "equation": "Ar_star -> Ar",
            "rate": {"kind": "first_order", "rate_s_inv": 2.0},
        }
    ]
    zone = document["reactor"]["zones"]["plasma"]
    zone["initial"].pop("mole_fractions", None)
    zone["initial"]["densities_m3"] = {"Ar": 1e20, "Ar_star": 1e18, "Ar_plus": 1e12}
    zone["initial"]["pressure_Pa"] = (
        sum(zone["initial"]["densities_m3"].values()) * 1.380649e-23 * 300.0
    )
    zone["initial"].pop("seed_densities_m3", None)
    document["run"] = {"duration_s": 0.1}
    document["solver"] = {"sample_interval_s": 0.001, "rtol": 1e-8, "atol": 1e-15}
    return document


def _case(tmp_path: Path, document: dict):
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_case(path)


def test_zone_probes_event_counts_and_column_units_roundtrip(tmp_path: Path) -> None:
    document = _document()
    document["output"] = {
        "reaction_rates": True,
        "csv": True,
        "probes": {
            "bulk": {
                "zone": "plasma",
                "species": ["Ar_star"],
                "quantities": ["number_density_m3", "mole_fraction", "pressure_Pa"],
            }
        },
    }
    result = simulate(_case(tmp_path, document))
    np.testing.assert_allclose(
        result.series("probe[bulk,number_density_m3,Ar_star]"),
        1e18 * np.exp(-2 * result.time_s),
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        result.series("probe[bulk,mole_fraction,Ar_star]"),
        result.series("n[plasma,Ar_star]") / (1.01e20 + 1e12),
    )
    count = result.series("gas_event_count[plasma,relax]")
    volume = document["reactor"]["zones"]["plasma"]["volume_m3"]
    np.testing.assert_allclose(count[-1], volume * 1e18 * (1 - np.exp(-0.2)), rtol=5e-7)
    assert series_unit("gas_reaction_rate_m3_s[plasma,relax]") == "m^-3 s^-1"
    paths = write_result(result, tmp_path / "run")
    loaded = read_result_h5(paths.result_h5)
    np.testing.assert_array_equal(loaded.series("gas_event_count[plasma,relax]"), count)
    columns = yaml.safe_load(
        (paths.directory / "csv" / "series.yaml").read_text(encoding="utf-8")
    )["columns"]
    probe = next(
        item
        for item in columns
        if item["series"] == "probe[bulk,number_density_m3,Ar_star]"
    )
    assert probe["unit"] == "m^-3"
    assert probe["source_series"] == "n[plasma,Ar_star]"
    assert build_summary(loaded)["probes"]["probe[bulk,pressure_Pa]"]["unit"] == "Pa"


def test_wall_probe_and_boundary_events_use_area_and_volume(tmp_path: Path) -> None:
    document = _document()
    document["chemistry"]["boundary_reactions"] = [
        {"id": "neutralize", "equation": "Ar_plus -> Ar"}
    ]
    document["reactor"]["surfaces"] = {
        "wall": {
            "zone_id": "plasma",
            "area_m2": 0.2,
            "temperature_K": 300.0,
            "wall_transport": {"kind": "prescribed_frequency", "frequency_s_inv": 0.02},
        }
    }
    document["output"] = {
        "reaction_rates": True,
        "probes": {"wafer": {"surface": "wall", "species": ["Ar_plus"]}},
    }
    result = simulate(_case(tmp_path, document))
    volume = document["reactor"]["zones"]["plasma"]["volume_m3"]
    flux = result.series("probe[wafer,incident_flux_m2_s,Ar_plus]")
    np.testing.assert_allclose(
        flux, result.series("n[plasma,Ar_plus]") * 0.02 * volume / 0.2
    )
    np.testing.assert_allclose(
        result.series("boundary_event_rate_s_inv[wall,neutralize]"), flux * 0.2
    )
    assert result.series("boundary_event_count[wall,neutralize]")[-1] == pytest.approx(
        volume * 1e12 * (1 - np.exp(-0.002)), rel=5e-7
    )


def test_cumulative_events_do_not_bridge_a_forcing_jump(tmp_path: Path) -> None:
    compiled = compile_case(_case(tmp_path, _document()))
    segments = (
        RecipeSegment("on", 0.0, 0.05),
        RecipeSegment(
            "off",
            0.05,
            0.1,
            port_commands={"field_plasma": CompiledPowerCommand(kind="off")},
        ),
    )
    chemistry = replace(
        compiled.chemistry, rate_evaluators=(lambda context: context.reduced_field_Td,)
    )
    model = CompiledGlobalModel(
        chemistry,
        compiled.model.zones,
        segments,
        compiled.model.electron_closure,
        power_coordinator=PowerCoordinator(
            (constant_field_port("plasma", 1.0),), ("plasma",)
        ),
    )
    initial = model.initial_state(compiled.initial_state)
    times = np.array([0.0, 0.05, 0.1])
    states = np.tile(initial, (3, 1))
    star = model.species_ids.index("Ar_star")
    ground = model.species_ids.index("Ar")
    states[1:, star] = initial[star] * np.exp(-0.05)
    states[1:, ground] += initial[star] - states[1:, star]
    counts = derive_observables(
        model, times, states, ["gas_event_count[plasma,relax]"]
    )["gas_event_count[plasma,relax]"]
    expected = 0.05 * 0.5 * (initial[star] + states[1, star]) * model.zones[0].volume_m3
    np.testing.assert_allclose(counts, [0, expected, expected])
    with pytest.raises(ValueError, match="sample at the run start"):
        derive_observables(
            model, times[1:], states[1:], ["gas_event_count[plasma,relax]"]
        )


def test_composition_selection_does_not_renormalize_away_other_species(
    tmp_path: Path,
) -> None:
    result = simulate(_case(tmp_path, _document()))
    config = PlotConfig(
        kind="composition", zone="plasma", species=["Ar_star"], quantity="mole_fraction"
    )
    data = composition_data(result, config)
    assert data["Ar_star"][0] == pytest.approx(1e18 / (1.01e20 + 1e12))
    assert data["Ar_star"][0] < 0.01


def test_mechanism_is_bipartite_and_weighted_data_survives_hdf5(tmp_path: Path) -> None:
    document = _document()
    document["output"] = {"networks": {"events": {"mode": "events", "format": "dot"}}}
    result = simulate(_case(tmp_path, document))
    paths = write_result(result, tmp_path / "run")
    loaded = read_result_h5(paths.result_h5)
    data = network_data(loaded, NetworkConfig(mode="events", format="dot"))
    assert data["unit"] == "events"
    assert data["events"][0]["value"] == loaded.final_value(
        "gas_event_count[plasma,relax]"
    )
    dot = network_graph(loaded, data).source
    assert "species_0 -> reaction_0" in dot
    assert "reaction_0 -> species_1" in dot
    assert "species_0 -> species_1" not in dot
    assert any(path.suffix == ".dot" for path in paths.analysis_files)
    exported = network_result_h5(
        paths.result_h5, tmp_path / "redraw", config=NetworkConfig(format="dot")
    )
    assert any(path.suffix == ".json" for path in exported)


def test_event_network_requires_actual_saved_rates_and_exact_time(
    tmp_path: Path,
) -> None:
    result = simulate(_case(tmp_path, _document()))
    assert network_data(result, NetworkConfig())["events"]
    with pytest.raises(ValueError, match="was not saved"):
        network_data(result, NetworkConfig(mode="events"))
    with pytest.raises(ValueError, match="must match a saved time"):
        network_data(result, NetworkConfig(mode="events", time_s=0.000123))


@pytest.mark.parametrize(
    "probe",
    [
        {"zone": "wall", "quantities": ["pressure_Pa"]},
        {"zone": "plasma", "quantities": ["incident_flux_m2_s"]},
        {"surface": "plasma", "quantities": ["number_density_m3"]},
    ],
)
def test_probe_targets_and_quantity_meanings_are_not_interchangeable(
    tmp_path: Path, probe: dict
) -> None:
    document = _document()
    document["output"] = {"probes": {"invalid": probe}}
    with pytest.raises(ValueError, match="probe"):
        compile_case(_case(tmp_path, document))


def test_grouped_plots_split_units_and_can_be_redrawn_from_result(
    tmp_path: Path,
) -> None:
    import matplotlib

    document = _document()
    document["output"] = {
        "probes": {
            "bulk": {
                "zone": "plasma",
                "quantities": ["electron_density_m3", "electron_temperature_eV"],
            }
        },
        "plots": {
            "composition": {
                "kind": "composition",
                "zone": "plasma",
                "species": ["Ar", "Ar_star", "Ar_plus"],
                "yscale": "log",
                "time_unit": "ms",
                "format": "svg",
            },
            "probe": {"series": ["probe[bulk,*]"], "format": "svg"},
        },
    }
    result = simulate(_case(tmp_path, document))
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        paths = write_result(result, tmp_path / "run")
        figures = plot_result_h5(paths.result_h5, tmp_path / "redraw")
        overridden = plot_result_h5(
            paths.result_h5, tmp_path / "redraw-pdf", image_format="pdf"
        )
    assert all(path.suffix == ".pdf" for path in overridden)
    assert len(figures) == 2
    probe = next(path for path in figures if path.stem == "probe").read_text(
        encoding="utf-8"
    )
    assert "[eV]" in probe
    assert "[m^-3]" in probe
    composition = next(
        path for path in figures if path.stem == "composition"
    ).read_text(encoding="utf-8")
    assert "time [ms]" in composition


def test_snapshot_time_is_checked_before_integration(tmp_path: Path) -> None:
    document = _document()
    document["output"] = {
        "networks": {"snapshot": {"mode": "events", "time_s": 0.000123}}
    }
    with pytest.raises(ValueError, match="must match a saved time"):
        compile_case(_case(tmp_path, document))


def test_dc_probe_exposes_the_computed_circuit_values(tmp_path: Path) -> None:
    from plasma_global.examples import initialize_example

    path = initialize_example("zdplaskin_example2", tmp_path / "dc")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["output"]["probes"] = {
        "circuit": {
            "power_port": "dc_series_drive",
            "quantities": [
                "current_A",
                "source_voltage_V",
                "plasma_voltage_V",
                "electron_power_W",
            ],
        }
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = simulate(load_case(path))
    current = result.series("probe[circuit,current_A]")
    voltage = result.series("probe[circuit,plasma_voltage_V]")
    np.testing.assert_allclose(
        result.series("probe[circuit,source_voltage_V]"), voltage + current * 1e5
    )
    np.testing.assert_allclose(
        result.series("probe[circuit,electron_power_W]"), voltage * current
    )


def test_dc_table_hold_keeps_the_actual_field_when_another_port_supplies_power(
    tmp_path: Path,
) -> None:
    from plasma_global.examples import initialize_example

    path = initialize_example("swarm_argon_dc", tmp_path / "dc")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["models"]["electrons"]["coefficients"]["bounds"] = "hold"
    document["reactor"]["power_ports"]["heat"] = {
        "zone_id": "plasma",
        "model": {
            "kind": "prescribed_power",
            "absorbed_power_W": 0.1,
            "electron_fraction": 1.0,
            "gas_fraction": 0.0,
        },
    }
    first, second = document["run"]["steps"]
    first["commands"]["power_ports"]["drive"]["source_voltage_V"] = 70.0
    second["commands"]["power_ports"]["drive"]["source_voltage_V"] = 10.0
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    result = simulate(load_case(path))
    field = result.series("probe[circuit,reduced_field_Td]")
    assert result.status.success
    assert field[0] > 60.0
    assert 0.0 < field[-1] < 40.0
    energy = result.series("mean_energy_eV[plasma]")
    np.testing.assert_allclose(energy[[0, -1]], [5.802317353093206, 5.3993440165709705])


def test_rf_series_retains_zero_voltage_and_frequency_during_off(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _document()
    document["reactor"]["power_ports"] = {
        "drive": {
            "zone_id": "plasma",
            "model": {"kind": "experimental.rf_envelope", "control": "voltage"},
        }
    }
    document["run"] = {
        "steps": [
            {
                "duration_s": 2e-6,
                "commands": {
                    "power_ports": {
                        "drive": {
                            "voltage_rms_V": 5.0,
                            "waveform": {
                                "kind": "square_pulse",
                                "repetition_Hz": 5e5,
                                "duty_cycle": 0.5,
                            },
                        }
                    }
                },
            }
        ]
    }
    document["solver"] = {"save_at_s": [5e-7, 1.5e-6]}
    document["output"] = {
        "probes": {
            "rf": {
                "power_port": "drive",
                "quantities": ["voltage_rms_V", "frequency_Hz"],
            }
        },
    }
    case = _case(tmp_path, document)
    path = tmp_path / "case.yaml"
    assert cli(["outputs", str(path), "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["probe[rf,voltage_rms_V]"]["unit"] == "V"
    assert catalog["probe[rf,voltage_rms_V]"]["stored"] is True

    result = read_result_h5(write_result(simulate(case), tmp_path / "result").result_h5)
    voltage = result.series("probe[rf,voltage_rms_V]")
    on = result.time_s <= 1e-6
    np.testing.assert_array_equal(voltage[on], 5.0)
    np.testing.assert_array_equal(voltage[~on], 0.0)
    np.testing.assert_array_equal(result.series("probe[rf,frequency_Hz]"), 13.56e6)


@pytest.mark.parametrize(
    ("port_id", "setpoint", "series"),
    [
        ("wafer_bias", "absorbed_power_W", "port_*sheath_thickness_m[wafer_bias]"),
        ("source_rf", "delivered_power_W", "port_coupling_efficiency[source_rf]"),
    ],
)
def test_plasma_diagnostic_is_rejected_in_a_zero_setpoint_interval(
    tmp_path: Path, port_id: str, setpoint: str, series: str
) -> None:
    from plasma_global.examples import initialize_example

    path = initialize_example("smoke", tmp_path / "rf")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    step = document["run"]["steps"][0]
    document["run"]["steps"].append(
        {
            **step,
            "commands": {
                **step["commands"],
                "power_ports": {
                    **step["commands"]["power_ports"],
                    port_id: {setpoint: 0.0},
                },
            },
        }
    )
    document["output"] = {"plots": {"diagnostic": {"series": [series]}}}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(CaseValidationError, match=r"undefined in segment 'step_2'"):
        compile_case(load_case(path))


def test_external_table_voltage_series_has_a_consistent_reference_plane(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    power = tmp_path / "power.csv"
    power.write_text(
        "time_s,electron_power_W,voltage_V\n0,0,5\n1,0,5\n", encoding="utf-8"
    )
    without_voltage = tmp_path / "power-only.csv"
    without_voltage.write_text("time_s,electron_power_W\n0,0\n1,0\n", encoding="utf-8")
    document = _document()
    port = {"kind": "external_table", "file": power.name, "gap_m": 0.004}
    document["reactor"]["power_ports"] = {"drive": {"zone_id": "plasma", "model": port}}
    document["run"] = {"duration_s": 2e-6}
    document["solver"] = {"sample_interval_s": 1e-6}
    series = "port_plasma_voltage_V[drive]"
    document["output"] = {"observables": [series, "port_reduced_field_Td[drive]"]}
    result = simulate(_case(tmp_path, document))
    np.testing.assert_array_equal(result.series(series), 5.0)
    np.testing.assert_allclose(
        result.series("port_reduced_field_Td[drive]"), 5.0 / (0.004 * 1.01e20 * 1e-21)
    )
    assert "port_source_voltage_V[drive]" not in result.observables

    port.pop("gap_m")
    document["run"] = {
        "steps": [
            {"duration_s": 1e-6},
            {
                "duration_s": 1e-6,
                "commands": {"power_ports": {"drive": {"file": without_voltage.name}}},
            },
        ]
    }
    document["output"] = {"observables": ["port_voltage_V[drive]"]}
    with pytest.raises(
        CaseValidationError,
        match=r"port_voltage_V\[drive\].*undefined in segment 'step_2'",
    ):
        compile_case(_case(tmp_path, document))
    document["output"] = {}
    _case(tmp_path, document)
    assert cli(["outputs", str(tmp_path / "case.yaml"), "--json"]) == 0
    unavailable = json.loads(capsys.readouterr().out)["port_voltage_V[drive]"]
    assert unavailable["available"] is False
    assert unavailable["unavailable_intervals"][0]["segment_id"] == "step_2"


def test_surface_event_rates_are_per_area_and_counts_are_per_surface(
    tmp_path: Path,
) -> None:
    from plasma_global.examples import initialize_example

    path = initialize_example("smoke", tmp_path / "surface")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["output"]["reaction_rates"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = simulate(load_case(path))
    assert result.metadata["series_units"]["film[wafer]"] == "m"
    assert result.metadata["series_units"]["inventory[wafer,F_reservoir]"] == "1"
    events = [
        event
        for event in result.metadata["reaction_network"]
        if event["domain"] == "surface"
    ]
    assert events
    for event in events:
        area = result.metadata["model_geometry"]["surfaces"][event["region"]]["area_m2"]
        np.testing.assert_allclose(
            result.series(event["frequency_series"]),
            result.series(event["rate_series"]) * area,
        )
        assert np.all(np.diff(result.series(event["count_series"])) >= 0)


def test_missing_renderer_preserves_numerical_result_and_reports_artifact_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document()
    document["output"] = {"networks": {"diagram": {"mode": "mechanism"}}}
    result = simulate(_case(tmp_path, document))
    monkeypatch.setattr(
        "plasma_global._result_network.shutil.which", lambda _name: None
    )
    paths = write_result(result, tmp_path / "partial-artifacts")
    assert paths.result_h5.is_file()
    assert paths.summary_yaml.is_file()
    assert read_result_h5(paths.result_h5).status.success
    assert paths.artifacts.status == "failed"
    assert "dot executable" in paths.artifacts.requests[0].message
    assert paths.artifacts.manifest.is_file()
    assert not (paths.directory / "networks" / "diagram").exists()


@pytest.mark.parametrize("artifact", ["plots", "networks"])
def test_result_artifact_names_cannot_write_outside_the_output_directory(
    tmp_path: Path, artifact: str
) -> None:
    from plasma_global.output import write_result_h5

    result = simulate(_case(tmp_path, _document()))
    definition = (
        {"series": ["n[plasma,Ar]"]} if artifact == "plots" else {"format": "dot"}
    )
    result = replace(
        result,
        metadata={
            **result.metadata,
            "output_requests": {artifact: {"../../outside": definition}},
        },
    )
    source = write_result_h5(tmp_path / "untrusted.h5", result)
    writer = plot_result_h5 if artifact == "plots" else network_result_h5
    with pytest.raises(ValueError, match="string_pattern_mismatch"):
        writer(source, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_species_focus_records_the_requested_reaction_subset(tmp_path: Path) -> None:
    document = _document()
    document["chemistry"]["boundary_reactions"] = [
        {"id": "neutralize", "equation": "Ar_plus -> Ar"}
    ]
    document["reactor"]["surfaces"] = {
        "wall": {
            "zone_id": "plasma",
            "area_m2": 0.2,
            "temperature_K": 300.0,
            "wall_transport": {"kind": "prescribed_frequency", "frequency_s_inv": 0.02},
        }
    }
    document["output"] = {
        "networks": {
            "ions": {"mode": "events", "species": ["Ar_plus"], "format": "dot"}
        }
    }
    result = simulate(_case(tmp_path, document))
    assert "gas_event_count[plasma,relax]" not in result.observables
    data = network_data(result, NetworkConfig(mode="events", species=["Ar_plus"]))
    assert [event["domain"] for event in data["events"]] == ["boundary"]
    assert data["selection"]["species"] == ["Ar_plus"]


@pytest.mark.parametrize("with_power_port", [False, True])
def test_mean_energy_table_field_is_selectable_and_saved(
    tmp_path: Path, with_power_port: bool
) -> None:
    table = tmp_path / "electrons.csv"
    table.write_text(
        "mean_energy_eV,reduced_field_Td\n1,10\n10,100\n", encoding="utf-8"
    )
    document = minimal_case_document()
    document["models"]["electrons"] = {
        "kind": "energy_balance",
        "coefficients": {"file": table.name},
    }
    if with_power_port:
        document["reactor"]["power_ports"] = {
            "heat": {
                "zone_id": "plasma",
                "model": {
                    "kind": "prescribed_power",
                    "absorbed_power_W": 1e-6,
                    "electron_fraction": 1.0,
                    "gas_fraction": 0.0,
                },
            }
        }
    document["solver"] = {"sample_interval_s": 0.0005}
    field = "reduced_field_Td[plasma]"
    document["output"] = {"observables": [field, "mean_energy_eV[plasma]"]}
    case = _case(tmp_path, document)
    assert compile_case(case).output_plan.available_units[field] == "Td"

    result = read_result_h5(write_result(simulate(case), tmp_path / "run").result_h5)

    np.testing.assert_allclose(
        result.series(field), 10.0 * result.series("mean_energy_eV[plasma]")
    )
    assert result.series(field)[0] == pytest.approx(30.0)
    assert result.metadata["series_units"][field] == "Td"


def test_saved_network_name_merges_only_explicit_overrides(tmp_path: Path) -> None:
    document = _document()
    document["output"] = {
        "networks": {
            "focus": {
                "mode": "events",
                "species": ["Ar_star"],
                "domains": ["gas"],
                "format": "dot",
            },
            "other": {"mode": "mechanism", "format": "dot"},
        }
    }
    paths = write_result(simulate(_case(tmp_path, document)), tmp_path / "run")
    for changes in (None, {"time_s": 0.0}):
        with pytest.raises(ValueError, match="multiple saved networks require --name"):
            network_result_h5(
                paths.result_h5, tmp_path / "ambiguous", overrides=changes
            )
    assert not (tmp_path / "ambiguous").exists()
    generated = network_result_h5(
        paths.result_h5,
        tmp_path / "redraw",
        name="focus",
        overrides={"time_s": 0.0},
    )
    record = json.loads(
        next(path for path in generated if path.suffix == ".json").read_text(
            encoding="utf-8"
        )
    )
    assert record["mode"] == "events"
    assert record["selection"]["species"] == ["Ar_star"]
    assert record["selection"]["domains"] == ["gas"]
    assert record["time_s"] == 0.0
    with pytest.raises(ValueError, match="unknown saved network"):
        network_result_h5(paths.result_h5, tmp_path / "wrong", name="unknown")


def test_direct_plot_expands_the_same_literal_star_selector(tmp_path: Path) -> None:
    paths = write_result(simulate(_case(tmp_path, _document())), tmp_path / "run")
    generated = plot_result_h5(
        paths.result_h5,
        tmp_path / "species",
        series=["n[plasma,*]"],
        image_format="svg",
    )
    assert len(generated) == 3
    assert all(path.is_file() for path in generated)
