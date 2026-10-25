from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global import load_case, simulate, write_result
from plasma_global._result_network import network_data, network_graph
from plasma_global._result_plot import composition_data
from plasma_global.build import compile_case
from plasma_global.core.compiled import CompiledGlobalModel
from plasma_global.core.domain import RecipeSegment
from plasma_global.core.result import series_unit
from plasma_global.input._schema_output import NetworkConfig, PlotConfig
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
    columns = yaml.safe_load((paths.directory / "series.yaml").read_text())["columns"]
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
        RecipeSegment("on", 0.0, 0.05, reduced_field_Td_by_zone={"plasma": 1.0}),
        RecipeSegment("off", 0.05, 0.1, reduced_field_Td_by_zone={"plasma": 0.0}),
    )
    chemistry = replace(
        compiled.chemistry, rate_evaluators=(lambda context: context.reduced_field_Td,)
    )
    model = CompiledGlobalModel(
        chemistry, compiled.model.zones, segments, compiled.model.electron_closure
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
    probe = next(path for path in figures if path.stem == "probe").read_text()
    assert "[eV]" in probe
    assert "[m^-3]" in probe
    composition = next(
        path for path in figures if path.stem == "composition"
    ).read_text()
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
    document = yaml.safe_load(path.read_text())
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


def test_surface_event_rates_are_per_area_and_counts_are_per_surface(
    tmp_path: Path,
) -> None:
    from plasma_global.examples import initialize_example

    path = initialize_example("smoke", tmp_path / "surface")
    document = yaml.safe_load(path.read_text())
    document["output"]["reaction_rates"] = True
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = simulate(load_case(path))
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


def test_missing_renderer_does_not_publish_an_incomplete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plasma_global.output import write_result_h5

    document = _document()
    document["output"] = {"networks": {"diagram": {"mode": "mechanism"}}}
    result = simulate(_case(tmp_path, document))
    monkeypatch.setattr(
        "plasma_global._result_network.shutil.which", lambda _name: None
    )
    with pytest.raises(RuntimeError, match="dot executable"):
        write_result(result, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()
    assert write_result_h5(tmp_path / "numerical-result.h5", result).is_file()


def test_units_do_not_rescale_a_known_physical_quantity(tmp_path: Path) -> None:
    document = _document()
    document["output"] = {
        "quantities": ["pressure_Pa"],
        "units": {"pressure_Pa[plasma]": "kPa"},
    }
    with pytest.raises(ValueError, match="canonical unit"):
        compile_case(_case(tmp_path, document))
    assert series_unit("film[wall]") == "m"
    assert series_unit("inventory[wall,F]") == "1"


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


def test_old_unknown_units_are_preserved_when_the_catalog_grows(tmp_path: Path) -> None:
    import h5py

    from plasma_global.core.result import SimulationResult
    from plasma_global.output import write_result_h5

    result = SimulationResult(
        np.array([0.0]),
        np.array([[1.0]]),
        ("film[wall]",),
        observables={"pressure_Pa": np.array([2.0])},
    )
    path = write_result_h5(tmp_path / "legacy.h5", result)
    with h5py.File(path, "a") as h5:
        h5["state/values"].attrs["column_units"] = ["unspecified"]
        h5["observables/pressure_Pa"].attrs["unit"] = "unspecified"
    loaded = read_result_h5(path)
    assert series_unit("film[wall]", loaded.metadata["series_units"]) == "unspecified"
    assert series_unit("pressure_Pa", loaded.metadata["series_units"]) == "unspecified"
    np.testing.assert_array_equal(loaded.state, result.state)
