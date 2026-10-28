from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

import plasma_global.output as output_module
from plasma_global import load_case, simulate
from plasma_global.core.result import (
    SimulationResult,
    SimulationStatus,
    to_plain_mapping,
)
from plasma_global.output import (
    audit_result_h5,
    build_summary,
    export_result_csv,
    plot_result_h5,
    read_result_csv,
    read_result_h5,
    write_result,
    write_result_csv,
    write_result_h5,
)


def test_plotting_reports_the_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(output_module, "read_result_h5", lambda _path: _result())
    monkeypatch.setitem(sys.modules, "matplotlib", None)

    with pytest.raises(RuntimeError, match="optional dependency"):
        plot_result_h5("result.h5", tmp_path)


def _result() -> SimulationResult:
    return SimulationResult(
        time_s=np.array([0.0, 1.0e-6, 2.0e-6]),
        state=np.array(
            [
                [1.0e15, 2.0],
                [2.0e15, 2.5],
                [3.0e15, 3.0],
            ]
        ),
        state_labels=("electron_density_m3", "mean_energy_eV"),
        observables={
            "absorbed_power_W": np.array([10.0, 12.0, 14.0]),
            "pressure_Pa": np.asarray(5.0),
            "particle_residual": np.array([0.0, 1.0e-9, -2.0e-9]),
        },
        status=SimulationStatus.completed("integration finished"),
        solver_stats={"nfev": 42, "method": "BDF"},
        metadata={
            "effective_case_yaml": "schema_version: 4\ncase:\n  name: argon\n",
            "model_ids": {"electrons": "maxwellian", "solver": "BDF"},
            "provenance": {"source": "fixture", "input_files": {"files": {}}},
            "summary_series": ["electron_density_m3", "absorbed_power_W"],
            "conservation_observables": ["particle_residual"],
            "default_conservation_tolerance": 1.0e-8,
        },
    )


def test_simulation_result_is_time_major_immutable_and_named() -> None:
    result = _result()

    assert result.state.shape == (3, 2)
    assert result.n_times == 3
    assert result.n_states == 2
    np.testing.assert_array_equal(result.series("mean_energy_eV"), [2.0, 2.5, 3.0])
    np.testing.assert_array_equal(result.series("absorbed_power_W"), [10.0, 12.0, 14.0])
    assert result.final_value("pressure_Pa") == 5.0
    assert result.state.flags.writeable is False
    assert result.series("absorbed_power_W").flags.writeable is False

    with pytest.raises(ValueError, match="time-major shape"):
        SimulationResult(
            time_s=np.array([0.0, 1.0]),
            state=np.zeros((1, 2)),
            state_labels=("a", "b"),
        )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_simulation_result_rejects_nonfinite_numeric_content(nonfinite: float) -> None:
    with pytest.raises(ValueError, match="state must contain only finite values"):
        SimulationResult(
            time_s=np.array([0.0]),
            state=np.array([[nonfinite]]),
            state_labels=("density",),
        )

    with pytest.raises(ValueError, match=r"observable.*finite values"):
        SimulationResult(
            time_s=np.array([0.0]),
            state=np.array([[1.0]]),
            state_labels=("density",),
            observables={"power": np.array([nonfinite])},
        )

    with pytest.raises(ValueError, match=r"metadata.*finite numeric values"):
        SimulationResult(
            time_s=np.array([0.0]),
            state=np.array([[1.0]]),
            state_labels=("density",),
            metadata={"nested": {"value": nonfinite}},
        )


def test_result_h5_has_only_the_fixed_public_layout_and_round_trips(
    tmp_path: Path,
) -> None:
    result = _result()
    path = write_result_h5(tmp_path / "result.h5", result)

    with h5py.File(path, "r") as h5:
        assert set(h5) == {"metadata", "observables", "solver", "state", "time_s"}
        assert set(h5["metadata"]) == {"values_yaml"}
        assert set(h5["solver"]) == {
            "success",
            "statistics_yaml",
            "status_code",
            "status_message",
        }
        assert h5["state/values"].shape == (3, 2)
        assert h5["state/labels"].asstr()[...].tolist() == list(result.state_labels)
        assert h5["state/values"].attrs["column_units"].tolist() == [
            "m^-3",
            "eV",
        ]
        assert h5["observables/absorbed_power_W"].attrs["unit"] == "W"
        assert set(h5["observables"]) == set(result.observables)

    loaded = read_result_h5(path)
    np.testing.assert_array_equal(loaded.time_s, result.time_s)
    np.testing.assert_array_equal(loaded.state, result.state)
    assert loaded.state_labels == result.state_labels
    for name in result.observables:
        np.testing.assert_array_equal(loaded.series(name), result.series(name))
    assert loaded.status == result.status
    assert to_plain_mapping(loaded.solver_stats) == to_plain_mapping(
        result.solver_stats
    )
    assert to_plain_mapping(loaded.metadata) == to_plain_mapping(result.metadata)
    assert build_summary(loaded) == build_summary(result)


def test_result_h5_reader_rejects_nonfinite_numeric_artifact(tmp_path: Path) -> None:
    path = write_result_h5(tmp_path / "nonfinite.h5", _result())
    with h5py.File(path, "a") as h5:
        h5["state/values"][1, 0] = np.nan

    with pytest.raises(ValueError, match="state must contain only finite values"):
        read_result_h5(path)


def test_result_h5_round_trips_input_and_software_provenance(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"
    result = simulate(load_case(fixture))

    loaded = read_result_h5(write_result_h5(tmp_path / "result.h5", result))

    provenance = to_plain_mapping(loaded.metadata)["provenance"]
    case_record = provenance["input_files"]["files"][str(fixture.resolve())]
    assert len(case_record["sha256"]) == 64
    assert case_record["size_bytes"] == fixture.stat().st_size
    assert provenance["software"]["runtime"]["numpy"] == np.__version__


def test_write_result_always_writes_exactly_h5_and_summary(tmp_path: Path) -> None:
    result = _result()
    paths = write_result(result, tmp_path / "run")

    assert {path.name for path in paths.directory.iterdir()} == {
        "result.h5",
        "summary.yaml",
    }
    summary = yaml.safe_load(paths.summary_yaml.read_text(encoding="utf-8"))
    assert summary["model_ids"] == {
        "electrons": "maxwellian",
        "solver": "BDF",
    }
    assert summary["final"] == {
        "electron_density_m3": 3.0e15,
        "absorbed_power_W": 14.0,
    }
    assert summary["assessment"] is None


def test_completed_run_is_immutable_and_can_reproduce_its_summary(
    tmp_path: Path,
) -> None:
    paths = write_result(_result(), tmp_path / "run")
    original = {path.name: path.read_bytes() for path in paths.directory.iterdir()}
    with pytest.raises(FileExistsError, match="choose a new output directory"):
        write_result(_result(), paths.directory)
    assert original == {
        path.name: path.read_bytes() for path in paths.directory.iterdir()
    }
    loaded = read_result_h5(paths.result_h5)
    assert build_summary(loaded) == yaml.safe_load(
        paths.summary_yaml.read_text(encoding="utf-8")
    )


def test_normal_run_and_case_audit_publish_the_same_assessment(tmp_path: Path) -> None:
    from plasma_global.audit import audit_case

    fixture = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"
    case = load_case(fixture)
    result = simulate(case)
    paths = write_result(result, tmp_path / "run")
    loaded = read_result_h5(paths.result_h5)
    assessment = to_plain_mapping(result.metadata)["assessment"]
    assert assessment == audit_case(case).assessment.to_dict()
    assert assessment == build_summary(loaded)["assessment"]
    assert assessment["issues"][0]["code"] == "MISSING_MOMENTUM_CROSS_SECTION"
    assert to_plain_mapping(result.metadata) == to_plain_mapping(loaded.metadata)


def test_write_result_does_not_publish_half_bundle_on_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"

    def fail_summary(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("summary serialization failed")

    monkeypatch.setattr(output_module, "write_summary_yaml", fail_summary)
    with pytest.raises(RuntimeError, match="summary serialization failed"):
        write_result(_result(), directory)

    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_result_reader_rejects_legacy_or_extra_hdf5_layout(tmp_path: Path) -> None:
    path = write_result_h5(tmp_path / "result.h5", _result())
    with h5py.File(path, "a") as h5:
        h5.create_group("status")

    with pytest.raises(ValueError, match="layout mismatch"):
        read_result_h5(path)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("format", np.bytes_("legacy_result"), "is not a plasma_global_result"),
        ("format_version", 1, "Unsupported result HDF5 format version 1"),
        ("format_version", 2.9, "format_version must be an integer"),
        ("state_layout", "state-major", "state layout must be time-major"),
    ],
)
def test_result_reader_validates_every_hdf5_header_contract(
    tmp_path: Path, attribute: str, value: object, message: str
) -> None:
    path = write_result_h5(tmp_path / f"invalid-{attribute}.h5", _result())
    with h5py.File(path, "a") as h5:
        h5.attrs[attribute] = value

    with pytest.raises(ValueError, match=message):
        read_result_h5(path)


@pytest.mark.parametrize(
    ("dataset_path", "content", "message"),
    [
        ("metadata/values_yaml", "- invalid\n", "metadata must be a mapping"),
        ("solver/statistics_yaml", "- invalid\n", "must contain a mapping"),
        ("solver/statistics_yaml", "false\n", "must contain a mapping"),
    ],
)
def test_result_reader_rejects_non_mapping_hdf5_documents(
    tmp_path: Path, dataset_path: str, content: str, message: str
) -> None:
    path = write_result_h5(tmp_path / "invalid-document.h5", _result())
    with h5py.File(path, "a") as h5:
        del h5[dataset_path]
        h5.create_dataset(dataset_path, data=content)

    with pytest.raises(TypeError, match=message):
        read_result_h5(path)


def test_result_reader_rejects_non_dataset_observable(tmp_path: Path) -> None:
    path = write_result_h5(tmp_path / "invalid-observable.h5", _result())
    with h5py.File(path, "a") as h5:
        del h5["observables/absorbed_power_W"]
        h5.create_group("observables/absorbed_power_W")

    with pytest.raises(TypeError, match="must be a dataset"):
        read_result_h5(path)


def test_summary_does_not_invent_a_case_assessment() -> None:
    result = SimulationResult(
        time_s=np.array([0.0]),
        state=np.array([[2.0]]),
        state_labels=("density",),
    )

    summary = build_summary(result)

    assert summary["final"] == {}
    assert summary["assessment"] is None


def test_csv_is_created_only_by_explicit_export(tmp_path: Path) -> None:
    result = _result()
    h5_path = write_result_h5(tmp_path / "result.h5", result)
    csv_path = export_result_csv(h5_path, tmp_path / "export.csv")
    loaded = read_result_csv(
        csv_path,
        status=result.status,
        solver_stats=result.solver_stats,
        metadata=result.metadata,
    )

    np.testing.assert_array_equal(loaded.state, result.state)
    assert loaded.state_labels == result.state_labels
    assert loaded.series("pressure_Pa").shape == ()
    for name in result.observables:
        np.testing.assert_array_equal(loaded.series(name), result.series(name))

    direct = write_result_csv(tmp_path / "direct.csv", result)
    assert direct.is_file()
    report = audit_result_h5(
        h5_path,
        conservation_observables=("particle_residual",),
        default_conservation_tolerance=1.0e-8,
    )
    assert report.passed


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "Result CSV is empty"),
        ("elapsed_s,state:x\n0,1\n", "first column must be 'time_s'"),
        ("time_s,state:x,state:x\n0,1,2\n", "column names must be unique"),
        ("time_s,state:x\n0\n", "rows do not match its header"),
        ("time_s,state:x\n0,not-a-number\n", "contains a non-numeric cell"),
        (
            "time_s,observable_scalar:pressure\n0,1\n1,2\n",
            "Invalid scalar observable column 'pressure'",
        ),
        ("time_s,unknown:value\n0,1\n", "Unsupported result CSV column"),
    ],
)
def test_result_csv_reader_rejects_malformed_public_artifacts(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_result_csv(path)


def test_result_csv_reader_rejects_nonfinite_artifact(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.csv"
    path.write_text("time_s,state:x\n0,nan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="state must contain only finite values"):
        read_result_csv(path)


def test_result_csv_reader_preserves_empty_state_and_default_status(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-result.csv"
    path.write_text("time_s,state:density\n", encoding="utf-8")

    result = read_result_csv(path)

    assert result.time_s.shape == (0,)
    assert result.state.shape == (0, 1)
    assert result.state_labels == ("density",)
    assert result.status.code == "imported_csv"
    assert result.status.success is None


@pytest.mark.parametrize("success", [True, False, None])
def test_execution_status_round_trips_without_boolean_coercion(
    tmp_path: Path, success: bool | None
) -> None:
    from dataclasses import replace

    result = replace(_result(), status=SimulationStatus(success, "imported_csv"))
    loaded = read_result_h5(write_result_h5(tmp_path / "status.h5", result))
    assert loaded.status.success is success
    assert build_summary(loaded)["status"]["success"] is success


@pytest.mark.parametrize("success", ["false", 0, 1])
def test_execution_status_rejects_non_boolean_values(success: object) -> None:
    with pytest.raises(TypeError, match="bool or None"):
        SimulationStatus(success=success)


@pytest.mark.parametrize(
    "content",
    [
        "time_s,observable_scalar:rate\n0,1e-20\n1,2e-20\n",
        "time_s,observable:rate,observable_scalar:rate\n0,1\u002c2\n1,1,2\n",
        "time_s,observable_scalar:rate,observable:rate\n0,1,2\n1,1,2\n",
        "time_s,observable:rate,observable: rate\n0,1,2\n1,1,2\n",
    ],
)
def test_csv_never_discards_variation_or_colliding_names(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "ambiguous.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"scalar observable|Duplicate observable|whitespace"
    ):
        read_result_csv(path)


def test_numeric_csv_import_cannot_claim_a_failed_run_succeeded(tmp_path: Path) -> None:
    from dataclasses import replace

    from plasma_global.audit import audit_result

    failed = replace(_result(), status=SimulationStatus.failed("solver failed"))
    loaded = read_result_csv(write_result_csv(tmp_path / "failed.csv", failed))
    assert loaded.status.success is None
    assert not audit_result(loaded).passed
    assert (
        read_result_csv(tmp_path / "failed.csv", status=failed.status).status
        == failed.status
    )


def test_hdf5_rejects_text_that_would_coerce_to_success(tmp_path: Path) -> None:
    path = write_result_h5(tmp_path / "status.h5", _result())
    with h5py.File(path, "a") as h5:
        del h5["solver/success"]
        h5.create_dataset("solver/success", data="false")
    with pytest.raises(TypeError, match="boolean dataset"):
        read_result_h5(path)


@pytest.mark.parametrize(
    "update",
    [
        {"metadata": {1: "numeric key", "1": "string key"}},
        {"metadata": []},
        {"state_labels": "x"},
        {"observables": []},
    ],
)
def test_result_rejects_coercions_that_lose_names_or_content(update: dict) -> None:
    values = {
        "time_s": np.array([0.0]),
        "state": np.array([[1.0]]),
        "state_labels": ("x",),
    }
    with pytest.raises(TypeError, match=r"string|mapping|sequence"):
        SimulationResult(**{**values, **update})


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"message": 1}, TypeError, "message must be a string"),
        (
            {"success": False, "code": "completed"},
            ValueError,
            "completed status requires",
        ),
        ({"success": True, "code": "failed"}, ValueError, "failed status requires"),
    ],
)
def test_status_rejects_contradictory_outcomes(
    values: dict, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        SimulationStatus(**values)


def test_hdf5_metadata_never_silently_overwrites_duplicate_keys(tmp_path: Path) -> None:
    path = write_result_h5(tmp_path / "duplicate.h5", _result())
    with h5py.File(path, "a") as h5:
        h5["metadata/values_yaml"][()] = "setting: 1\nsetting: 2\n"
    with pytest.raises(ValueError, match="duplicate key"):
        read_result_h5(path)


def test_hdf5_v2_remains_readable_without_allowing_unknown_status(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    path = write_result_h5(tmp_path / "v2.h5", _result())
    with h5py.File(path, "a") as h5:
        h5.attrs["format_version"] = 2
    assert read_result_h5(path).status.success is True
    unknown = replace(_result(), status=SimulationStatus(None, "imported_csv"))
    path = write_result_h5(tmp_path / "unknown.h5", unknown)
    with h5py.File(path, "a") as h5:
        h5.attrs["format_version"] = 2
    with pytest.raises(ValueError, match="requires result format version 3"):
        read_result_h5(path)


def test_custom_units_are_explicit_and_never_inferred_from_ambiguous_suffixes(
    tmp_path: Path,
) -> None:
    result = SimulationResult(
        time_s=np.array([0.0]),
        state=np.array([[0.001, 2.0]]),
        state_labels=("volume_m3", "custom_m3"),
        observables={"flux_m2_s": np.array([3.0])},
        metadata={"series_units": {"volume_m3": "m^3", "flux_m2_s": "m^-2 s^-1"}},
    )
    path = write_result_h5(tmp_path / "units.h5", result)
    with h5py.File(path, "r") as h5:
        assert h5["state/values"].attrs["column_units"].tolist() == [
            "m^3",
            "unspecified",
        ]
        assert h5["observables/flux_m2_s"].attrs["unit"] == "m^-2 s^-1"
    loaded = read_result_h5(path)
    assert to_plain_mapping(loaded.metadata) == to_plain_mapping(result.metadata)


@pytest.mark.parametrize(
    ("dataset", "attribute", "value"),
    [
        ("state/values", "column_units", ["eV", "m^-3"]),
        ("observables/absorbed_power_W", "unit", "eV"),
    ],
)
def test_hdf5_rejects_a_unit_that_changes_the_quantity_meaning(
    tmp_path: Path, dataset: str, attribute: str, value: object
) -> None:
    path = write_result_h5(tmp_path / "units.h5", _result())
    with h5py.File(path, "a") as h5:
        h5[dataset].attrs[attribute] = value
    with pytest.raises(ValueError, match="disagree"):
        read_result_h5(path)


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ({"electron_density_m3": "cm^-3"}, "requires canonical unit"),
        ({"typo": "m^-3"}, "unknown series"),
        ([], "must be a mapping"),
    ],
)
def test_unit_declarations_cannot_rename_or_rescale_quantities(
    declared: object, message: str
) -> None:
    from dataclasses import replace

    with pytest.raises((TypeError, ValueError), match=message):
        replace(_result(), metadata={"series_units": declared})


def test_plot_series_keep_distinct_names_and_explicit_units(tmp_path: Path) -> None:
    import matplotlib

    result = SimulationResult(
        time_s=np.array([0.0, 1.0]),
        state=np.array([[1.0, 2.0], [3.0, 4.0]]),
        state_labels=("n[plasma,Ar]", "n[plasma,Ar+]"),
        observables={"flux": np.array([5.0, 6.0])},
        metadata={"series_units": {"flux": "m^-2 s^-1"}},
    )
    source = write_result_h5(tmp_path / "result.h5", result)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        paths = plot_result_h5(
            source,
            tmp_path / "plots",
            series=(*result.state_labels, "flux"),
            image_format="svg",
        )
    assert len(paths) == len(set(paths)) == 3
    assert len(list((tmp_path / "plots").iterdir())) == 3
    text = paths[-1].read_text(encoding="utf-8")
    assert "flux [m^-2 s^-1]" in text
    assert read_result_h5(source).state_labels == result.state_labels


@pytest.mark.parametrize(
    "series", [("mean_energy_eV", "missing"), ("mean_energy_eV", "mean_energy_eV")]
)
def test_invalid_plot_selection_never_publishes_partial_output(
    tmp_path: Path, series: tuple[str, ...]
) -> None:
    source = write_result_h5(tmp_path / "result.h5", _result())
    directory = tmp_path / "plots"
    with pytest.raises(
        (KeyError, ValueError), match=r"Unknown result series|duplicates"
    ):
        plot_result_h5(source, directory, series=series)
    assert not directory.exists()


def test_existing_plot_directory_and_source_hdf5_are_never_overwritten(
    tmp_path: Path,
) -> None:
    source = write_result_h5(tmp_path / "result.h5", _result())
    original = source.read_bytes()
    with pytest.raises(FileExistsError):
        export_result_csv(source, source)
    assert source.read_bytes() == original
    directory = tmp_path / "plots"
    directory.mkdir()
    marker = directory / "original.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="choose a new output directory"):
        plot_result_h5(source, directory)
    assert list(directory.iterdir()) == [marker]
    assert marker.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ([1, 1], TypeError, "must be a string"),
        (["mean_energy_eV", "mean_energy_eV"], ValueError, "duplicates"),
        ("mean_energy_eV", TypeError, "sequence of names"),
        (["missing"], ValueError, "unknown series"),
    ],
)
def test_summary_selection_preserves_names_without_coercion_or_deduplication(
    selection: object, error: type[Exception], message: str
) -> None:
    from dataclasses import replace

    with pytest.raises(error, match=message):
        replace(_result(), metadata={"summary_series": selection})
