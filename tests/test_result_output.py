from __future__ import annotations

import sys
from dataclasses import replace
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
            "effective_case_yaml": "schema_version: 3\ncase:\n  name: argon\n",
            "model_ids": {"electrons": "maxwellian", "solver": "BDF"},
            "provenance": {"source": "fixture", "include_chain": []},
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
        assert set(h5["metadata"]) == {
            "effective_case_yaml",
            "model_ids",
            "provenance",
        }
        assert set(h5["solver"]) == {
            "success",
            "statistics_yaml",
            "status_code",
            "status_message",
        }
        assert h5["state/values"].shape == (3, 2)
        assert h5["state/labels"].asstr()[...].tolist() == list(result.state_labels)
        assert set(h5["observables"]) == set(result.observables)
        assert (
            h5["metadata/effective_case_yaml"]
            .asstr()[()]
            .startswith("schema_version: 3")
        )

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
    assert to_plain_mapping(loaded.metadata) == {
        "effective_case_yaml": result.metadata["effective_case_yaml"],
        "model_ids": to_plain_mapping(result.metadata["model_ids"]),
        "provenance": to_plain_mapping(result.metadata["provenance"]),
    }


def test_result_h5_reader_rejects_nonfinite_numeric_artifact(tmp_path: Path) -> None:
    path = write_result_h5(tmp_path / "nonfinite.h5", _result())
    with h5py.File(path, "a") as h5:
        h5["state/values"][1, 0] = np.nan

    with pytest.raises(ValueError, match="state must contain only finite values"):
        read_result_h5(path)


def test_result_h5_round_trips_bohm_closure_provenance(tmp_path: Path) -> None:
    result = _result()
    metadata = to_plain_mapping(result.metadata)
    metadata["provenance"]["wall_transport_closure"] = {
        "bohm_h_factor": {
            "version": "direct-multiplier-v2",
            "surface_modes": {"wall": "numeric", "electrode": "auto"},
        }
    }

    loaded = read_result_h5(
        write_result_h5(
            tmp_path / "result.h5",
            replace(result, metadata=metadata),
        )
    )

    assert (
        to_plain_mapping(loaded.metadata)["provenance"]["wall_transport_closure"]
        == metadata["provenance"]["wall_transport_closure"]
    )


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
    assert summary["conservation"]["max_abs_residual"] == pytest.approx(2.0e-9)
    assert summary["audit"]["passed"] is True
    assert paths.audit_passed is True


def test_write_result_preserves_artifacts_and_reports_failed_audit(
    tmp_path: Path,
) -> None:
    result = _result()
    metadata = to_plain_mapping(result.metadata)
    metadata["default_conservation_tolerance"] = 1.0e-12

    paths = write_result(replace(result, metadata=metadata), tmp_path / "failed-audit")

    assert paths.audit_passed is False
    assert paths.result_h5.is_file()
    summary = yaml.safe_load(paths.summary_yaml.read_text(encoding="utf-8"))
    assert summary["audit"]["passed"] is False


def test_write_result_does_not_publish_half_bundle_on_serialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    old_h5 = directory / "result.h5"
    old_summary = directory / "summary.yaml"
    old_h5.write_bytes(b"old h5")
    old_summary.write_text("old summary\n", encoding="utf-8")

    def fail_summary(*_args: object, **_kwargs: object) -> Path:
        raise RuntimeError("summary serialization failed")

    monkeypatch.setattr(output_module, "write_summary_yaml", fail_summary)
    with pytest.raises(RuntimeError, match="summary serialization failed"):
        write_result(_result(), directory)

    assert old_h5.read_bytes() == b"old h5"
    assert old_summary.read_text(encoding="utf-8") == "old summary\n"
    assert {path.name for path in directory.iterdir()} == {
        "result.h5",
        "summary.yaml",
    }


def test_write_result_rolls_back_if_second_atomic_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    old_h5 = directory / "result.h5"
    old_summary = directory / "summary.yaml"
    old_h5.write_bytes(b"old h5")
    old_summary.write_text("old summary\n", encoding="utf-8")
    real_replace = output_module.os.replace
    public_replacements = 0

    def fail_second_public_replace(source: Path, target: Path) -> None:
        nonlocal public_replacements
        if Path(target).parent == directory:
            public_replacements += 1
            if public_replacements == 2:
                raise OSError("second atomic replace failed")
        real_replace(source, target)

    monkeypatch.setattr(output_module.os, "replace", fail_second_public_replace)
    with pytest.raises(OSError, match="second atomic replace failed"):
        write_result(_result(), directory)

    assert old_h5.read_bytes() == b"old h5"
    assert old_summary.read_text(encoding="utf-8") == "old summary\n"
    assert {path.name for path in directory.iterdir()} == {
        "result.h5",
        "summary.yaml",
    }


def test_simulation_persists_nonempty_runtime_conservation_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"
    from plasma_global.core.compiled import CompiledGlobalModel

    evaluations = 0
    original = CompiledGlobalModel.evaluate

    def counted_evaluate(self: object, *args: object, **kwargs: object) -> object:
        nonlocal evaluations
        evaluations += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CompiledGlobalModel, "evaluate", counted_evaluate)
    result = simulate(load_case(fixture))
    # This fixture requests no derived observables.  Standard quasineutral
    # diagnostics are algebraic identities and need no post-solve RHS replay.
    assert evaluations == 0
    paths = write_result(result, tmp_path / "run")
    loaded = read_result_h5(paths.result_h5)
    summary = yaml.safe_load(paths.summary_yaml.read_text(encoding="utf-8"))

    runtime = loaded.metadata["provenance"]["runtime_diagnostics"]
    maxima = runtime["conservation_max_abs_residual"]
    assert set(maxima) == {"charge_closure_normalized"}
    assert summary["conservation"]["max_abs_residual"] is not None
    assert summary["conservation"]["max_abs_residual_by_quantity"] == maxima


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
        ("format_version", 2, "Unsupported result HDF5 format version 2"),
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
        ("metadata/model_ids", "- invalid\n", "model_ids and provenance"),
        ("metadata/provenance", "- invalid\n", "model_ids and provenance"),
        ("solver/statistics_yaml", "- invalid\n", "must contain a mapping"),
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


def test_summary_defaults_to_no_implicit_final_or_conservation_quantities() -> None:
    result = SimulationResult(
        time_s=np.array([0.0]),
        state=np.array([[2.0]]),
        state_labels=("density",),
    )

    summary = build_summary(result)

    assert summary["final"] == {}
    assert summary["conservation"]["max_abs_residual"] is None
    assert summary["conservation"]["max_abs_residual_by_quantity"] == {}


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
