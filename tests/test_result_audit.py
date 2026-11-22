from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from plasma_global import compile_case, load_case, simulate
from plasma_global.audit import audit_result
from plasma_global.core.result import SimulationResult, SimulationStatus


def test_audit_reports_status_required_series_and_nonnegativity() -> None:
    result = SimulationResult(
        time_s=np.array([0.0, 1.0]),
        state=np.array([[1.0, 2.0], [-1.0e-2, 3.0]]),
        state_labels=("density", "energy"),
        observables={"power": np.array([1.0, 2.0])},
        status=SimulationStatus.failed("integrator diverged"),
    )
    report = audit_result(
        result,
        required_series=("missing",),
        nonnegative_series=("density",),
        negative_tolerance=1.0e-3,
    )

    codes = {issue.code for issue in report.issues}
    assert report.passed is False
    assert {
        "SIMULATION_NOT_SUCCESSFUL",
        "REQUIRED_SERIES_MISSING",
        "NEGATIVE_SERIES_VALUE",
    } <= codes


def test_audit_records_each_conservation_maximum_and_tolerance_failure() -> None:
    result = SimulationResult(
        time_s=np.array([0.0, 1.0, 2.0]),
        state=np.ones((3, 1)),
        state_labels=("density",),
    )
    residual = np.array([0.0, -2.0e-5, 1.0e-5])
    before = residual.copy()

    report = audit_result(
        result,
        conservation_residuals={"particles": residual, "charge": np.asarray(3.0e-8)},
        conservation_tolerances={"particles": 1.0e-6},
        default_conservation_tolerance=1.0e-7,
    )

    np.testing.assert_array_equal(residual, before)
    assert report.max_conservation_residual == pytest.approx(2.0e-5)
    assert report.conservation_max_abs_residual == {
        "particles": pytest.approx(2.0e-5),
        "charge": pytest.approx(3.0e-8),
    }
    exceeded = [
        issue
        for issue in report.issues
        if issue.code == "CONSERVATION_RESIDUAL_EXCEEDED"
    ]
    assert [issue.quantity for issue in exceeded] == ["particles"]


def test_audit_rejects_invalid_control_tolerances() -> None:
    result = SimulationResult(
        time_s=np.array([0.0]),
        state=np.ones((1, 1)),
        state_labels=("density",),
    )

    with pytest.raises(ValueError, match="non-negative"):
        audit_result(result, default_conservation_tolerance=-1.0)
    with pytest.raises(ValueError, match="negative_tolerance"):
        audit_result(result, negative_tolerance=-1.0)
    with pytest.raises(ValueError, match="missing residuals"):
        audit_result(
            result,
            conservation_residuals={"charge": np.zeros(1)},
            conservation_tolerances={"chrage": 1e-9},
        )
    with pytest.raises(ValueError, match="require explicit tolerances"):
        audit_result(result, conservation_residuals={"charge": np.ones(1)})


def test_model_assumptions_are_specific_and_omissions_remain_actionable() -> None:
    fixture = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"
    prepared = compile_case(load_case(fixture))
    assessment = prepared.assessment.to_dict()
    assert "classification" not in assessment
    assert any("Maxwellian" in item for item in assessment["assumptions"])
    assert any("charge neutrality" in item for item in assessment["assumptions"])
    assert any(
        issue.code == "MISSING_MOMENTUM_CROSS_SECTION" for issue in prepared.warnings
    )


def test_file_provenance_covers_case_chemistry_curves_and_electron_table() -> None:
    case_path = (
        Path(__file__).parents[1]
        / "plasma_global"
        / "examples"
        / "cases"
        / "zdplaskin_example2.yaml"
    )
    prepared = compile_case(load_case(case_path))
    provenance = prepared.metadata["provenance"]["input_files"]
    files = provenance["files"]
    names = {Path(path).name for path in files}

    assert {
        "zdplaskin_example2.yaml",
        "chemistry.yaml",
        "electrons.csv",
    } <= names
    assert provenance["algorithm"] == "sha256"
    assert all(len(record["sha256"]) == 64 for record in files.values())
    assert all(record["size_bytes"] > 0 for record in files.values())


def test_normal_simulation_records_input_and_software_provenance() -> None:
    fixture = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"

    result = simulate(load_case(fixture))

    assert result.status.success
    provenance = result.metadata["provenance"]
    input_files = provenance["input_files"]
    assert input_files["algorithm"] == "sha256"
    case_record = input_files["files"][str(fixture.resolve())]
    assert len(case_record["sha256"]) == 64
    assert case_record["size_bytes"] == fixture.stat().st_size
    assert case_record["roles"] == ("case",)

    software = provenance["software"]
    assert software["package"]["name"] == "plasma-global-model"
    assert software["package"]["version"]
    assert software["runtime"]["numpy"] == np.__version__
    assert len(software["source_tree"]["sha256"]) == 64
    assert software["source_tree"]["file_count"] > 0
    assert len(software["project_lock"]["sha256"]) == 64


def test_source_identity_tracks_contents_and_module_paths_across_copies(
    tmp_path: Path,
) -> None:
    from plasma_global._provenance import _source_fingerprint

    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    (left / "model.py").write_text("coefficient = 1.0\n", encoding="utf-8")
    shutil.copytree(left, right)
    initial = _source_fingerprint(left)
    assert initial == _source_fingerprint(right)
    (right / "model.py").write_text("coefficient = 2.0\n", encoding="utf-8")
    assert initial != _source_fingerprint(right)
    (right / "model.py").write_text("coefficient = 1.0\n", encoding="utf-8")
    (right / "model.py").rename(right / "other_model.py")
    assert initial != _source_fingerprint(right)
