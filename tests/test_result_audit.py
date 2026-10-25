from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global import load_case, simulate
from plasma_global._audit_classification import case_classification
from plasma_global._provenance import collect_file_provenance
from plasma_global.audit import audit_case, audit_result
from plasma_global.chemistry.data import load_chemistry
from plasma_global.core.result import SimulationResult, SimulationStatus
from plasma_global.input._resolved_case import SurfaceKineticsConfig


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


def test_standard_classification_is_an_explicit_minimal_allow_list() -> None:
    fixture = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"
    case = load_case(fixture)
    chemistry = load_chemistry(case.chemistry.manifest)

    def replace_model(field: str, kind: str):
        selected = getattr(case.models, field).model_copy(update={"kind": kind})
        models = case.models.model_copy(update={field: selected})
        return case.model_copy(update={"models": models})

    dc_model = case.reactor.power_ports[0].model.model_copy(
        update={"kind": "dc_series"}
    )
    dc_port = case.reactor.power_ports[0].model_copy(update={"model": dc_model})
    dc_reactor = case.reactor.model_copy(update={"power_ports": [dc_port]})
    dc_case = case.model_copy(update={"reactor": dc_reactor})

    surface_models = case.models.model_copy(
        update={"surface_kinetics": SurfaceKineticsConfig()}
    )
    surface_case = case.model_copy(update={"models": surface_models})

    prescribed_wall = case.reactor.surfaces[0].wall_transport.model_copy(
        update={"kind": "prescribed_frequency"}
    )
    prescribed_surface = case.reactor.surfaces[0].model_copy(
        update={"wall_transport": prescribed_wall}
    )
    wall_reactor = case.reactor.model_copy(update={"surfaces": [prescribed_surface]})
    wall_case = case.model_copy(update={"reactor": wall_reactor})

    assert case_classification(case, chemistry) == "standard"
    for candidate in (
        replace_model("electrons", "table"),
        replace_model("electron_closure", "local_field"),
        replace_model("electron_density", "experimental.prescribed_profile"),
        replace_model("gas_energy", "evolved"),
        dc_case,
        surface_case,
        wall_case,
    ):
        assert case_classification(candidate, chemistry) == ("experimental")

    surface_chemistry = replace(
        chemistry,
        surface_reactions=(chemistry.boundary_reactions[0],),
    )
    assert case_classification(case, surface_chemistry) == ("experimental")
    no_boundary_chemistry = replace(chemistry, boundary_reactions=())
    nonapplicable_boundary_chemistry = replace(
        chemistry,
        boundary_reactions=(
            replace(chemistry.boundary_reactions[0], surfaces=("other_wall",)),
        ),
    )
    assert case_classification(case, no_boundary_chemistry) == ("experimental")
    assert case_classification(case, nonapplicable_boundary_chemistry) == (
        "experimental"
    )

    positive_ion = next(item for item in chemistry.species if item.charge > 0)
    no_positive_chemistry = replace(
        chemistry,
        species=tuple(
            replace(item, charge=0) if item.id == positive_ion.id else item
            for item in chemistry.species
        ),
    )
    multiple_positive_chemistry = replace(
        chemistry,
        species=(*chemistry.species, replace(positive_ion, id="Ar2_plus")),
    )
    assert case_classification(case, no_positive_chemistry) == ("experimental")
    assert case_classification(case, multiple_positive_chemistry) == ("experimental")

    crane_path = (
        Path(__file__).parents[1]
        / "plasma_global"
        / "assets"
        / "examples"
        / "cases"
        / "crane_two_reaction_argon.yaml"
    )
    crane = load_case(crane_path)
    crane_chemistry = load_chemistry(crane.chemistry.manifest)
    # CRANE remains a valuable scalar cross-code regression, but its published
    # fixed electron-rate coefficients are not the standard real-XS closure.
    assert case_classification(crane, crane_chemistry) == ("experimental")


def test_case_audit_classifies_experimental_chemistry_rate(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parent / "fixtures" / "minimal"
    fixture = tmp_path / "experimental-rate"
    shutil.copytree(source, fixture)
    manifest = fixture / "chemistry/chemistry.yaml"
    document = yaml.safe_load(manifest.read_text())
    document["gas_reactions"] = [
        {
            "id": "recombine",
            "equation": "Ar_plus + e -> Ar",
            "rate": {
                "kind": "experimental.electron_temperature_power_law",
                "A": 1e-20,
                "reference_temperature_K": 300.0,
                "exponent": -0.5,
            },
        }
    ]
    manifest.write_text(yaml.safe_dump(document), encoding="utf-8")

    report = audit_case(load_case(fixture / "case.yaml"))
    experimental = [
        issue for issue in report.issues if issue.code == "EXPERIMENTAL_MODEL"
    ]

    assert report.classification == "experimental"
    assert [issue.quantity for issue in experimental] == ["case"]


def test_file_provenance_covers_case_chemistry_curves_and_electron_table() -> None:
    case_path = (
        Path(__file__).parents[1]
        / "plasma_global"
        / "assets"
        / "examples"
        / "cases"
        / "zdplaskin_example2.yaml"
    )
    case = load_case(case_path)
    chemistry = load_chemistry(case.chemistry.manifest)

    provenance = collect_file_provenance(case, chemistry)
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
