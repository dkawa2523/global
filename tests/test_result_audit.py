from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import yaml

from plasma_global import _provenance as provenance_module
from plasma_global import audit as audit_module
from plasma_global import load_case, simulate
from plasma_global.audit import audit_case, audit_result, collect_file_provenance
from plasma_global.chemistry.data import load_chemistry
from plasma_global.core.result import SimulationResult, SimulationStatus
from plasma_global.input.schema import SurfaceKineticsConfig


def test_audit_reports_status_finiteness_required_series_and_nonnegativity() -> None:
    result = SimulationResult(
        time_s=np.array([0.0, 1.0]),
        state=np.array([[1.0, 2.0], [-1.0e-2, 3.0]]),
        state_labels=("density", "energy"),
        observables={"power": np.array([1.0, 2.0])},
        status=SimulationStatus.failed("integrator diverged"),
    )
    # The public constructor rejects non-finite content. The audit still
    # diagnoses an old in-memory result or an object corrupted after loading.
    legacy_state = np.array([[1.0, 2.0], [-1.0e-2, np.nan]])
    legacy_power = np.array([1.0, np.inf])
    legacy_state.setflags(write=False)
    legacy_power.setflags(write=False)
    object.__setattr__(result, "state", legacy_state)
    object.__setattr__(result, "observables", MappingProxyType({"power": legacy_power}))

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
        "STATE_NONFINITE",
        "OBSERVABLE_NONFINITE",
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


def test_case_audit_reports_missing_momentum_cross_section_target() -> None:
    fixture = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"

    report = audit_case(load_case(fixture))
    missing = [
        issue
        for issue in report.issues
        if issue.code == "MISSING_MOMENTUM_CROSS_SECTION"
    ]

    assert [(issue.level, issue.quantity) for issue in missing] == [("WARNING", "Ar")]
    assert report.model_ids["elastic_heating"] is None
    assert report.provenance["missing_momentum_cross_section_targets"] == ["Ar"]
    assert report.provenance["wall_transport_closure"] == {
        "bohm_h_factor": {
            "version": "direct-multiplier-v2",
            "surface_modes": {"wall": "numeric"},
        }
    }
    assert report.simulation["status"]["success"] is True
    assert report.simulation["time_count"] > 1
    assert report.classification == "standard"
    assert report.production_qualified is False
    assert report.conservation_max_abs_residual[
        "charge_closure_normalized"
    ] == pytest.approx(0.0)
    assert "electron_energy_ledger_normalized" in (report.conservation_max_abs_residual)
    input_files = report.provenance["input_files"]["files"]
    assert str(fixture.resolve()) in input_files
    assert len(input_files[str(fixture.resolve())]["sha256"]) == 64
    assert "input_files:" in yaml.safe_dump(
        report.to_dict(), sort_keys=False, allow_unicode=True
    )
    assert report.passed


def test_standard_classification_is_an_explicit_minimal_allow_list() -> None:
    fixture = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"
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

    assert audit_module._case_classification(case, chemistry) == "standard"
    for candidate in (
        replace_model("electrons", "table"),
        replace_model("electron_closure", "local_field"),
        replace_model("electron_density", "experimental.prescribed_profile"),
        replace_model("gas_energy", "evolved"),
        dc_case,
        surface_case,
        wall_case,
    ):
        assert audit_module._case_classification(candidate, chemistry) == (
            "experimental"
        )

    surface_chemistry = replace(
        chemistry,
        surface_reactions=(chemistry.boundary_reactions[0],),
    )
    assert audit_module._case_classification(case, surface_chemistry) == (
        "experimental"
    )
    no_boundary_chemistry = replace(chemistry, boundary_reactions=())
    nonapplicable_boundary_chemistry = replace(
        chemistry,
        boundary_reactions=(
            replace(chemistry.boundary_reactions[0], surfaces=("other_wall",)),
        ),
    )
    assert audit_module._case_classification(case, no_boundary_chemistry) == (
        "experimental"
    )
    assert audit_module._case_classification(
        case, nonapplicable_boundary_chemistry
    ) == ("experimental")

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
    assert audit_module._case_classification(case, no_positive_chemistry) == (
        "experimental"
    )
    assert audit_module._case_classification(case, multiple_positive_chemistry) == (
        "experimental"
    )

    crane_path = (
        Path(__file__).parents[1]
        / "examples"
        / "v3"
        / "cases"
        / "crane_two_reaction_argon.yaml"
    )
    crane = load_case(crane_path)
    crane_chemistry = load_chemistry(crane.chemistry.manifest)
    # CRANE remains a valuable scalar cross-code regression, but its published
    # fixed electron-rate coefficients are not the standard real-XS closure.
    assert audit_module._case_classification(crane, crane_chemistry) == ("experimental")


def test_case_audit_classifies_experimental_chemistry_rate(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parent / "fixtures" / "v3_minimal"
    fixture = tmp_path / "experimental-rate"
    shutil.copytree(source, fixture)
    (fixture / "gas_reactions.csv").write_text(
        "id,equation,rate_model\n"
        "recombine,Ar_plus + e -> Ar,experimental_recombination\n",
        encoding="utf-8",
    )
    (fixture / "rate_models.yaml").write_text(
        "rate_models:\n"
        "  experimental_recombination:\n"
        "    kind: experimental.electron_temperature_power_law\n"
        "    A: 1.0e-20\n"
        "    reference_temperature_K: 300.0\n"
        "    exponent: -0.5\n",
        encoding="utf-8",
    )

    report = audit_case(load_case(fixture / "case.yaml"))
    experimental = [
        issue for issue in report.issues if issue.code == "EXPERIMENTAL_MODEL"
    ]

    assert report.classification == "experimental"
    assert report.production_qualified is False
    assert [issue.quantity for issue in experimental] == [
        "experimental.electron_temperature_power_law"
    ]


def test_file_provenance_covers_case_chemistry_curves_and_electron_table() -> None:
    case_path = (
        Path(__file__).parents[1]
        / "examples"
        / "v3"
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
        "species.csv",
        "gas_reactions.csv",
        "rate_models.yaml",
        "cross_sections.yaml",
        "xs_zdp_ar_ionization.csv",
        "zdplaskin_example2_eovern_rates.h5",
    } <= names
    assert provenance["algorithm"] == "sha256"
    assert all(len(record["sha256"]) == 64 for record in files.values())
    assert all(record["size_bytes"] > 0 for record in files.values())


def test_case_audit_closes_evolved_heavy_and_electron_energy_ledgers() -> None:
    case_path = Path(__file__).parents[1] / "examples" / "v3" / "cases" / "smoke.yaml"

    report = audit_case(load_case(case_path))

    assert report.passed
    assert report.classification == "experimental"
    assert report.production_qualified is False
    assert (
        report.conservation_max_abs_residual["electron_energy_ledger_normalized"]
        < 1.0e-12
    )
    assert (
        report.conservation_max_abs_residual["heavy_energy_ledger_normalized"] < 1.0e-12
    )
    assert report.conservation_max_abs_residual["particle_ledger_normalized"] < 1.0e-12


def test_normal_simulation_records_input_and_software_provenance() -> None:
    fixture = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"

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
    assert set(software["runtime"]) == {"python", "numpy", "scipy", "h5py"}
    assert software["git"]["revision"] is None or len(software["git"]["revision"]) == 40
    assert software["git"]["dirty"] in {True, False, None}


def test_normal_simulation_does_not_reopen_inputs_for_audit_checksums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "v3_minimal" / "case.yaml"
    calls: list[Path] = []
    checksum = provenance_module._file_sha256

    def recorded_checksum(path: Path) -> tuple[str, int]:
        calls.append(path)
        return checksum(path)

    monkeypatch.setattr(provenance_module, "_file_sha256", recorded_checksum)
    result = simulate(load_case(fixture))
    compilation_calls = tuple(calls)

    assert compilation_calls
    assert len(compilation_calls) == len(set(compilation_calls))
    assert audit_result(result).passed
    assert tuple(calls) == compilation_calls
