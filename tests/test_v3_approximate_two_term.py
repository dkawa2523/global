from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global.build import compile_case
from plasma_global.chemistry.data import load_chemistry
from plasma_global.core.solver import solve_compiled_model
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.eedf import ApproximateTwoTermEEDF
from plasma_global.experimental.prepared_kinetics import (
    prepare_approximate_two_term_kinetics,
)
from plasma_global.input.load import load_case
from plasma_global.input.schema import CaseSpec

ROOT = Path(__file__).parents[1]
MINIMAL_CASE = ROOT / "tests" / "fixtures" / "v3_minimal" / "case.yaml"


def _write_chemistry(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "species.csv").write_text(
        "id,phase,charge,mass_amu,elements,state_tags\n"
        "e,gas,-1,0.00054858,,electron\n"
        "Ar,gas,0,39.948,Ar:1,\n"
        "Ar_plus,gas,1,39.948,Ar:1,ion\n",
        encoding="utf-8",
    )
    (directory / "gas.csv").write_text(
        "id,equation,rate_model,energy_loss_eV\n"
        "ionize,e + Ar -> Ar_plus + e + e,ionization,15.76\n",
        encoding="utf-8",
    )
    (directory / "boundary.csv").write_text(
        "id,equation,surfaces\nneutralize,Ar_plus -> Ar,wall\n",
        encoding="utf-8",
    )
    (directory / "rates.yaml").write_text(
        "rate_models:\n"
        "  ionization:\n"
        "    kind: electron_impact\n"
        "    cross_section: xs_ionization\n"
        "    branching_yield: 1.0\n",
        encoding="utf-8",
    )
    (directory / "momentum.csv").write_text(
        "energy_eV,sigma_m2\n0.0,1.0e-19\n1.0,1.1e-19\n20.0,8.0e-20\n100.0,5.0e-20\n",
        encoding="utf-8",
    )
    (directory / "ionization.csv").write_text(
        "energy_eV,sigma_m2\n0.0,0.0\n15.76,0.0\n30.0,2.0e-20\n100.0,1.0e-20\n",
        encoding="utf-8",
    )
    (directory / "cross_sections.yaml").write_text(
        "cross_sections:\n"
        "  - id: xs_momentum\n"
        "    kind: momentum_transfer\n"
        "    target: Ar\n"
        "    threshold_eV: 0.0\n"
        "    energy_loss_eV: 0.0\n"
        "    file: momentum.csv\n"
        "  - id: xs_ionization\n"
        "    kind: ionization\n"
        "    target: Ar\n"
        "    threshold_eV: 15.76\n"
        "    energy_loss_eV: 15.76\n"
        "    file: ionization.csv\n",
        encoding="utf-8",
    )
    manifest = directory / "chemistry.yaml"
    manifest.write_text(
        "schema_version: 3\n"
        "species: species.csv\n"
        "gas_reactions: gas.csv\n"
        "boundary_reactions: boundary.csv\n"
        "rate_models: rates.yaml\n"
        "cross_sections: cross_sections.yaml\n",
        encoding="utf-8",
    )
    return manifest


def _approximate_case(manifest: Path) -> CaseSpec:
    data = load_case(MINIMAL_CASE).model_dump(mode="python")
    data["chemistry"]["manifest"] = manifest
    zone = data["reactor"]["zones"][0]
    zone.pop("initial_mean_energy_eV")
    data["reactor"]["power_ports"][0]["model"] = {
        "kind": "experimental.rf_envelope",
        "role": "source",
        "control": "absorbed_power",
        "base_reduced_field_Td": 40.0,
    }
    data["recipe"]["steps"][0]["commands"]["power_ports"]["source"] = {
        "kind": "experimental.rf_envelope",
        "absorbed_power_W": 1.0e-6,
    }
    data["models"]["electrons"] = {
        "kind": "experimental.approximate_two_term",
        "mixture_key_species": ["Ar"],
        "cache": {"max_entries": 2, "fraction_decimals": 4},
        "energy_grid": {"min_eV": 1.0e-3, "max_eV": 100.0, "n": 32},
        "reduced_field_grid": {"min_Td": 5.0, "max_Td": 100.0, "n": 4},
        "max_shape_iterations": 48,
    }
    data["models"]["electron_closure"] = {"kind": "local_field"}
    return CaseSpec.model_validate(data)


def test_compilation_prepares_eedf_once_and_rhs_only_interpolates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _approximate_case(_write_chemistry(tmp_path / "chemistry"))
    original_evaluate = ApproximateTwoTermEEDF.evaluate
    preparation_calls = 0

    def counted_evaluate(self: object, *args: object, **kwargs: object) -> object:
        nonlocal preparation_calls
        preparation_calls += 1
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(ApproximateTwoTermEEDF, "evaluate", counted_evaluate)
    compiled = compile_case(case)
    assert preparation_calls == 4

    def forbidden_runtime_solve(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the EEDF solver was called from the ODE runtime")

    monkeypatch.setattr(ApproximateTwoTermEEDF, "evaluate", forbidden_runtime_solve)
    state = compiled.model.initial_state(compiled.initial_state)
    evaluation = compiled.model.evaluate(0.0, state, compiled.segments[0])
    result = solve_compiled_model(
        compiled.model, compiled.initial_state, compiled.solver_settings
    )

    kinetics = evaluation.kinetics_by_zone["plasma"]
    assert kinetics.rate_coefficients["xs_ionization"] >= 0.0
    assert np.isfinite(evaluation.derivative).all()
    assert result.status.success
    assert preparation_calls == 4
    provenance = compiled.metadata["provenance"]["electron_kinetics"]
    assert provenance["classification"] == (
        "experimental_fixed_initial_neutral_mixture"
    )
    assert provenance["fixed_mixture"] is True
    assert provenance["runtime_eedf_solve"] is False
    assert provenance["closure"] == "local_field"


def test_distinct_initial_neutral_mixtures_get_distinct_immutable_tables(
    tmp_path: Path,
) -> None:
    chemistry = load_chemistry(_write_chemistry(tmp_path / "chemistry"))
    prepared = prepare_approximate_two_term_kinetics(
        chemistry,
        {
            "same_a": {"Ar": 2.0e20},
            "same_b": {"Ar": 2.0e20},
            "different": {"Ar": 1.0e20},
        },
        mixture_key_species=("Ar",),
        cache_max_entries=2,
        energy_max_eV=100.0,
        energy_points=32,
        field_min_Td=5.0,
        field_max_Td=10.0,
        field_points=2,
        max_iterations=48,
    )

    same_a = prepared.by_zone["same_a"]
    same_b = prepared.by_zone["same_b"]
    different = prepared.by_zone["different"]
    assert same_a is same_b
    assert same_a is not different
    assert prepared.provenance["unique_table_count"] == 2
    assert not same_a.axis.flags.writeable
    assert not same_a.rate_tables["xs_momentum"].flags.writeable
    assert different.mobility_m2_V_s == pytest.approx(2.0 * same_a.mobility_m2_V_s)

    with pytest.raises(CaseValidationError, match=r"cache\.max_entries"):
        prepare_approximate_two_term_kinetics(
            chemistry,
            {"a": {"Ar": 2.0e20}, "b": {"Ar": 1.0e20}},
            cache_max_entries=1,
            energy_max_eV=100.0,
            energy_points=32,
            field_min_Td=5.0,
            field_max_Td=10.0,
            field_points=2,
            max_iterations=48,
        )


def test_approximate_two_term_rejects_superelastic_energy_gain(
    tmp_path: Path,
) -> None:
    manifest = _write_chemistry(tmp_path / "chemistry")
    path = manifest.parent / "cross_sections.yaml"
    path.write_text(
        path.read_text(encoding="utf-8") + "  - id: xs_superelastic\n"
        "    kind: deexcitation\n"
        "    target: Ar\n"
        "    threshold_eV: 0.0\n"
        "    electron_energy_transfer_eV: 11.5\n"
        "    file: ionization.csv\n",
        encoding="utf-8",
    )

    with pytest.raises(CaseValidationError, match="does not support superelastic"):
        prepare_approximate_two_term_kinetics(
            load_chemistry(manifest),
            {"plasma": {"Ar": 1.0e20}},
            energy_max_eV=100.0,
            energy_points=32,
            field_min_Td=5.0,
            field_max_Td=10.0,
            field_points=2,
            max_iterations=48,
        )


def test_superelastic_error_precedes_deferred_invalid_target_error(
    tmp_path: Path,
) -> None:
    manifest = _write_chemistry(tmp_path / "chemistry")
    path = manifest.parent / "cross_sections.yaml"
    contents = path.read_text(encoding="utf-8").replace(
        "    target: Ar\n",
        "    target: Ar_plus\n",
        1,
    )
    path.write_text(
        contents + "  - id: xs_superelastic\n"
        "    kind: deexcitation\n"
        "    target: Ar\n"
        "    threshold_eV: 0.0\n"
        "    electron_energy_transfer_eV: 11.5\n"
        "    file: ionization.csv\n",
        encoding="utf-8",
    )

    with pytest.raises(CaseValidationError, match="does not support superelastic"):
        prepare_approximate_two_term_kinetics(
            load_chemistry(manifest),
            {"plasma": {"Ar": 1.0e20}},
            energy_max_eV=100.0,
            energy_points=32,
            field_min_Td=5.0,
            field_max_Td=10.0,
            field_points=2,
            max_iterations=48,
        )


def test_invalid_target_error_precedes_missing_momentum_error(tmp_path: Path) -> None:
    manifest = _write_chemistry(tmp_path / "chemistry")
    path = manifest.parent / "cross_sections.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    target: Ar\n",
            "    target: Ar_plus\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(CaseValidationError, match="invalid targets: Ar_plus"):
        prepare_approximate_two_term_kinetics(
            load_chemistry(manifest),
            {"plasma": {"Ar": 1.0e20}},
            energy_max_eV=100.0,
            energy_points=32,
            field_min_Td=5.0,
            field_max_Td=10.0,
            field_points=2,
            max_iterations=48,
        )


def test_preparation_normalizes_numpy_scalars_and_plain_provenance(
    tmp_path: Path,
) -> None:
    chemistry = load_chemistry(_write_chemistry(tmp_path / "chemistry"))

    prepared = prepare_approximate_two_term_kinetics(
        chemistry,
        {"plasma": {"Ar": np.float32(1.0e20)}},
        mixture_key_species=("Ar",),
        cache_max_entries=np.int64(2),
        fraction_decimals=np.int64(4),
        energy_min_eV=np.float32(1.0e-3),
        energy_max_eV=np.float32(10.0),
        energy_points=np.int64(32),
        field_min_Td=np.float32(5.0),
        field_max_Td=np.float32(10.0),
        field_points=np.int64(2),
        max_iterations=np.int64(48),
    )

    document = yaml.safe_load(yaml.safe_dump(dict(prepared.provenance)))
    assert set(prepared.by_zone) == {"plasma"}
    assert document["energy_grid_eV"] == {
        "minimum": pytest.approx(1.0e-3),
        "maximum": pytest.approx(10.0),
        "count": 32,
    }
    assert document["reduced_field_grid_Td"]["count"] == 2
    assert isinstance(document["zones"]["plasma"]["target_densities_m3"]["Ar"], float)


def test_approximate_two_term_schema_rejects_electron_energy_closure(
    tmp_path: Path,
) -> None:
    case = _approximate_case(_write_chemistry(tmp_path / "chemistry"))
    data = case.model_dump(mode="python")
    data["models"]["electron_closure"] = {"kind": "electron_energy"}
    data["reactor"]["zones"][0]["initial_mean_energy_eV"] = 3.0

    with pytest.raises(ValueError, match=r"require.*local_field"):
        CaseSpec.model_validate(data)


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/v3/cases/argon_lxcat.yaml",
        "examples/v3/cases/rf_envelope_calibration.yaml",
    ],
)
def test_experimental_argon_examples_compile_and_reach_the_rhs(
    relative_path: str,
) -> None:
    compiled = compile_case(load_case(ROOT / relative_path))
    state = compiled.model.initial_state(compiled.initial_state)
    evaluation = compiled.model.evaluate(
        compiled.segments[0].start_s, state, compiled.segments[0]
    )

    assert not compiled.model.layout.evolves_electron_energy
    assert set(evaluation.kinetics_by_zone) == {
        zone.zone_id for zone in compiled.case.reactor.zones
    }
    assert np.isfinite(evaluation.derivative).all()
    assert (
        compiled.metadata["provenance"]["electron_kinetics"]["classification"]
        == "experimental_fixed_initial_neutral_mixture"
    )
