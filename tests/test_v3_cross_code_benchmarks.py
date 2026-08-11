from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global import load_case, simulate
from plasma_global.build import compile_case
from tools.benchmarks.build_zdplaskin_rate_table import (
    build_zdplaskin_rate_table,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "benchmarks" / "references"
RAW_ZDP_TABLE = ROOT / "benchmarks" / "raw" / "zdplaskin_example2_eovern"
CRANE_CASE = ROOT / "examples" / "v3" / "cases" / "crane_two_reaction_argon.yaml"
ZDP_CASE = ROOT / "examples" / "v3" / "cases" / "zdplaskin_example2.yaml"
ZDP_TABLE = (
    ROOT
    / "examples"
    / "v3"
    / "chemistry"
    / "zdplaskin_example2"
    / "tables"
    / "zdplaskin_example2_eovern_rates.h5"
)
ZDP_TABLE_SHA256 = "fa469b82fe486c10d25de10587a0cb3cb057661b7a7d26f201c5f12e678417de"


def _load_yaml(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_reference_circuit_row() -> dict[str, float]:
    path = REFERENCES / "zdplaskin_example2_circuit.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    return {name: float(value) for name, value in rows[-1].items()}


def test_zdplaskin_table_is_reproducible_from_strict_raw_inputs(
    tmp_path: Path,
) -> None:
    rebuilt = build_zdplaskin_rate_table(
        tmp_path / "zdplaskin.h5", raw_dir=RAW_ZDP_TABLE
    )

    assert _sha256(ZDP_TABLE) == ZDP_TABLE_SHA256
    assert _sha256(rebuilt) == ZDP_TABLE_SHA256


def test_v3_crane_scalar_chemistry_cross_code_regression() -> None:
    reference = _load_yaml(REFERENCES / "crane_two_reaction_argon_reference.yaml")[
        "converted_output_final"
    ]
    assert isinstance(reference, dict)
    result = simulate(load_case(CRANE_CASE))

    assert result.status.success
    assert result.time_s[-1] == pytest.approx(7.5e-7)
    assert result.series("n[plasma,Ar_plus]")[-1] == pytest.approx(
        float(reference["Ar_plus_density_m3"]), rel=5.0e-4
    )


def test_crane_is_stable_when_dimensionless_solver_tolerances_are_halved() -> None:
    case = load_case(CRANE_CASE)
    fine_solver = case.solver.model_copy(
        update={"rtol": 0.5 * case.solver.rtol, "atol": 0.5 * case.solver.atol}
    )
    fine_case = case.model_copy(update={"solver": fine_solver})

    coarse = simulate(case)
    fine = simulate(fine_case)
    for name in (
        "n[plasma,Ar]",
        "n[plasma,Ar_plus]",
        "electron_density_m3[plasma]",
        "mean_energy_eV[plasma]",
    ):
        assert coarse.series(name)[-1] == pytest.approx(fine.series(name)[-1], rel=0.01)


def test_v3_zdplaskin_species_and_circuit_cross_code_regression() -> None:
    case = load_case(ZDP_CASE)
    compiled = compile_case(case)
    result = simulate(case)
    summary = _load_yaml(REFERENCES / "zdplaskin_example2_summary.yaml")["saved_output"]
    assert isinstance(summary, dict)

    ar_plus = result.series("n[plasma,Ar_plus]")
    ar2_plus = result.series("n[plasma,Ar2_plus]")
    electron_density = ar_plus + ar2_plus
    species_comparisons = {
        "n[plasma,Ar_star]": "final_saved_Ar_star_density_m3",
        "n[plasma,Ar_plus]": "final_saved_Ar_plus_density_m3",
        "n[plasma,Ar2_plus]": "final_saved_Ar2_plus_density_m3",
    }

    assert result.status.success
    for label, reference_key in species_comparisons.items():
        assert result.series(label)[-1] == pytest.approx(
            float(summary[reference_key]), rel=0.10
        )
    assert electron_density[-1] == pytest.approx(
        float(summary["final_saved_electron_density_m3"]), rel=0.10
    )
    assert np.max(electron_density) == pytest.approx(
        float(summary["peak_saved_electron_density_m3"]), rel=0.10
    )

    evaluation = compiled.model.evaluate(
        float(result.time_s[-1]),
        result.state[-1],
        compiled.model.segments[-1],
    )
    assert evaluation.power_coupling is not None
    port = evaluation.power_coupling.port_results["dc_series_drive"]
    circuit = _final_reference_circuit_row()
    assert port.observables["plasma_voltage_V"] == pytest.approx(
        circuit["voltage_V"], rel=0.05
    )
    assert port.observables["current_A"] == pytest.approx(
        circuit["current_A"], rel=0.05
    )
    assert port.reduced_field_Td == pytest.approx(circuit["reduced_field_Td"], rel=0.05)
