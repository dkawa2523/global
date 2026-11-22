from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from plasma_global import load_case, simulate
from plasma_global._numeric_csv import read_numeric_csv
from tools.benchmarks.prepare_zdplaskin_reference import HISTORY_FILE

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "benchmarks" / "references"
CRANE_CASE = (
    ROOT / "plasma_global" / "examples" / "cases" / "crane_two_reaction_argon.yaml"
)
ZDP_CASE = ROOT / "plasma_global" / "examples" / "cases" / "zdplaskin_example2.yaml"


def _load_yaml(path: Path) -> dict[str, object]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_crane_scalar_chemistry_cross_code_regression() -> None:
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


def test_zdplaskin_seed_and_circuit_at_reference_times() -> None:
    history = read_numeric_csv(REFERENCES / HISTORY_FILE)
    case = load_case(ZDP_CASE)
    solver = case.solver.model_copy(
        update={"sample_interval_s": None, "save_at_s": history["time_s"].tolist()}
    )
    result = simulate(case.model_copy(update={"solver": solver}))

    assert result.status.success
    np.testing.assert_array_equal(result.time_s[1:], history["time_s"])
    assert result.series("electron_density_m3[plasma]")[0] == 1.0e12
    assert result.series("n[plasma,Ar2_plus]")[0] == 0.0
    current = result.series("port_current_A[dc_series_drive]")
    voltage = result.series("port_plasma_voltage_V[dc_series_drive]")
    np.testing.assert_allclose(voltage + current * 1.0e5, 1000.0, rtol=1.0e-12)
    np.testing.assert_allclose(
        result.series("absorbed_power_W[plasma]"), voltage * current, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        result.series("port_reduced_field_Td[dc_series_drive]"),
        voltage / (0.004 * result.series("neutral_density_m3[plasma]") * 1e-21),
        rtol=1.0e-12,
    )
