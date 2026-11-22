"""Extract the DC reference history and independent BOLSIG+ 07/2024 coefficients."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

HISTORY_FILE = "zdplaskin_example2_history.csv"
COEFFICIENT_FILE = "electrons.csv"
_HISTORY_COLUMNS = {
    "time_s": ("Time", 1.0),
    "electron_density_m3": ("E", 1.0e6),
    "Ar_star_density_m3": ("AR*", 1.0e6),
    "Ar_plus_density_m3": ("AR^+", 1.0e6),
    "Ar2_plus_density_m3": ("AR2^+", 1.0e6),
    "plasma_voltage_V": ("V", 1.0),
    "current_A": ("J", 1.0),
    "reduced_field_Td": ("E/N", 1.0),
}
_COEFFICIENT_COLUMNS = {
    "reduced_field_Td": r"Electric field / N \(Td\)",
    "mean_energy_eV": r"Mean energy \(eV\)",
    "reduced_mobility_1_V_m_s": r"Mobility \*N \(1/m/V/s\)",
    "k:xs_zdp_ar_excitation": r"C2 +Ar +Excitation +11\.50 eV",
    "k:xs_zdp_ar_ionization": r"C3 +Ar +Ionization +15\.80 eV",
    "k:xs_zdp_ar_star_deexcitation": r"C4 +Ar\* +De-excitation +11\.50 eV",
    "k:xs_zdp_ar_star_ionization": r"C5 +Ar\* +Ionization +4\.30 eV",
}


def _save(path: Path, columns: dict[str, np.ndarray]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        np.savetxt(
            stream,
            np.column_stack(list(columns.values())),
            fmt="%.9g",
            delimiter=",",
            header=",".join(columns),
            comments="",
        )


def _history(source: Path) -> dict[str, np.ndarray]:
    with source.open(encoding="ascii") as stream:
        labels = next(stream).split()
        data = np.loadtxt(stream, ndmin=2)
    if not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0.0):
        raise ValueError("ZDPlasKin history needs finite rows and increasing time")
    return {
        name: data[:, labels.index(column)] * multiplier
        for name, (column, multiplier) in _HISTORY_COLUMNS.items()
    }


def _block(report: str, heading: str) -> np.ndarray:
    if heading.startswith("C"):
        heading += r" *\nRate coefficient \(m3/s\)"
    match = re.search(rf"^{heading}[ \t]*$", report, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing BOLSIG+ block {heading!r}")
    return np.array(
        report[match.end() :].strip().split("\n\n", 1)[0].split(), dtype=float
    )


def _coefficients(source: Path) -> dict[str, np.ndarray]:
    report = source.read_text(encoding="ascii")
    if not report.startswith("BOLSIG+ version: 07/2024"):
        raise ValueError("expected BOLSIG+ 07/2024")
    columns = {
        name: _block(report, heading) for name, heading in _COEFFICIENT_COLUMNS.items()
    }
    iterations = _block(report, "# of iterations")
    if iterations.size != 641 or np.any((iterations <= 0) | (iterations >= 1000)):
        raise ValueError("expected 641 converged BOLSIG+ points")
    if any(
        values.shape != iterations.shape
        or not np.isfinite(values).all()
        or np.any(values < 0)
        for values in columns.values()
    ):
        raise ValueError("BOLSIG+ coefficient columns must share the converged grid")
    return columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zdplaskin-output", type=Path, required=True)
    parser.add_argument("--bolsig-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    history = _history(args.zdplaskin_output)
    coefficients = _coefficients(args.bolsig_output)
    args.output.mkdir(parents=True, exist_ok=False)
    _save(args.output / HISTORY_FILE, history)
    _save(args.output / COEFFICIENT_FILE, coefficients)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
