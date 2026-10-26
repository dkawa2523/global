"""Reproduce the numerical evidence in docs/THEORY.md without network access."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.integrate import quad

from plasma_global import simulate, write_result
from plasma_global.build import compile_case
from plasma_global.chemistry._contracts import normalized_cross_section_curve
from plasma_global.chemistry._maxwell import compile_maxwell_rate
from plasma_global.chemistry.data import CrossSectionData, load_chemistry
from plasma_global.core.result import SimulationResult, to_plain_mapping
from plasma_global.input.schema import CaseSpec
from tools.generate_schema_docs import minimal_case_document

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "plasma_global/assets/examples/cases"
REFERENCES = ROOT / "benchmarks/references"
E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837139e-31


@dataclass(frozen=True)
class Run:
    name: str
    result: SimulationResult
    elapsed_s: float

    def record(self) -> dict[str, Any]:
        files = self.result.metadata["provenance"]["input_files"]["files"]
        return {
            "name": self.name,
            "elapsed_s": self.elapsed_s,
            "time_count": self.result.n_times,
            "solver": to_plain_mapping(self.result.solver_stats),
            "input_files": {
                Path(path).relative_to(ROOT).as_posix(): to_plain_mapping(values)
                for path, values in files.items()
            },
            "result_h5": f"runs/{self.name}/result.h5",
        }


def _raw(name: str) -> dict[str, Any]:
    return yaml.safe_load((CASES / f"{name}.yaml").read_text(encoding="utf-8"))


def _minimal() -> dict[str, Any]:
    return minimal_case_document()


def _run(raw: dict[str, Any], root: Path, name: str) -> Run:
    case = CaseSpec.model_validate(raw).resolve(CASES)
    started = time.perf_counter()
    result = simulate(case)
    elapsed = time.perf_counter() - started
    if result.status.success is not True:
        raise RuntimeError(f"{name}: {result.status.message}")
    write_result(result, root / "runs" / name)
    print(f"{name}: {elapsed:.3f} s, {result.n_times} points", flush=True)
    return Run(name, result, elapsed)


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def _waveform_error(run: Run, reference: Run, series: str) -> float:
    if not np.array_equal(run.result.time_s, reference.result.time_s):
        raise ValueError("waveform comparisons require the same saved times")
    actual, expected = run.result.series(series), reference.result.series(series)
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


def maxwell_benchmark() -> dict[str, Any]:
    rows = []
    sigma, maximum = 2.0e-20, 200.0
    for threshold in (0.0, 2.0):
        curve = CrossSectionData(
            "constant",
            "ionization",
            "Ar",
            threshold,
            threshold,
            np.array([threshold, maximum]),
            np.array([sigma, sigma]),
        )
        rate = compile_maxwell_rate(curve)
        for mean in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0):
            thermal = 2.0 * mean / 3.0
            lower, upper = threshold / thermal, maximum / thermal
            exact = sigma * math.sqrt(
                8.0 * E_CHARGE * thermal / (math.pi * ELECTRON_MASS)
            )
            exact *= (1.0 + lower) * math.exp(-lower) - (1.0 + upper) * math.exp(-upper)
            actual = rate(mean)
            rows.append(
                {
                    "threshold_eV": threshold,
                    "mean_energy_eV": mean,
                    "actual_m3_s": actual,
                    "analytic_m3_s": exact,
                    "relative_error": _relative_error(actual, exact),
                }
            )
    data = load_chemistry(CASES.parent / "chemistry/argon_lxcat/chemistry.yaml")
    curve = data.cross_sections["xs_ar_ionization_zenodo"]
    rate = compile_maxwell_rate(curve)
    energy, cross_section = normalized_cross_section_curve(curve)
    sigma_scale = float(np.max(cross_section))
    quadrature = []
    for mean in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        thermal = 2.0 * mean / 3.0

        def integrand(value: float, temperature: float = thermal) -> float:
            return (
                value
                * float(np.interp(value, energy, cross_section))
                / sigma_scale
                * math.exp(-value / temperature)
            )

        integral, _ = quad(
            integrand,
            float(energy[0]),
            float(energy[-1]),
            points=energy[1:-1],
            epsabs=1.0e-100,
            epsrel=1.0e-10,
            limit=2 * len(energy),
        )
        expected = 2.0 / math.sqrt(math.pi) * math.sqrt(2.0 * E_CHARGE / ELECTRON_MASS)
        expected *= sigma_scale * integral / thermal**1.5
        actual = rate(mean)
        quadrature.append(
            {
                "mean_energy_eV": mean,
                "actual_m3_s": actual,
                "quadrature_m3_s": expected,
                "relative_error": _relative_error(actual, expected),
            }
        )
    return {"constant": rows, "argon": quadrature}


def analytic_benchmarks(root: Path) -> dict[str, Any]:
    raw = _minimal()
    raw["chemistry"]["species"].append(
        {"id": "Ar_star", "mass_amu": 39.948, "elements": {"Ar": 1.0}}
    )
    raw["chemistry"]["gas_reactions"] = [
        {
            "id": "decay",
            "equation": "Ar_star -> Ar",
            "rate": {"kind": "first_order", "rate_s_inv": 25.0},
        }
    ]
    raw["reactor"]["zones"]["plasma"]["initial"]["mole_fractions"] = {
        "Ar": 0.7,
        "Ar_star": 0.3,
    }
    raw["run"] = {"duration_s": 0.2}
    raw["solver"] = {"sample_interval_s": 0.002}
    decay = _run(raw, root, "analytic-decay")
    actual = decay.result.series("n[plasma,Ar_star]")
    expected = actual[0] * np.exp(-25.0 * decay.result.time_s)
    decay_error = float(np.max(np.abs(actual - expected) / expected))
    raw = _minimal()
    source = raw["reactor"]["zones"].pop("plasma")
    target = copy.deepcopy(source)
    target["volume_m3"] = 0.03
    target["initial"]["pressure_Pa"] = 0.25
    raw["reactor"]["zones"] = {"source": source, "target": target}
    raw["reactor"]["edges"] = {
        "transfer": {
            "from_zone": "source",
            "to_zone": "target",
            "conductance_m3_s": 0.02,
        }
    }
    raw["run"] = {"duration_s": 0.5}
    raw["solver"] = {"sample_interval_s": 0.005}
    transport = _run(raw, root, "analytic-transport")
    ns = transport.result.series("n[source,Ar]")
    nt = transport.result.series("n[target,Ar]")
    reference_s = ns[0] * np.exp(-2.0 * transport.result.time_s)
    reference_t = nt[0] + (ns[0] - reference_s) / 3.0
    particles = 0.01 * ns + 0.03 * nt
    return {
        "decay": {**decay.record(), "max_pointwise_relative_error": decay_error},
        "transport": {
            **transport.record(),
            "source_error": float(np.max(np.abs(ns - reference_s) / reference_s)),
            "target_error": float(np.max(np.abs(nt - reference_t) / reference_t)),
            "particle_conservation_error": float(
                np.max(np.abs(particles - particles[0])) / particles[0]
            ),
        },
    }


def cross_code_benchmarks(root: Path) -> dict[str, Any]:
    crane = _run(_raw("crane_two_reaction_argon"), root, "crane")
    reference = yaml.safe_load(
        (REFERENCES / "crane_two_reaction_argon_reference.yaml").read_text()
    )
    expected = float(reference["converted_output_final"]["Ar_plus_density_m3"])
    actual = float(crane.result.series("n[plasma,Ar_plus]")[-1])
    raw = _raw("zdplaskin_example2")
    zdp = _run(raw, root, "zdplaskin")
    saved = yaml.safe_load(
        (REFERENCES / "zdplaskin_example2_summary.yaml").read_text()
    )["saved_output"]
    labels = {
        "Ar_star": "n[plasma,Ar_star]",
        "Ar_plus": "n[plasma,Ar_plus]",
        "Ar2_plus": "n[plasma,Ar2_plus]",
    }
    rows = []
    for name, label in labels.items():
        value = float(zdp.result.series(label)[-1])
        expected_value = float(saved[f"final_saved_{name}_density_m3"])
        rows.append(
            {
                "quantity": name,
                "actual": value,
                "reference": expected_value,
                "relative_error": _relative_error(value, expected_value),
                "unit": "m^-3",
            }
        )
    electrons = zdp.result.series(labels["Ar_plus"]) + zdp.result.series(
        labels["Ar2_plus"]
    )
    for name, value in (
        ("final", float(electrons[-1])),
        ("peak", float(np.max(electrons))),
    ):
        expected_value = float(saved[f"{name}_saved_electron_density_m3"])
        rows.append(
            {
                "quantity": f"electron_{name}",
                "actual": value,
                "reference": expected_value,
                "relative_error": _relative_error(value, expected_value),
                "unit": "m^-3",
            }
        )
    compiled = compile_case(CaseSpec.model_validate(raw).resolve(CASES))
    evaluation = compiled.model.evaluate(
        float(zdp.result.time_s[-1]), zdp.result.state[-1], compiled.segments[-1]
    )
    if evaluation.power_coupling is None:
        raise RuntimeError("DC benchmark requires a port result")
    port = evaluation.power_coupling.port_results["dc_series_drive"]
    with (REFERENCES / "zdplaskin_example2_circuit.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        circuit = list(csv.DictReader(stream))[-1]
    values = [
        ("voltage", port.observables["plasma_voltage_V"], "voltage_V", "V"),
        ("current", port.observables["current_A"], "current_A", "A"),
        ("reduced_field", port.reduced_field_Td, "reduced_field_Td", "Td"),
    ]
    for name, value, key, unit in values:
        if value is None:
            raise RuntimeError(f"DC benchmark did not produce {name}")
        value = float(value)
        expected_value = float(circuit[key])
        rows.append(
            {
                "quantity": name,
                "actual": value,
                "reference": expected_value,
                "relative_error": _relative_error(value, expected_value),
                "unit": unit,
            }
        )
    return {
        "crane": {
            **crane.record(),
            "actual": actual,
            "reference": expected,
            "relative_error": _relative_error(actual, expected),
        },
        "zdplaskin": {**zdp.record(), "comparisons": rows},
    }


def _argon_input() -> dict[str, Any]:
    raw = _raw("argon_maxwellian")
    initial = raw["reactor"]["zones"]["plasma"]["initial"]
    initial.pop("densities_m3")
    initial.update(mole_fractions={"Ar": 1.0}, seed_densities_m3={"Ar_plus": 1.0e14})
    raw["output"] = {
        "quantities": [
            "electron_density_m3",
            "electron_temperature_eV",
            "pressure_Pa",
            "absorbed_power_W",
        ],
        "reaction_rates": True,
    }
    return raw


def _final(run: Run) -> dict[str, float]:
    return {
        "electron_density_m3": float(
            run.result.series("electron_density_m3[plasma]")[-1]
        ),
        "electron_temperature_eV": float(
            run.result.series("electron_temperature_eV[plasma]")[-1]
        ),
    }


def argon_convergence(root: Path) -> tuple[dict[str, Any], Run]:
    raw = _argon_input()
    runs = []
    for index, factor in enumerate((1.0, 0.1, 0.01)):
        config = copy.deepcopy(raw)
        config["solver"].update(rtol=1.0e-6 * factor, atol=1.0e-14 * factor)
        runs.append(_run(config, root, f"argon-tolerance-{index}"))
    reference = runs[-1]
    rows = [
        {
            **run.record(),
            "rtol": 1.0e-6 * factor,
            "atol": 1.0e-14 * factor,
            **_final(run),
            "density_waveform_error": _waveform_error(
                run, reference, "electron_density_m3[plasma]"
            ),
            "temperature_waveform_error": _waveform_error(
                run, reference, "electron_temperature_eV[plasma]"
            ),
        }
        for run, factor in zip(runs, (1.0, 0.1, 0.01), strict=True)
    ]
    return {"runs": rows}, runs[0]


def argon_sensitivity(root: Path, baseline: Run) -> list[dict[str, Any]]:
    rows = []
    for parameter in ("power", "pressure", "area"):
        for factor in (0.9, 1.1):
            raw = _argon_input()
            if parameter == "power":
                raw["run"]["steps"][0]["commands"]["power_ports"]["source"][
                    "absorbed_power_W"
                ] *= factor
            elif parameter == "pressure":
                raw["reactor"]["zones"]["plasma"]["initial"]["pressure_Pa"] *= factor
            else:
                raw["reactor"]["surfaces"]["wall"]["area_m2"] *= factor
            run = _run(raw, root, f"argon-{parameter}-{factor:g}")
            final, original = _final(run), _final(baseline)
            rows.append(
                {
                    **run.record(),
                    "parameter": parameter,
                    "factor": factor,
                    **final,
                    "density_change_percent": 100.0
                    * (
                        final["electron_density_m3"] / original["electron_density_m3"]
                        - 1.0
                    ),
                    "temperature_change_percent": 100.0
                    * (
                        final["electron_temperature_eV"]
                        / original["electron_temperature_eV"]
                        - 1.0
                    ),
                }
            )
    return rows


def event_sampling(root: Path) -> list[dict[str, Any]]:
    rows = []
    for interval in (4.0e-5, 2.0e-5, 1.0e-5, 5.0e-6, 2.5e-6, 1.25e-6):
        raw = _argon_input()
        raw["solver"].update(rtol=1.0e-8, atol=1.0e-16, sample_interval_s=interval)
        run = _run(raw, root, f"events-{interval:g}")
        ionize = float(run.result.series("gas_event_count[plasma,G_AR_ION]")[-1])
        wall = float(
            run.result.series("boundary_event_count[wall,wall_Ar_plus_neutralization]")[
                -1
            ]
        )
        density = run.result.series("n[plasma,Ar_plus]")
        delta = 0.01 * float(density[-1] - density[0])
        rows.append(
            {
                **run.record(),
                "sample_interval_s": interval,
                "ionization_count": ionize,
                "wall_count": wall,
                "ion_particle_change": delta,
                "balance_residual": abs(delta - (ionize - wall)) / max(ionize, wall),
            }
        )
    for row in rows:
        row["ionization_relative_to_finest"] = _relative_error(
            row["ionization_count"], rows[-1]["ionization_count"]
        )
        row["wall_relative_to_finest"] = _relative_error(
            row["wall_count"], rows[-1]["wall_count"]
        )
    return rows


def write_figures(report: dict[str, Any], baseline: Run, target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target.mkdir()
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    time_ms = baseline.result.time_s * 1000.0
    figure, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.2), sharex=True, layout="constrained"
    )
    axes[0].plot(
        time_ms, baseline.result.series("electron_density_m3[plasma]"), color="#175a80"
    )
    axes[0].set_ylabel("Electron density [m$^{-3}$]")
    axes[1].plot(
        time_ms,
        baseline.result.series("electron_temperature_eV[plasma]"),
        color="#007f74",
    )
    axes[1].set(xlabel="Time [ms]", ylabel="Equivalent electron temperature [eV]")
    for axis in axes:
        axis.axvline(1.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
    figure.savefig(target / "theory-transient.png", dpi=220)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.7), layout="constrained")
    rows = report["maxwell"]["argon"]
    energy = [row["mean_energy_eV"] for row in rows]
    axes[0].semilogy(
        energy,
        [row["actual_m3_s"] for row in rows],
        "o-",
        color="#175a80",
        label="Analytic interval integral",
    )
    axes[0].semilogy(
        energy,
        [row["quadrature_m3_s"] for row in rows],
        "+",
        color="#bd6a19",
        markersize=10,
        label="Independent quadrature",
    )
    axes[0].set(
        xlabel="Mean electron energy [eV]", ylabel="Ar ionization coefficient [m$^3$/s]"
    )
    axes[0].legend(fontsize=8)
    axes[1].plot(energy, [row["relative_error"] for row in rows], "o-", color="#007f74")
    axes[1].set(xlabel="Mean electron energy [eV]", ylabel="Relative difference")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(target / "theory-maxwell.png", dpi=220)
    plt.close(figure)
    _sampling_figure(plt, report["event_sampling"], target)
    _sensitivity_figure(plt, report["sensitivity"], target)


def _sampling_figure(plt: Any, rows: list[dict[str, Any]], target: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    interval_us = [row["sample_interval_s"] * 1.0e6 for row in rows[:-1]]
    axis.loglog(
        interval_us,
        [row["ionization_relative_to_finest"] for row in rows[:-1]],
        "o-",
        label="Ionization count",
        color="#175a80",
    )
    axis.loglog(
        interval_us,
        [row["wall_relative_to_finest"] for row in rows[:-1]],
        "s-",
        label="Wall event count",
        color="#007f74",
    )
    axis.set(
        xlabel="Saved interval [microseconds]",
        ylabel="Relative difference from 1.25 microseconds",
    )
    axis.grid(which="both", alpha=0.2)
    axis.legend()
    figure.savefig(target / "theory-sampling.png", dpi=220)
    plt.close(figure)


def _sensitivity_figure(plt: Any, rows: list[dict[str, Any]], target: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    positions = np.arange(len(rows))
    axis.bar(
        positions - 0.18,
        [row["density_change_percent"] for row in rows],
        width=0.36,
        label="Final electron density",
        color="#175a80",
    )
    axis.bar(
        positions + 0.18,
        [row["temperature_change_percent"] for row in rows],
        width=0.36,
        label="Final electron temperature",
        color="#007f74",
    )
    axis.set_xticks(
        positions,
        [f"{row['parameter']}\n{100 * (row['factor'] - 1):+.0f}%" for row in rows],
    )
    axis.set_ylabel("Change at 2 ms [%]")
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.legend(fontsize=9)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(target / "theory-sensitivity.png", dpi=220)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in ("numpy", "scipy", "h5py", "pydantic", "PyYAML", "matplotlib")
        },
        "maxwell": maxwell_benchmark(),
        "analytic": analytic_benchmarks(target),
        "cross_code": cross_code_benchmarks(target),
    }
    report["argon_convergence"], baseline = argon_convergence(target)
    report["software"] = to_plain_mapping(
        baseline.result.metadata["provenance"]["software"]
    )
    report["sensitivity"] = argon_sensitivity(target, baseline)
    report["event_sampling"] = event_sampling(target)
    (target / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_figures(report, baseline, target / "figures")
    print(target / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
