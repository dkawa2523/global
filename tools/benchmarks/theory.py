"""Compare the shipped YAML benchmark inputs with analytic and saved references."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass, field
from hashlib import sha256
from importlib.metadata import version
from operator import itemgetter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.integrate import quad

from plasma_global import load_case, simulate, write_result
from plasma_global.build import compile_case
from plasma_global.chemistry._contracts import normalized_cross_section_curve
from plasma_global.chemistry._maxwell import compile_maxwell_rate
from plasma_global.chemistry.data import CrossSectionData, load_chemistry
from plasma_global.core.result import SimulationResult, to_plain_mapping
from tools.benchmarks import _theory_figures

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "plasma_global/examples/cases"
REFERENCES = ROOT / "benchmarks/references"
E_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837139e-31


@dataclass(frozen=True)
class Run:
    name: str
    case_file: Path
    result: SimulationResult
    elapsed_s: float

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "case_yaml": self.case_file.relative_to(ROOT).as_posix(),
            "elapsed_s": self.elapsed_s,
            "time_count": self.result.n_times,
            "time_start_s": float(self.result.time_s[0]),
            "time_end_s": float(self.result.time_s[-1]),
            "solver": to_plain_mapping(self.result.solver_stats),
            "result_h5": f"runs/{self.name}/result.h5",
        }


@dataclass
class Runner:
    output: Path
    runs: list[Run] = field(default_factory=list)

    def run(self, identifier: str) -> Run:
        source = CASES / f"{identifier}.yaml"
        case = load_case(source)
        started = time.perf_counter()
        result = simulate(case)
        elapsed = time.perf_counter() - started
        if result.status.success is not True:
            raise RuntimeError(f"{identifier}: {result.status.message}")
        name = source.stem
        write_result(result, self.output / "runs" / name)
        print(f"{identifier}: {elapsed:.3f} s, {result.n_times} points", flush=True)
        run = Run(name, source, result, elapsed)
        self.runs.append(run)
        return run

    def inputs(self) -> dict[str, Any]:
        paths = {
            Path(__file__),
            Path(_theory_figures.__file__),
            *REFERENCES.glob("*.yaml"),
            *REFERENCES.glob("*.csv"),
        }
        for run in self.runs:
            paths.update(
                Path(path)
                for path in run.result.metadata["provenance"]["input_files"]["files"]
            )
        digest = sha256()
        names = []
        for path in sorted(paths):
            name = path.relative_to(ROOT).as_posix()
            names.append(name)
            digest.update(name.encode() + b"\0" + sha256(path.read_bytes()).digest())
        return {
            "algorithm": "sha256-path-content-v1",
            "sha256": digest.hexdigest(),
            "files": names,
        }


def _config(run: Run) -> dict[str, Any]:
    return yaml.safe_load(run.result.metadata["effective_case_yaml"])


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / abs(expected)


def _waveform_error(run: Run, reference: Run, series: str) -> float:
    if not np.array_equal(run.result.time_s, reference.result.time_s):
        raise ValueError("waveform comparisons require the same saved times")
    actual, expected = run.result.series(series), reference.result.series(series)
    return float(np.max(np.abs(actual - expected)) / np.max(np.abs(expected)))


def _same_end_time(run: Run, expected_s: float) -> None:
    actual = float(run.result.time_s[-1])
    if actual != expected_s:
        raise ValueError(
            f"{run.name}: comparison requires end time {expected_s:g} s; "
            f"got {actual:g} s"
        )


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


def analytic_benchmarks(runner: Runner) -> dict[str, Any]:
    decay = runner.run("benchmarks/analytic-decay")
    rate = float(_config(decay)["chemistry"]["gas_reactions"][0]["rate"]["rate_s_inv"])
    actual = decay.result.series("n[plasma,Ar_star]")
    expected = actual[0] * np.exp(-rate * decay.result.time_s)
    decay_error = float(np.max(np.abs(actual - expected) / expected))

    transport = runner.run("benchmarks/analytic-transport")
    reactor = _config(transport)["reactor"]
    volume_s = float(reactor["zones"]["source"]["volume_m3"])
    volume_t = float(reactor["zones"]["target"]["volume_m3"])
    conductance = float(reactor["edges"]["transfer"]["conductance_m3_s"])
    ns = transport.result.series("n[source,Ar]")
    nt = transport.result.series("n[target,Ar]")
    reference_s = ns[0] * np.exp(-conductance / volume_s * transport.result.time_s)
    reference_t = nt[0] + volume_s / volume_t * (ns[0] - reference_s)
    particles = volume_s * ns + volume_t * nt
    return {
        "decay": {
            **decay.record(),
            "rate_s_inv": rate,
            "max_pointwise_relative_error": decay_error,
        },
        "transport": {
            **transport.record(),
            "source_volume_m3": volume_s,
            "target_volume_m3": volume_t,
            "conductance_m3_s": conductance,
            "source_error": float(np.max(np.abs(ns - reference_s) / reference_s)),
            "target_error": float(np.max(np.abs(nt - reference_t) / reference_t)),
            "particle_conservation_error": float(
                np.max(np.abs(particles - particles[0])) / particles[0]
            ),
        },
    }


def _dc_trace(run: Run) -> dict[str, list[float]]:
    compiled = compile_case(load_case(run.case_file))
    trace: dict[str, list[float]] = {
        "time_s": run.result.time_s.tolist(),
        "plasma_voltage_V": [],
        "current_A": [],
        "reduced_field_Td": [],
    }
    for time_s, state in zip(run.result.time_s, run.result.state, strict=True):
        evaluation = compiled.model.evaluate(float(time_s), state, compiled.segments[0])
        if evaluation.power_coupling is None:
            raise RuntimeError("DC benchmark requires a port result")
        port = evaluation.power_coupling.port_results["dc_series_drive"]
        trace["plasma_voltage_V"].append(float(port.observables["plasma_voltage_V"]))
        trace["current_A"].append(float(port.observables["current_A"]))
        if port.reduced_field_Td is None:
            raise RuntimeError("DC benchmark requires the reduced field")
        trace["reduced_field_Td"].append(float(port.reduced_field_Td))
    return trace


def cross_code_benchmarks(runner: Runner) -> dict[str, Any]:
    crane = runner.run("crane_two_reaction_argon")
    reference = yaml.safe_load(
        (REFERENCES / "crane_two_reaction_argon_reference.yaml").read_text()
    )
    _same_end_time(crane, float(reference["conditions"]["end_time_s"]))
    expected = float(reference["converted_output_final"]["Ar_plus_density_m3"])
    actual = float(crane.result.series("n[plasma,Ar_plus]")[-1])
    zdp = runner.run("zdplaskin_example2")
    saved = yaml.safe_load(
        (REFERENCES / "zdplaskin_example2_summary.yaml").read_text()
    )["saved_output"]
    _same_end_time(zdp, float(saved["final_saved_time_s"]))
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
    trace = _dc_trace(zdp)
    with (REFERENCES / "zdplaskin_example2_circuit.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        circuit_reference = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    circuit = circuit_reference[-1]
    if float(circuit["time_s"]) != float(zdp.result.time_s[-1]):
        raise ValueError("DC circuit reference must use the same terminal time")
    values = [
        ("voltage", trace["plasma_voltage_V"][-1], "voltage_V", "V"),
        ("current", trace["current_A"][-1], "current_A", "A"),
        ("reduced_field", trace["reduced_field_Td"][-1], "reduced_field_Td", "Td"),
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
            "conditions": reference["converted_conditions"],
        },
        "zdplaskin": {
            **zdp.record(),
            "comparisons": rows,
            "reference_values": saved,
            "circuit_trace": trace,
            "circuit_reference": circuit_reference,
        },
    }


def _final(run: Run) -> dict[str, float]:
    return {
        "electron_density_m3": float(
            run.result.series("electron_density_m3[plasma]")[-1]
        ),
        "electron_temperature_eV": float(
            run.result.series("electron_temperature_eV[plasma]")[-1]
        ),
    }


def argon_convergence(runner: Runner) -> tuple[dict[str, Any], Run]:
    runs = [
        runner.run(f"benchmarks/argon-rtol-1e-{exponent}") for exponent in (6, 7, 8)
    ]
    reference = runs[-1]
    rows = [
        {
            **run.record(),
            "rtol": float(_config(run)["solver"]["rtol"]),
            "atol": float(_config(run)["solver"]["atol"]),
            **_final(run),
            "density_waveform_error": _waveform_error(
                run, reference, "electron_density_m3[plasma]"
            ),
            "temperature_waveform_error": _waveform_error(
                run, reference, "electron_temperature_eV[plasma]"
            ),
        }
        for run in runs
    ]
    return {"runs": rows}, runs[0]


def _parameter_value(run: Run, parameter: str) -> float:
    paths: dict[str, tuple[str | int, ...]] = {
        "power": (
            "run",
            "steps",
            0,
            "commands",
            "power_ports",
            "source",
            "absorbed_power_W",
        ),
        "pressure": ("reactor", "zones", "plasma", "initial", "pressure_Pa"),
        "area": ("reactor", "surfaces", "wall", "area_m2"),
    }
    value: Any = _config(run)
    for key in paths[parameter]:
        value = value[key]
    return float(value)


def argon_sensitivity(runner: Runner, baseline: Run) -> list[dict[str, Any]]:
    rows = []
    for parameter in ("power", "pressure", "area"):
        for change in ("minus10", "plus10"):
            run = runner.run(f"benchmarks/argon-{parameter}-{change}")
            final, original = _final(run), _final(baseline)
            rows.append(
                {
                    **run.record(),
                    "parameter": parameter,
                    "factor": _parameter_value(run, parameter)
                    / _parameter_value(baseline, parameter),
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


def event_sampling(runner: Runner) -> list[dict[str, Any]]:
    rows = []
    for microseconds in (40, 20, 10, 5, 2.5, 1.25):
        run = runner.run(f"benchmarks/argon-sampling-{microseconds:g}us")
        config = _config(run)
        ionize = float(run.result.series("gas_event_count[plasma,G_AR_ION]")[-1])
        wall = float(
            run.result.series("boundary_event_count[wall,wall_Ar_plus_neutralization]")[
                -1
            ]
        )
        density = run.result.series("n[plasma,Ar_plus]")
        volume = float(config["reactor"]["zones"]["plasma"]["volume_m3"])
        delta = volume * float(density[-1] - density[0])
        rows.append(
            {
                **run.record(),
                "sample_interval_s": float(config["solver"]["sample_interval_s"]),
                "ionization_count": ionize,
                "wall_count": wall,
                "ion_particle_change": delta,
                "balance_residual": abs(delta - (ionize - wall)) / max(ionize, wall),
            }
        )
    finest = min(rows, key=itemgetter("sample_interval_s"))
    for row in rows:
        row["ionization_relative_to_finest"] = _relative_error(
            row["ionization_count"], finest["ionization_count"]
        )
        row["wall_relative_to_finest"] = _relative_error(
            row["wall_count"], finest["wall_count"]
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    target.mkdir(parents=True, exist_ok=False)
    runner = Runner(target)
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in ("numpy", "scipy", "h5py", "pydantic", "PyYAML", "matplotlib")
        },
        "maxwell": maxwell_benchmark(),
        "analytic": analytic_benchmarks(runner),
        "cross_code": cross_code_benchmarks(runner),
    }
    report["argon_convergence"], baseline = argon_convergence(runner)
    report["software"] = to_plain_mapping(
        baseline.result.metadata["provenance"]["software"]
    )
    report["sensitivity"] = argon_sensitivity(runner, baseline)
    report["event_sampling"] = event_sampling(runner)
    report["inputs"] = runner.inputs()
    (target / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _theory_figures.write_figures(
        report, {run.name: run.result for run in runner.runs}, target / "figures"
    )
    print(target / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
