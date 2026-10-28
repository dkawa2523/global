"""Render benchmark evidence from measured runs and explicit saved references."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.core.result import SimulationResult

BLUE = "#175a80"
TEAL = "#007f74"
GOLD = "#bd6a19"
PURPLE = "#80549b"
REFERENCE = "#30343b"


def _save(plt: Any, figure: Any, axes: Any, target: Path, name: str) -> None:
    for axis in np.asarray(axes, dtype=object).flat:
        axis.grid(which="both", alpha=0.2)
    figure.savefig(target / f"theory-{name}.png", dpi=220)
    plt.close(figure)


def _constant_cross_section(axes: Any, data: list[dict[str, Any]]) -> None:
    for threshold, color, marker in [(0, BLUE, "o"), (2, GOLD, "s")]:
        rows = [r for r in data if r["threshold_eV"] == threshold]
        energy = [r["mean_energy_eV"] for r in rows]
        axes[0, 0].plot(
            energy,
            [r["analytic_m3_s"] for r in rows],
            color=color,
            label=f"Analytic, threshold {threshold} eV",
        )
        axes[0, 0].plot(
            energy,
            [r["actual_m3_s"] for r in rows],
            marker,
            color=color,
            fillstyle="none",
        )
        axes[1, 0].plot(
            energy,
            [r["relative_error"] for r in rows],
            marker + "-",
            color=color,
            label=f"Threshold {threshold} eV",
        )


def _maxwell(plt: Any, report: dict[str, Any], target: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 6.3), layout="constrained")
    _constant_cross_section(axes, report["constant"])
    rows = report["argon"]
    energy = [r["mean_energy_eV"] for r in rows]
    axes[0, 1].semilogy(
        energy,
        [r["actual_m3_s"] for r in rows],
        "o-",
        color=BLUE,
        label="Interval integral",
    )
    axes[0, 1].semilogy(
        energy,
        [r["quadrature_m3_s"] for r in rows],
        "+",
        color=REFERENCE,
        markersize=9,
        label="Independent quadrature",
    )
    axes[1, 1].plot(energy, [r["relative_error"] for r in rows], "o-", color=BLUE)
    axes[0, 0].set(title="Constant cross section", ylabel="Coefficient [m$^3$/s]")
    axes[0, 1].set(title="Ar ionization cross section", ylabel="Coefficient [m$^3$/s]")
    for axis in axes[0]:
        axis.legend(fontsize=8)
    for axis in axes[1]:
        axis.set(xlabel="Mean electron energy [eV]", ylabel="Relative difference")
    _save(plt, figure, axes, target, "maxwell")


def _decay(
    plt: Any, result: SimulationResult, data: dict[str, Any], target: Path
) -> None:
    actual = result.series("n[plasma,Ar_star]")
    expected = actual[0] * np.exp(-data["rate_s_inv"] * result.time_s)
    figure, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.2), sharex=True, layout="constrained"
    )
    axes[0].semilogy(
        result.time_s, expected, color=REFERENCE, label="Analytic solution"
    )
    axes[0].semilogy(
        result.time_s,
        actual,
        "o",
        markevery=5,
        fillstyle="none",
        color=BLUE,
        label="BDF (every fifth saved point shown)",
    )
    axes[0].set_ylabel("Ar* density [m$^{-3}$]")
    axes[0].legend(fontsize=8)
    axes[1].plot(result.time_s, np.abs(actual - expected) / expected, color=BLUE)
    axes[1].set(xlabel="Time [s]", ylabel="Pointwise relative error")
    _save(plt, figure, axes, target, "decay")


def _transport(
    plt: Any, result: SimulationResult, data: dict[str, Any], target: Path
) -> None:
    va, vb = data["source_volume_m3"], data["target_volume_m3"]
    na, nb = result.series("n[source,Ar]"), result.series("n[target,Ar]")
    exact_a = na[0] * np.exp(-data["conductance_m3_s"] / va * result.time_s)
    exact_b = nb[0] + va / vb * (na[0] - exact_a)
    particles = va * na + vb * nb
    figure, axes = plt.subplots(
        3, 1, figsize=(7.4, 6.7), sharex=True, layout="constrained"
    )
    for actual, exact, color, marker, label in [
        (na, exact_a, BLUE, "o", "Source"),
        (nb, exact_b, GOLD, "s", "Target"),
    ]:
        axes[0].plot(result.time_s, exact, color=color, label=f"{label} analytic")
        axes[0].plot(
            result.time_s, actual, marker, markevery=8, color=color, fillstyle="none"
        )
        axes[1].plot(
            result.time_s, np.abs(actual - exact) / exact, color=color, label=label
        )
    axes[0].set_ylabel("Ar density [m$^{-3}$]")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Pointwise relative error")
    axes[1].legend(fontsize=8)
    axes[2].plot(result.time_s, (particles - particles[0]) / particles[0], color=BLUE)
    axes[2].set(xlabel="Time [s]", ylabel="Particle drift (N-N0)/N0")
    _save(plt, figure, axes, target, "transport")


def _crane(
    plt: Any, result: SimulationResult, data: dict[str, Any], target: Path
) -> None:
    density = result.series("n[plasma,Ar_plus]")
    conditions = data["conditions"]
    steady = conditions["ionization_rate_m3_s"] / conditions["recombination_rate_m6_s"]
    figure, axes = plt.subplots(2, 1, figsize=(7.4, 5.3), layout="constrained")
    axes[0].semilogy(result.time_s * 1e6, density, color=BLUE, label="This code")
    axes[0].plot(
        result.time_s[-1] * 1e6,
        data["reference"],
        "o",
        color=GOLD,
        fillstyle="none",
        label="CRANE saved endpoint",
    )
    axes[0].axhline(steady, color=REFERENCE, linestyle=":", label="ki / kr")
    axes[0].set(xlabel="Time [microseconds]", ylabel="Ar+ density [m$^{-3}$]")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        [0, 1, 2],
        [density[-1] / steady, data["reference"] / steady, 1],
        "o",
        color=BLUE,
    )
    axes[1].set_xticks([0, 1, 2], ["This code", "CRANE saved endpoint", "ki / kr"])
    axes[1].set(ylabel="Density / (ki / kr)", title="Endpoint detail (focused scale)")
    axes[1].ticklabel_format(axis="y", useOffset=False, style="plain")
    _save(plt, figure, axes, target, "crane")


def _dc_species(
    plt: Any, result: SimulationResult, data: dict[str, Any], target: Path
) -> None:
    saved = data["reference_values"]
    figure, axis = plt.subplots(figsize=(7.4, 4.1), layout="constrained")
    for species, label, color, style in [
        ("Ar_star", "Ar*", GOLD, "-"),
        ("Ar_plus", "Ar+", BLUE, "--"),
        ("Ar2_plus", "Ar2+", PURPLE, "-."),
    ]:
        values = result.series(f"n[plasma,{species}]")
        axis.semilogy(
            result.time_s * 1e3,
            np.where(values > 0, values, np.nan),
            color=color,
            linestyle=style,
            label=label,
        )
        reference = float(saved[f"final_saved_{species}_density_m3"])
        axis.plot(
            result.time_s[-1] * 1e3, reference, "o", color=color, fillstyle="none"
        )
    axis.semilogy(
        result.time_s * 1e3,
        result.series("electron_density_m3[plasma]"),
        color=REFERENCE,
        linestyle=":",
        label="Electron",
    )
    axis.plot(
        float(saved["peak_saved_electron_density_time_s"]) * 1e3,
        float(saved["peak_saved_electron_density_m3"]),
        "x",
        color=REFERENCE,
        markersize=8,
        label="Reference saved electron peak",
    )
    axis.set(
        xlabel="Time [ms]",
        ylabel="Density [m$^{-3}$]",
        title="Computed histories; open circles: reference endpoints",
    )
    axis.legend(fontsize=8)
    _save(plt, figure, axis, target, "dc-species")


def _dc_circuit(plt: Any, data: dict[str, Any], target: Path) -> None:
    actual, reference = data["circuit_trace"], data["circuit_reference"]
    figure, axes = plt.subplots(
        3, 1, figsize=(7.4, 6.5), sharex=True, layout="constrained"
    )
    for axis, key, ref_key, label in zip(
        axes,
        ("plasma_voltage_V", "current_A", "reduced_field_Td"),
        ("voltage_V", "current_A", "reduced_field_Td"),
        ("Plasma voltage [V]", "Current [A]", "Reduced field [Td]"),
        strict=True,
    ):
        axis.plot(
            np.asarray(actual["time_s"]) * 1e3,
            actual[key],
            color=BLUE,
            label="This code at saved states",
        )
        axis.plot(
            [row["time_s"] * 1e3 for row in reference],
            [row[ref_key] for row in reference],
            color=REFERENCE,
            linestyle="--",
            label="ZDPlaskin saved output",
        )
        axis.set_ylabel(label)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Time [ms]")
    _save(plt, figure, axes, target, "dc-circuit")


def _transient(plt: Any, result: SimulationResult, target: Path) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.2), sharex=True, layout="constrained"
    )
    for axis, name, label, color in zip(
        axes,
        ("electron_density_m3", "electron_temperature_eV"),
        ("Electron density [m$^{-3}$]", "Equivalent electron temperature [eV]"),
        (BLUE, TEAL),
        strict=True,
    ):
        axis.plot(result.time_s * 1e3, result.series(f"{name}[plasma]"), color=color)
        axis.axvline(1, color=REFERENCE, linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Time [ms]")
    _save(plt, figure, axes, target, "transient")


def _tolerances(plt: Any, runs: Mapping[str, SimulationResult], target: Path) -> None:
    reference = runs["argon-rtol-1e-8"]
    figure, axes = plt.subplots(
        2, 1, figsize=(7.4, 5.3), sharex=True, layout="constrained"
    )
    for exponent, color, style in [(6, BLUE, "-"), (7, GOLD, "--")]:
        result = runs[f"argon-rtol-1e-{exponent}"]
        if not np.array_equal(result.time_s, reference.time_s):
            raise ValueError("tolerance figures require identical saved times")
        for axis, name in zip(
            axes, ("electron_density_m3", "electron_temperature_eV"), strict=True
        ):
            actual = result.series(f"{name}[plasma]")
            expected = reference.series(f"{name}[plasma]")
            error = np.abs(actual - expected) / np.max(np.abs(expected))
            axis.plot(
                result.time_s * 1e3,
                error,
                color=color,
                linestyle=style,
                label=f"rtol 1e-{exponent}, atol 1e-{exponent + 8}",
            )
    for axis, quantity in zip(axes, ("Density", "Temperature"), strict=True):
        axis.set_ylabel(f"{quantity} normalized difference")
        axis.axvline(1, color=REFERENCE, linestyle=":", linewidth=0.8)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Time [ms]")
    _save(plt, figure, axes, target, "tolerances")


def _sampling(plt: Any, rows: list[dict[str, Any]], target: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 3.8), layout="constrained")
    for key, label, marker, color in [
        ("ionization_relative_to_finest", "Ionization count", "o", BLUE),
        ("wall_relative_to_finest", "Wall event count", "s", TEAL),
    ]:
        axis.loglog(
            [r["sample_interval_s"] * 1e6 for r in rows[:-1]],
            [r[key] for r in rows[:-1]],
            marker + "-",
            label=label,
            color=color,
        )
    axis.set(
        xlabel="Saved interval [microseconds]",
        ylabel="Relative difference from 1.25 microseconds",
    )
    axis.set_xticks([2.5, 5, 10, 20, 40], ["2.5", "5", "10", "20", "40"])
    axis.tick_params(axis="x", which="minor", labelbottom=False)
    axis.legend()
    _save(plt, figure, axis, target, "sampling")


def _sensitivity(plt: Any, rows: list[dict[str, Any]], target: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 3.8), layout="constrained")
    positions = np.arange(len(rows))
    for shift, key, label, color, hatch in [
        (-0.18, "density_change_percent", "Final electron density", BLUE, None),
        (0.18, "temperature_change_percent", "Final electron temperature", TEAL, "//"),
    ]:
        axis.bar(
            positions + shift,
            [r[key] for r in rows],
            width=0.36,
            label=label,
            color=color,
            hatch=hatch,
        )
    axis.set_xticks(
        positions, [f"{r['parameter']}\n{100 * (r['factor'] - 1):+.0f}%" for r in rows]
    )
    axis.set_ylabel("Change at 2 ms [%]")
    axis.axhline(0, color=REFERENCE, linewidth=0.8)
    axis.legend(fontsize=9)
    _save(plt, figure, axis, target, "sensitivity")


def write_figures(
    report: dict[str, Any], runs: Mapping[str, SimulationResult], target: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target.mkdir()
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    _maxwell(plt, report["maxwell"], target)
    _decay(plt, runs["analytic-decay"], report["analytic"]["decay"], target)
    _transport(plt, runs["analytic-transport"], report["analytic"]["transport"], target)
    _crane(plt, runs["crane_two_reaction_argon"], report["cross_code"]["crane"], target)
    _dc_species(
        plt, runs["zdplaskin_example2"], report["cross_code"]["zdplaskin"], target
    )
    _dc_circuit(plt, report["cross_code"]["zdplaskin"], target)
    _transient(plt, runs["argon-rtol-1e-6"], target)
    _tolerances(plt, runs, target)
    _sampling(plt, report["event_sampling"], target)
    _sensitivity(plt, report["sensitivity"], target)
