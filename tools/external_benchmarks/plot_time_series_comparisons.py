from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plasma_global.workflows.context import load_case_from_yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks'
FIGURE_DIR = OUTPUT_DIR / 'timeseries_figures'


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _read_solution(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(path, 'r') as h5:
        labels = [x.decode() if isinstance(x, bytes) else str(x) for x in h5['state_labels'][:]]
        time_s = np.asarray(h5['t_s'][:], dtype=float)
        values = np.asarray(h5['y'][:], dtype=float)
    return time_s, {label: values[i] for i, label in enumerate(labels)}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def _positive(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 1.0e-300)


def _positive_or_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > 0.0, values, np.nan)


def _format(value: float) -> str:
    if not math.isfinite(value):
        return ''
    if abs(value) >= 1.0e4 or (value != 0.0 and abs(value) < 1.0e-3):
        return f'{value:.2e}'
    return f'{value:.3g}'


def _zone_volumes(case_path: Path) -> dict[str, float]:
    loaded = load_case_from_yaml(case_path)
    return {zone.zone_id: float(zone.volume_m3) for zone in loaded.chamber.zones}


def _volume_weighted_species(data: dict[str, np.ndarray], species: str, volumes: dict[str, float]) -> np.ndarray:
    numerator: np.ndarray | None = None
    denominator = 0.0
    for zone_id, volume in volumes.items():
        label = f'n[{zone_id},{species}]'
        if label not in data:
            continue
        contribution = data[label] * volume
        numerator = contribution if numerator is None else numerator + contribution
        denominator += volume
    if numerator is None or denominator <= 0.0:
        raise KeyError(f'No density state found for species {species}')
    return numerator / denominator


def plot_crane_species() -> Path:
    time_s, data = _read_solution(ROOT / 'examples' / 'outputs' / 'crane_two_reaction_argon' / 'solution.h5')
    reference = _load_yaml(ROOT / 'examples' / 'external' / 'crane_two_reaction_argon_reference.yaml')
    ref_final = reference['converted_output_final']
    final_t = float(reference['committed_output_final']['time_s'])
    time_us = time_s * 1.0e6
    initial_ar = float(data['n[plasma,Ar]'][0])

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    ax = axes[0]
    ax.plot(time_us, (1.0 - data['n[plasma,Ar]'] / initial_ar) * 1.0e6, color='#4c78a8', label='This code Ar depletion')
    ax.scatter(
        [final_t * 1.0e6],
        [(1.0 - float(ref_final['Ar_density_m3']) / initial_ar) * 1.0e6],
        color='#e15759',
        marker='x',
        s=70,
        label='External CRANE final Ar depletion',
    )
    ax.set_title('CRANE neutral species')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('Ar depletion from initial value (ppm)')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ar_plus = data['n[plasma,Ar_plus]']
    ax.plot(time_us, _positive(ar_plus), color='#59a14f', label='This code Ar+ / e')
    ax.scatter([final_t * 1.0e6], [float(ref_final['Ar_plus_density_m3'])], color='#f28e2b', marker='x', s=70, label='External CRANE final Ar+ / e')
    ax.set_yscale('log')
    ax.set_title('CRANE charged species')
    ax.set_xlabel('time (us)')
    ax.set_ylabel('density (m^-3, log)')
    ax.grid(which='both', alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle('CRANE composition: this code time series with external final reference')
    fig.text(0.01, 0.01, 'External CRANE time series is not stored in this repository; x markers show committed external final values.', fontsize=8, color='#555555')
    return _save(fig, FIGURE_DIR / 'timeseries_crane_species_this_code_with_external_final.png')


def plot_zdplaskin_species() -> Path:
    time_s, data = _read_solution(ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate' / 'solution.h5')
    reference = _load_yaml(ROOT / 'examples' / 'external' / 'zdplaskin_example2_summary.yaml')
    saved = reference['saved_output']
    time_ms = time_s * 1.0e3
    electron = data['n[plasma,Ar_plus]'] + data['n[plasma,Ar2_plus]']

    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    series = [
        ('This code e', electron, '#e15759'),
        ('This code Ar*', data['n[plasma,Ar_star]'], '#4c78a8'),
        ('This code Ar+', data['n[plasma,Ar_plus]'], '#59a14f'),
        ('This code Ar2+', data['n[plasma,Ar2_plus]'], '#f28e2b'),
    ]
    for label, values, color in series:
        ax.plot(time_ms, _positive_or_nan(values), label=label, color=color)

    final_t_ms = float(saved['final_saved_time_s']) * 1.0e3
    marker_rows = [
        ('External final e', saved['final_saved_electron_density_m3'], '#e15759'),
        ('External final Ar*', saved['final_saved_Ar_star_density_m3'], '#4c78a8'),
        ('External final Ar+', saved['final_saved_Ar_plus_density_m3'], '#59a14f'),
        ('External final Ar2+', saved['final_saved_Ar2_plus_density_m3'], '#f28e2b'),
    ]
    for label, value, color in marker_rows:
        ax.scatter([final_t_ms], [float(value)], color=color, marker='x', s=72, label=label)
    ax.scatter(
        [float(saved['peak_saved_electron_density_time_s']) * 1.0e3],
        [float(saved['peak_saved_electron_density_m3'])],
        color='#222222',
        marker='*',
        s=110,
        label='External peak e',
    )
    ax.set_yscale('log')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('density (m^-3, log)')
    ax.set_title('ZDPlaskin composition: this code time series with external markers')
    ax.grid(which='both', alpha=0.25)
    ax.legend(ncols=2, frameon=False, fontsize=8)
    fig.text(0.01, 0.01, 'External ZDPlaskin species time series is not stored here; markers show committed final and peak values.', fontsize=8, color='#555555')
    return _save(fig, FIGURE_DIR / 'timeseries_zdplaskin_species_this_code_with_external_markers.png')


def plot_zdplaskin_circuit() -> Path:
    local = pd.read_csv(ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate' / 'observables.csv')
    external = pd.read_csv(ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    ax = axes[0]
    ax.plot(local['time_s'] * 1.0e3, local['EoverN_plasma_Td'], label='This code E/N', color='#4c78a8')
    ax.plot(external['time_s'] * 1.0e3, external['reduced_field_Td'], label='External ZDPlaskin E/N', color='#e15759', linestyle='--')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('E/N (Td)')
    ax.set_title('Reduced field')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(local['time_s'] * 1.0e3, local['total_absorbed_power_W'], label='This code absorbed power', color='#59a14f')
    ax.plot(external['time_s'] * 1.0e3, external['absorbed_power_W'], label='External ZDPlaskin power', color='#f28e2b', linestyle='--')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('absorbed power (W)')
    ax.set_title('Absorbed power')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.suptitle('ZDPlaskin circuit quantities: external time series vs this code')
    fig.text(0.01, 0.01, 'Both panels compare stored external ZDPlaskin circuit CSV against this code observables.', fontsize=8, color='#555555')
    return _save(fig, FIGURE_DIR / 'timeseries_zdplaskin_circuit_external_vs_this_code.png')


def plot_zdplaskin_circuit_ratios() -> Path:
    local = pd.read_csv(ROOT / 'examples' / 'outputs' / 'zdplaskin_example2_surrogate' / 'observables.csv')
    external = pd.read_csv(ROOT / 'examples' / 'external' / 'zdplaskin_example2_circuit.csv')
    ext_time = external['time_s'].to_numpy(dtype=float)
    local_time = local['time_s'].to_numpy(dtype=float)
    common = ext_time <= local_time.max()
    ext_time = ext_time[common]
    ext_en = external.loc[common, 'reduced_field_Td'].to_numpy(dtype=float)
    ext_power = external.loc[common, 'absorbed_power_W'].to_numpy(dtype=float)
    local_en = np.interp(ext_time, local_time, local['EoverN_plasma_Td'].to_numpy(dtype=float))
    local_power = np.interp(ext_time, local_time, local['total_absorbed_power_W'].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(ext_time * 1.0e3, ext_en / np.maximum(local_en, 1.0e-300), label='External ZDPlaskin E/N / this code E/N', color='#4c78a8')
    ax.plot(ext_time * 1.0e3, ext_power / np.maximum(local_power, 1.0e-300), label='External ZDPlaskin power / this code power', color='#f28e2b')
    ax.axhline(1.0, color='#222222', linewidth=1.0, label='perfect match')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('external / this code ratio')
    ax.set_title('ZDPlaskin circuit comparison ratios')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, FIGURE_DIR / 'comparison_zdplaskin_circuit_external_over_this_code.png')


def plot_pygmol_species() -> Path:
    time_s, data = _read_solution(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'solution.h5')
    volumes = _zone_volumes(ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml')
    local_ar = _volume_weighted_species(data, 'Ar', volumes)
    local_ar_plus = _volume_weighted_species(data, 'Ar_plus', volumes)
    external = pd.read_csv(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'comparison_pygmol_solution.csv')

    fig, ax = plt.subplots(figsize=(10.4, 5.3))
    ax.plot(time_s * 1.0e3, _positive(local_ar), label='This code Ar', color='#4c78a8')
    ax.plot(time_s * 1.0e3, _positive(local_ar_plus), label='This code Ar+ / e', color='#59a14f')
    ax.plot(external['t'] * 1.0e3, _positive(external['Ar'].to_numpy(dtype=float)), label='External PyGMol Ar', color='#4c78a8', linestyle='--')
    ax.plot(external['t'] * 1.0e3, _positive(external['e'].to_numpy(dtype=float)), label='External PyGMol e / Ar+', color='#e15759', linestyle='--')
    ax.set_yscale('log')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('density (m^-3, log)')
    ax.set_title('PyGMol composition time series: external vs this code')
    ax.grid(which='both', alpha=0.25)
    ax.legend(ncols=2, frameon=False)
    fig.text(0.01, 0.01, 'External = PyGMol compact single-cylinder surrogate. This code = volume-weighted two-zone local case.', fontsize=8, color='#555555')
    return _save(fig, FIGURE_DIR / 'timeseries_pygmol_species_external_vs_this_code.png')


def plot_pygmol_properties() -> Path:
    local = pd.read_csv(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'observables.csv')
    external = pd.read_csv(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'comparison_pygmol_solution.csv')

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    ax = axes[0]
    ax.plot(local['time_s'] * 1.0e3, _positive(local['electron_density_m3'].to_numpy(dtype=float)), label='This code e', color='#4c78a8')
    ax.plot(external['t'] * 1.0e3, _positive(external['e'].to_numpy(dtype=float)), label='External PyGMol e', color='#e15759', linestyle='--')
    ax.set_yscale('log')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('electron density (m^-3, log)')
    ax.set_title('Electron density')
    ax.grid(which='both', alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(local['time_s'] * 1.0e3, local['mean_electron_energy_eV'], label='This code mean energy', color='#59a14f')
    ax.plot(external['t'] * 1.0e3, 1.5 * external['T_e'], label='External PyGMol 1.5 Te', color='#f28e2b', linestyle='--')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('energy equivalent (eV)')
    ax.set_title('Electron energy')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(local['time_s'] * 1.0e3, local['total_absorbed_power_W'], label='This code power', color='#b07aa1')
    ax.plot(external['t'] * 1.0e3, external['P'], label='External PyGMol power', color='#f28e2b', linestyle='--')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('absorbed power (W)')
    ax.set_title('Absorbed power')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle('PyGMol plasma quantities: external time series vs this code')
    fig.text(0.01, 0.01, 'Energy definitions differ: this code reports EEDF mean energy; PyGMol reports Te, shown here as 1.5 Te.', fontsize=8, color='#555555')
    return _save(fig, FIGURE_DIR / 'timeseries_pygmol_properties_external_vs_this_code.png')


def plot_pygmol_ratios() -> Path:
    local = pd.read_csv(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'observables.csv')
    external = pd.read_csv(ROOT / 'examples' / 'outputs' / 'argon_lxcat_icp_baseline' / 'comparison_pygmol_solution.csv')
    ext_time = external['t'].to_numpy(dtype=float)
    local_time = local['time_s'].to_numpy(dtype=float)
    common = ext_time <= local_time.max()
    ext_time = ext_time[common]
    ext_e = external.loc[common, 'e'].to_numpy(dtype=float)
    ext_energy = 1.5 * external.loc[common, 'T_e'].to_numpy(dtype=float)
    ext_power = external.loc[common, 'P'].to_numpy(dtype=float)
    local_e = np.interp(ext_time, local_time, local['electron_density_m3'].to_numpy(dtype=float))
    local_energy = np.interp(ext_time, local_time, local['mean_electron_energy_eV'].to_numpy(dtype=float))
    local_power = np.interp(ext_time, local_time, local['total_absorbed_power_W'].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(ext_time * 1.0e3, ext_e / np.maximum(local_e, 1.0e-300), label='External PyGMol e / this code e', color='#4c78a8')
    ax.plot(ext_time * 1.0e3, ext_energy / np.maximum(local_energy, 1.0e-300), label='External PyGMol energy / this code energy', color='#59a14f')
    power_mask = (local_power > 1.0) & (ext_power > 1.0)
    ax.plot(ext_time[power_mask] * 1.0e3, ext_power[power_mask] / local_power[power_mask], label='External PyGMol power / this code power', color='#f28e2b')
    ax.axhline(1.0, color='#222222', linewidth=1.0, label='perfect match')
    ax.set_yscale('log')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('external / this code ratio (log)')
    ax.set_title('PyGMol comparison ratios over time')
    ax.grid(which='both', alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, FIGURE_DIR / 'comparison_pygmol_external_over_this_code_timeseries.png')


def write_summary(paths: list[Path]) -> None:
    summary = OUTPUT_DIR / 'timeseries_graph_summary.md'
    lines = [
        '# Time-Series Benchmark Graph Summary',
        '',
        'Generated figures:',
        '',
    ]
    for path in paths:
        lines.append(f'- `{path.relative_to(ROOT).as_posix()}`')
    lines.extend(
        [
            '',
            'Data availability:',
            '',
            '- CRANE external species time series is not stored; the graph overlays committed external final values on this code time series.',
            '- ZDPlaskin external species time series is not stored; the species graph overlays committed final and peak markers on this code time series.',
            '- ZDPlaskin external circuit time series is stored and compared directly against this code observables.',
            '- PyGMol external time series was regenerated as `comparison_pygmol_solution.csv` and compared directly against this code outputs.',
            '',
        ]
    )
    summary.write_text('\n'.join(lines), encoding='utf-8')

    artifacts = OUTPUT_DIR / 'timeseries_graph_artifacts.csv'
    with artifacts.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['artifact'])
        for path in paths:
            writer.writerow([path.relative_to(ROOT).as_posix()])
        writer.writerow([summary.relative_to(ROOT).as_posix()])


def main() -> int:
    paths = [
        plot_crane_species(),
        plot_zdplaskin_species(),
        plot_zdplaskin_circuit(),
        plot_zdplaskin_circuit_ratios(),
        plot_pygmol_species(),
        plot_pygmol_properties(),
        plot_pygmol_ratios(),
    ]
    write_summary(paths)
    for path in paths:
        print(path.relative_to(ROOT))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
