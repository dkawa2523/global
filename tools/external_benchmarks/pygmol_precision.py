from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import constants
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.workflows.context import load_case_from_yaml
from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_BENCHMARK_MODEL,
    DEFAULT_CASE,
    _pygmol_parameters,
    case_geometry,
    load_benchmark_model,
    run_pygmol,
)


DEFAULT_OUTPUT_DIR = ROOT / 'examples' / 'outputs' / 'external_benchmarks' / 'diagnostic_suite' / 'pygmol_precision'
INITIAL_DENSITIES = {'Ar': 1.0, 'Ar+': 1.0e-9}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_rel_error(candidate: float, reference: float, floor: float = 1.0e-30) -> float:
    return abs(float(candidate) - float(reference)) / max(abs(float(reference)), floor)


def _nrmse(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = max(float(np.nanmax(reference) - np.nanmin(reference)), float(np.nanmean(np.abs(reference))), 1.0e-30)
    return float(np.sqrt(np.nanmean((candidate - reference) ** 2)) / scale)


def _sanitize_power_series(t_power: np.ndarray, power: np.ndarray, t_end: float) -> tuple[np.ndarray, np.ndarray]:
    times = [float('-inf'), *[float(v) for v in t_power], float('inf')]
    values = [float(power[0]), *[float(v) for v in power], float(power[-1])]
    dt = 1.0e-5 * float(t_end)
    for i in range(len(times) - 1):
        if times[i] == times[i + 1]:
            times[i] -= dt
            times[i + 1] += dt
    if sorted(times) != times:
        raise ValueError('Power time series is not monotonic after PyGMol-style sanitization.')
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


@dataclass
class SameFootingChemistry:
    species_ids: list[str]
    charges: np.ndarray
    masses_kg: np.ndarray
    lj_sigma_m: np.ndarray
    sticking: np.ndarray
    return_matrix: np.ndarray
    reaction_ids: list[str]
    arrh_a: np.ndarray
    arrh_b: np.ndarray
    arrh_c: np.ndarray
    electron_energy_losses: np.ndarray
    elastic: np.ndarray
    electron_lhs: np.ndarray
    electron_rhs: np.ndarray
    species_lhs: np.ndarray
    species_rhs: np.ndarray

    @property
    def species_net(self) -> np.ndarray:
        return self.species_rhs - self.species_lhs

    @property
    def electron_net(self) -> np.ndarray:
        return self.electron_rhs - self.electron_lhs


@dataclass
class SameFootingParameters:
    radius_m: float
    length_m: float
    pressure_Pa: float
    temp_e_eV: float
    temp_n_K: float
    feeds_sccm: dict[str, float]
    t_power_s: np.ndarray
    power_W: np.ndarray
    t_end_s: float

    @property
    def volume_m3(self) -> float:
        return math.pi * self.radius_m**2 * self.length_m

    @property
    def area_m2(self) -> float:
        return 2.0 * math.pi * self.radius_m * (self.radius_m + self.length_m)

    @property
    def diffusion_length_m(self) -> float:
        return ((math.pi / self.length_m) ** 2 + (2.405 / self.radius_m) ** 2) ** -0.5


class LocalPyGMolSameFootingODE:
    """Local tools-only implementation of the PyGMol compact global-model ODE.

    This class exists to make the PyGMol precision benchmark same-footing without
    importing PyGMol's equations for the local side. It is intentionally kept under
    tools so production plasma_global physics is not tuned to PyGMol.
    """

    mtorr = 0.133322
    sccm = 4.485e17
    pi = constants.pi
    m_e = constants.m_e
    k = constants.k
    e = constants.e
    epsilon_0 = constants.epsilon_0

    def __init__(self, chemistry: SameFootingChemistry, params: SameFootingParameters) -> None:
        self.chemistry = chemistry
        self.params = params
        self.mask_positive = chemistry.charges > 0
        self.mask_negative = chemistry.charges < 0
        self.mask_neutral = chemistry.charges == 0
        self.mask_electron_reaction = chemistry.electron_lhs > 0
        self.mask_elastic = chemistry.elastic.astype(bool)
        self.feeds = np.asarray([params.feeds_sccm.get(sp_id, 0.0) for sp_id in chemistry.species_ids], dtype=float)
        masses_i = chemistry.masses_kg[:, np.newaxis]
        masses_k = chemistry.masses_kg[np.newaxis, :]
        self.reduced_mass = masses_i * masses_k / (masses_i + masses_k)
        self.sigma_sc = (chemistry.lj_sigma_m[:, np.newaxis] + chemistry.lj_sigma_m[np.newaxis, :]) ** 2
        self.sigma_sc[np.diag([True] * len(chemistry.species_ids))] = 0.0
        mean_cation_mass = float(np.mean(chemistry.masses_kg[self.mask_positive])) if np.any(self.mask_positive) else float('nan')
        self.sheath_voltage_per_ev = math.log(math.sqrt(mean_cation_mass / (2.0 * math.pi * self.m_e)))

    @classmethod
    def from_model(cls, model_spec: dict[str, Any], params: SameFootingParameters) -> LocalPyGMolSameFootingODE:
        species = model_spec['species']
        reactions = model_spec['reactions']
        chemistry = SameFootingChemistry(
            species_ids=[str(item['id']) for item in species],
            charges=np.asarray([int(item['charge']) for item in species], dtype=float),
            masses_kg=np.asarray([float(item['mass_amu']) * constants.atomic_mass for item in species], dtype=float),
            lj_sigma_m=np.asarray([float(item['lj_sigma']) * 1.0e-10 for item in species], dtype=float),
            sticking=np.asarray([float(item['surface_sticking']) for item in species], dtype=float),
            return_matrix=np.asarray(model_spec['surface_return_matrix'], dtype=float),
            reaction_ids=[str(item['id']) for item in reactions],
            arrh_a=np.asarray([float(item['arrhenius']['a']) for item in reactions], dtype=float),
            arrh_b=np.asarray([float(item['arrhenius']['b']) for item in reactions], dtype=float),
            arrh_c=np.asarray([float(item['arrhenius']['c_eV']) for item in reactions], dtype=float),
            electron_energy_losses=np.asarray([float(item['electron_energy_loss_eV']) for item in reactions], dtype=float),
            elastic=np.asarray([bool(item['elastic']) for item in reactions], dtype=bool),
            electron_lhs=np.asarray([float(item['electron_stoich']['lhs']) for item in reactions], dtype=float),
            electron_rhs=np.asarray([float(item['electron_stoich']['rhs']) for item in reactions], dtype=float),
            species_lhs=np.asarray([item['species_stoich_lhs'] for item in reactions], dtype=float),
            species_rhs=np.asarray([item['species_stoich_rhs'] for item in reactions], dtype=float),
        )
        return cls(chemistry, params)

    def initial_state(self, initial_densities: dict[str, float] | None = None) -> np.ndarray:
        n_total = self.params.pressure_Pa / (self.k * self.params.temp_n_K)
        if initial_densities:
            n0 = np.asarray([float(initial_densities.get(sp_id, 0.0)) for sp_id in self.chemistry.species_ids], dtype=float)
            n0 = n0 / max(float(np.sum(n0)), 1.0e-300) * n_total
            ne = float(np.sum(n0 * self.chemistry.charges))
        else:
            n0 = np.zeros(len(self.chemistry.species_ids))
            ne = n_total * 1.0e-15
            n0[self.mask_positive] = ne / max(float(np.sum(self.chemistry.charges[self.mask_positive])), 1.0)
            feed_mask = self.feeds > 0.0
            if np.any(feed_mask):
                n0[feed_mask] = self.feeds[feed_mask] / float(np.sum(self.feeds[feed_mask])) * (n_total - float(np.sum(n0)))
            else:
                n0[self.mask_neutral] = (n_total - float(np.sum(n0))) / max(int(np.sum(self.mask_neutral)), 1)
        return np.r_[n0, 1.5 * ne * self.params.temp_e_eV]

    def density(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y[:-1], dtype=float)

    def electron_density(self, n: np.ndarray) -> float:
        return float(np.sum(n * self.chemistry.charges))

    def electron_temperature_eV(self, n: np.ndarray, rho: float) -> float:
        ne = max(self.electron_density(n), 1.0e-300)
        temp_n_eV = self.params.temp_n_K * self.k / self.e
        return max(float(rho) / ne * 2.0 / 3.0, temp_n_eV)

    def power_W(self, t: float) -> float:
        if len(set(float(v) for v in self.params.power_W)) == 1:
            return float(self.params.power_W[0])
        return float(np.interp(float(t), self.params.t_power_s, self.params.power_W))

    def reaction_rate_coefficients(self, temp_e: float) -> np.ndarray:
        k = np.empty_like(self.chemistry.arrh_a)
        e_mask = self.mask_electron_reaction
        k[e_mask] = (
            self.chemistry.arrh_a[e_mask]
            * temp_e ** self.chemistry.arrh_b[e_mask]
            * np.exp(-self.chemistry.arrh_c[e_mask] / max(temp_e, 1.0e-300))
        )
        k[~e_mask] = (
            self.chemistry.arrh_a[~e_mask]
            * (self.params.temp_n_K / 300.0) ** self.chemistry.arrh_b[~e_mask]
            * np.exp(-self.chemistry.arrh_c[~e_mask] / max(self.params.temp_n_K, 1.0e-300))
        )
        return k

    def reaction_rates(self, n: np.ndarray, temp_e: float) -> np.ndarray:
        ne = self.electron_density(n)
        n_all = np.r_[ne, n, float(np.sum(n))]
        lhs = np.c_[self.chemistry.electron_lhs, self.chemistry.species_lhs, np.zeros(len(self.chemistry.reaction_ids))]
        mass_action = np.prod(n_all[np.newaxis, :] ** lhs, axis=1)
        return mass_action * self.reaction_rate_coefficients(temp_e)

    def ion_temperature_K(self, pressure_Pa: float) -> float:
        if pressure_Pa > self.mtorr:
            return (0.5 * self.e / self.k - self.params.temp_n_K) / (pressure_Pa / self.mtorr) + self.params.temp_n_K
        return 0.5 * self.e / self.k

    def mean_speeds(self, temp_i_K: float) -> np.ndarray:
        speeds = np.empty(len(self.chemistry.species_ids))
        speeds[self.mask_neutral] = np.sqrt(8.0 * self.k * self.params.temp_n_K / (math.pi * self.chemistry.masses_kg[self.mask_neutral]))
        speeds[~self.mask_neutral] = np.sqrt(8.0 * self.k * temp_i_K / (math.pi * self.chemistry.masses_kg[~self.mask_neutral]))
        return speeds

    def debye_length_m(self, ne: float, temp_e: float) -> float:
        return math.sqrt(self.epsilon_0 * temp_e / self.e / max(ne, 1.0e-300))

    def scattering_cross_sections(self, speeds: np.ndarray, debye_length: float) -> np.ndarray:
        sigma = np.array(self.sigma_sc, copy=True)
        non_neutral = ~self.mask_neutral
        if np.any(non_neutral):
            charges = self.chemistry.charges[non_neutral]
            reduced = self.reduced_mass[np.ix_(non_neutral, non_neutral)]
            v = speeds[non_neutral]
            b0 = (
                self.e**2
                * np.abs(charges[:, np.newaxis] * charges[np.newaxis, :])
                / (2.0 * math.pi * self.epsilon_0)
                / np.maximum(reduced * v[:, np.newaxis] ** 2, 1.0e-300)
            )
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma[np.ix_(non_neutral, non_neutral)] = math.pi * b0**2 * np.log(2.0 * debye_length / b0)
        sigma[~np.isfinite(sigma)] = 0.0
        sigma[np.diag([True] * len(self.chemistry.species_ids))] = 0.0
        return sigma

    def diffusivities(self, n: np.ndarray, temp_i_K: float, temp_e: float, ne: float) -> tuple[np.ndarray, np.ndarray]:
        speeds = self.mean_speeds(temp_i_K)
        sigma = self.scattering_cross_sections(speeds, self.debye_length_m(ne, temp_e))
        mfp = 1.0 / np.maximum(np.sum(n[np.newaxis, :] * sigma, axis=1), 1.0e-300)
        free = math.pi / 8.0 * mfp * speeds
        diff = np.empty_like(free)
        diff[self.mask_neutral] = free[self.mask_neutral]
        if np.any(self.mask_positive):
            gamma = temp_e * self.e / self.k / max(temp_i_K, 1.0e-300)
            alpha = float(np.sum(n[self.mask_negative])) / max(ne, 1.0e-300)
            diff_free_pos = float(np.mean(free[self.mask_positive]))
            diff[self.mask_positive] = diff_free_pos * (1.0 + gamma * (1.0 + 2.0 * alpha)) / (1.0 + alpha * gamma)
        if np.any(self.mask_negative):
            diff[self.mask_negative] = 0.0
        return diff, speeds

    def diffusion_source_rates(self, n: np.ndarray, diff: np.ndarray, speeds: np.ndarray) -> np.ndarray:
        sticking = self.chemistry.sticking
        wall_fluxes = -diff * n * sticking
        mask = sticking != 0.0
        wall_fluxes[mask] /= (sticking * self.params.diffusion_length_m + (4.0 * diff / speeds))[mask]
        surface_loss = wall_fluxes * self.params.area_m2 / self.params.volume_m3
        surface_source = np.sum(-surface_loss[np.newaxis, :] * self.chemistry.return_matrix, axis=1)
        return surface_loss + surface_source

    def flow_source_rates(self, n: np.ndarray, pressure_Pa: float) -> np.ndarray:
        t_flow = 1.0e-3
        source = np.zeros(len(self.chemistry.species_ids))
        source[self.mask_neutral] -= n[self.mask_neutral] * (1.0 / t_flow) * (pressure_Pa - self.params.pressure_Pa) / self.params.pressure_Pa
        source += self.feeds * self.sccm / self.params.volume_m3
        neutral_density = max(float(np.sum(n[self.mask_neutral])), 1.0e-300)
        source[self.mask_neutral] -= (
            float(np.sum(self.feeds * self.sccm / self.params.volume_m3))
            * n[self.mask_neutral]
            / neutral_density
        )
        return source

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        n = self.density(y)
        rho = float(y[-1])
        ne = self.electron_density(n)
        temp_e = self.electron_temperature_eV(n, rho)
        pressure = self.k * self.params.temp_n_K * float(np.sum(n))
        temp_i = self.ion_temperature_K(pressure)
        rates = self.reaction_rates(n, temp_e)
        source_vol = np.sum(self.chemistry.species_net * rates[:, np.newaxis], axis=0)
        diff, speeds = self.diffusivities(n, temp_i, temp_e, ne)
        source_diff = self.diffusion_source_rates(n, diff, speeds)
        min_n = np.zeros(len(n))
        below = n < 1.0
        min_n[below] = (1.0 - n[below]) / 1.0e-10
        dn_dt = source_vol + self.flow_source_rates(n, pressure) + source_diff + min_n

        losses = np.array(self.chemistry.electron_energy_losses, copy=True)
        elastic_mask = self.mask_electron_reaction & self.mask_elastic
        losses[elastic_mask] = 3.0 * self.m_e / self.chemistry.masses_kg[0] * (temp_e - self.params.temp_n_K * self.k / self.e)
        drho_ext = self.power_W(t) / self.params.volume_m3 / self.e
        drho_el_inel = float(np.sum(losses[self.mask_electron_reaction] * rates[self.mask_electron_reaction]))
        drho_gain_loss = 1.5 * temp_e * float(np.sum(self.chemistry.electron_net * rates))
        total_electron_outflux = float(np.sum(self.chemistry.charges * source_diff))
        drho_el_walls = -2.0 * temp_e * total_electron_outflux
        sheath_voltage = temp_e * self.sheath_voltage_per_ev
        drho_ions_walls = (
            -0.5 * temp_e * float(np.sum(source_diff[self.mask_positive]))
            - sheath_voltage * float(np.sum(self.chemistry.charges[self.mask_positive] * source_diff[self.mask_positive]))
        )
        min_rho = (1.0 - rho) / 1.0e-10 if rho < 1.0 else 0.0
        drho_dt = drho_ext - drho_el_inel - drho_gain_loss - drho_el_walls - drho_ions_walls + min_rho
        return np.r_[dn_dt, drho_dt]

    def solution_values(self, t: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        n = np.asarray(y[:-1, :], dtype=float)
        rho = np.asarray(y[-1, :], dtype=float)
        ne = np.sum(n * self.chemistry.charges[:, np.newaxis], axis=0)
        temp_e = np.maximum(rho / np.maximum(ne, 1.0e-300) * 2.0 / 3.0, self.params.temp_n_K * self.k / self.e)
        out = {sp_id: n[i] for i, sp_id in enumerate(self.chemistry.species_ids)}
        out.update({
            'e': ne,
            'T_e': temp_e,
            'T_n': np.full_like(t, self.params.temp_n_K, dtype=float),
            'p': self.k * self.params.temp_n_K * np.sum(n, axis=0),
            'P': np.asarray([self.power_W(float(item)) for item in t], dtype=float),
        })
        return out


def _same_footing_parameters(loaded: Any, geometry: dict[str, float]) -> SameFootingParameters:
    raw = _pygmol_parameters(loaded, geometry)
    t_power, power = _sanitize_power_series(
        np.asarray(raw['t_power'], dtype=float),
        np.asarray(raw['power'], dtype=float),
        float(raw['t_end']),
    )
    return SameFootingParameters(
        radius_m=float(raw['radius']),
        length_m=float(raw['length']),
        pressure_Pa=float(raw['pressure']),
        temp_e_eV=float(raw['temp_e']),
        temp_n_K=float(raw['temp_n']),
        feeds_sccm={str(k): float(v) for k, v in raw['feeds'].items()},
        t_power_s=t_power,
        power_W=power,
        t_end_s=float(raw['t_end']),
    )


def _run_local_same_footing(model_spec: dict[str, Any], params: SameFootingParameters, sample_times: np.ndarray) -> tuple[LocalPyGMolSameFootingODE, Any, dict[str, np.ndarray]]:
    ode = LocalPyGMolSameFootingODE.from_model(model_spec, params)
    y0 = ode.initial_state(INITIAL_DENSITIES)
    solution = solve_ivp(
        ode.rhs,
        (0.0, params.t_end_s),
        y0,
        method='BDF',
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f'Local PyGMol same-footing solve failed: {solution.message}')
    y_sampled = solution.sol(sample_times)
    values = ode.solution_values(sample_times, y_sampled)
    return ode, solution, values


def _comparison_rows(sample_times: np.ndarray, pygmol_solution: Any, local_values: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    columns = ['Ar', 'Ar+', 'e', 'T_e', 'T_n', 'p', 'P']
    rows: list[dict[str, Any]] = []
    for i, time_s in enumerate(sample_times):
        row: dict[str, Any] = {'time_s': float(time_s)}
        for col in columns:
            reference = float(pygmol_solution[col].iloc[i])
            candidate = float(local_values[col][i])
            key = col.replace('+', '_plus')
            row[f'pygmol_{key}'] = reference
            row[f'local_{key}'] = candidate
            row[f'rel_error_{key}'] = _safe_rel_error(candidate, reference)
        rows.append(row)
    return rows


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {'same_footing_axes_aligned': 1.0}
    for key in ['Ar', 'Ar_plus', 'e', 'T_e', 'T_n', 'p', 'P']:
        local = np.asarray([float(row[f'local_{key}']) for row in rows], dtype=float)
        reference = np.asarray([float(row[f'pygmol_{key}']) for row in rows], dtype=float)
        final_rel = _safe_rel_error(float(local[-1]), float(reference[-1]))
        waveform = _nrmse(local, reference)
        metrics[f'final_{key}_relative_error'] = final_rel
        metrics[f'{key}_waveform_nrmse'] = waveform
    metrics['max_final_relative_error'] = max(metrics[f'final_{key}_relative_error'] for key in ['Ar', 'Ar_plus', 'e', 'T_e', 'T_n', 'p', 'P'])
    metrics['max_waveform_nrmse'] = max(metrics[f'{key}_waveform_nrmse'] for key in ['Ar', 'Ar_plus', 'e', 'T_e', 'T_n', 'p', 'P'])
    return metrics


def build_precision_report(
    case_path: Path = DEFAULT_CASE,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    loaded = load_case_from_yaml(case_path.resolve())
    geometry = case_geometry(loaded)
    model_spec = load_benchmark_model(DEFAULT_BENCHMARK_MODEL)
    params = _same_footing_parameters(loaded, geometry)
    pygmol_model, package_metadata = run_pygmol(loaded, geometry, model_spec)
    pygmol_solution = pygmol_model.get_solution().reset_index(drop=True)
    sample_times = np.asarray(pygmol_solution['t'], dtype=float)
    ode, local_solution, local_values = _run_local_same_footing(model_spec, params, sample_times)
    rows = _comparison_rows(sample_times, pygmol_solution, local_values)
    metrics = _metric_summary(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    time_series_path = output_dir / 'pygmol_precision_timeseries.csv'
    fieldnames = list(rows[0]) if rows else ['time_s']
    _write_csv(time_series_path, rows, fieldnames)
    metrics_path = output_dir / 'pygmol_precision_metrics.csv'
    _write_csv(metrics_path, [{'metric_id': key, 'value': value} for key, value in sorted(metrics.items())], ['metric_id', 'value'])
    fixture_path = output_dir / 'pygmol_precision_fixture.yaml'
    fixture = {
        'scope': 'tools_only_same_footing_precision_fixture',
        'reference': 'PyGMol',
        'local_side': 'tools.external_benchmarks.pygmol_precision.LocalPyGMolSameFootingODE',
        'production_core_claim': 'none',
        'case_source': str(case_path.resolve()),
        'model_file': str(DEFAULT_BENCHMARK_MODEL),
        'model_id': model_spec['model']['id'],
        'aligned_axes': {
            'geometry': {
                'radius_m': params.radius_m,
                'length_m': params.length_m,
                'volume_m3': params.volume_m3,
                'area_m2': params.area_m2,
                'diffusion_length_m': params.diffusion_length_m,
            },
            'pressure_Pa': params.pressure_Pa,
            'neutral_temperature_K': params.temp_n_K,
            'initial_electron_temperature_eV': params.temp_e_eV,
            'feeds_sccm': params.feeds_sccm,
            't_power_s': [float(v) for v in params.t_power_s],
            'power_W': [float(v) for v in params.power_W],
            'initial_densities': INITIAL_DENSITIES,
            'surface_return_matrix': model_spec['surface_return_matrix'],
            'reaction_ids': ode.chemistry.reaction_ids,
        },
    }
    _write_yaml(fixture_path, fixture)
    report_path = output_dir / 'pygmol_precision_report.yaml'
    report = {
        'tool': 'pygmol_precision',
        'case': str(case_path.resolve()),
        'output_dir': str(output_dir),
        'pygmol': package_metadata,
        'local_solver': {
            'success': bool(local_solution.success),
            'nfev': int(local_solution.nfev),
            'njev': int(local_solution.njev or 0),
            'nlu': int(local_solution.nlu or 0),
            'sample_count': int(len(sample_times)),
        },
        'artifacts': {
            'time_series_csv': str(time_series_path),
            'metrics_csv': str(metrics_path),
            'fixture_yaml': str(fixture_path),
            'report_yaml': str(report_path),
        },
        'metrics': metrics,
        'interpretation': (
            'This is a tools-only same-footing precision benchmark. It checks that a local '
            'implementation of the PyGMol compact equations matches PyGMol when geometry, '
            'chemistry, power, wall return, initial state, and sampled times are aligned. '
            'It does not claim that the production plasma_global low-pressure model matches PyGMol.'
        ),
    }
    _write_yaml(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run the PyGMol same-footing precision benchmark.')
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        report = build_precision_report(args.case, output_dir=args.output_dir)
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['metrics']['max_final_relative_error'] <= 1.0e-3 and report['metrics']['max_waveform_nrmse'] <= 1.0e-3 else 1


if __name__ == '__main__':
    raise SystemExit(main())
