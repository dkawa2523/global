from __future__ import annotations

import argparse
import csv
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.eedf.base import EEDFRequest
from plasma_global.workflows.context import build_case
from plasma_global.workflows.context import load_case_from_yaml


DEFAULT_CASE = ROOT / 'examples' / 'configs' / 'case_argon_lxcat.yaml'
DEFAULT_BENCHMARK_MODEL = ROOT / 'tools' / 'external_benchmarks' / 'pygmol_argon_model.yaml'
RATE_MODE_SURROGATE = 'surrogate'
RATE_MODE_LOCAL_FIT = 'local-fit'
RATE_MODE_LOCAL_TABLE = 'local-table'
POWER_MODE_RECIPE = 'recipe'
POWER_MODE_LOCAL_ABSORBED = 'local-absorbed'
WALL_LOSS_MODE_PYGMOL = 'pygmol'
WALL_LOSS_MODE_LOCAL_COEFFICIENT = 'local-coefficient'
BOHM_FLUX_COEFF = 0.61


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def load_benchmark_model(path: Path = DEFAULT_BENCHMARK_MODEL) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f'Invalid PyGMol benchmark model YAML: {path}')
    model = data.get('model', {})
    if model.get('scope') != 'external_benchmark_only':
        raise ValueError(f'PyGMol benchmark model must be marked external_benchmark_only: {path}')
    if model.get('rate_alignment_with_local_case') != 'intentionally_not_aligned':
        raise ValueError(f'PyGMol benchmark model must declare intentionally_not_aligned rates: {path}')
    return data


def evaluate_pygmol_arrhenius(te_eV: np.ndarray, arrhenius: dict[str, float]) -> np.ndarray:
    te = np.asarray(te_eV, dtype=float)
    return (
        float(arrhenius['a'])
        * np.maximum(te, 1.0e-30) ** float(arrhenius['b'])
        * np.exp(-float(arrhenius['c_eV']) / np.maximum(te, 1.0e-30))
    )


def fit_pygmol_arrhenius(te_eV: np.ndarray, rate_m3_s: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Fit local electron-impact rates to the PyGMol Arrhenius form.

    PyGMol accepts electron-impact rate coefficients as
    k = A * Te**b * exp(-C/Te), with Te in eV.  This helper performs a log-space
    least-squares fit and reports log10 errors so the comparison can state how
    close the "same-rate" projection actually is.
    """

    te = np.asarray(te_eV, dtype=float)
    rates = np.asarray(rate_m3_s, dtype=float)
    mask = np.isfinite(te) & np.isfinite(rates) & (te > 0.0) & (rates > 0.0)
    if int(np.count_nonzero(mask)) < 3:
        raise ValueError('At least three positive local rate samples are required for Arrhenius fitting.')
    x = np.column_stack([np.ones(np.count_nonzero(mask)), np.log(te[mask]), -1.0 / te[mask]])
    beta, *_ = np.linalg.lstsq(x, np.log(rates[mask]), rcond=None)
    arrhenius = {
        'a': float(np.exp(beta[0])),
        'b': float(beta[1]),
        'c_eV': float(beta[2]),
    }
    predicted = evaluate_pygmol_arrhenius(te[mask], arrhenius)
    log10_error = np.log10(np.clip(predicted, 1.0e-300, None) / rates[mask])
    metrics = {
        'sample_count': int(np.count_nonzero(mask)),
        'te_min_eV': float(te[mask][0]),
        'te_max_eV': float(te[mask][-1]),
        'rms_log10_error': float(np.sqrt(np.mean(log10_error * log10_error))),
        'max_abs_log10_error': float(np.max(np.abs(log10_error))),
        'min_rate_m3_s': float(np.min(rates[mask])),
        'max_rate_m3_s': float(np.max(rates[mask])),
    }
    return arrhenius, metrics


def _local_rate_fit_te_grid() -> np.ndarray:
    # PyGMol's Te maps to local mean energy through <epsilon> = 3/2 Te.
    # Fit the powered-step range tightly; the afterglow tail is reported but is
    # not the acceptance target for this same-rate comparison.
    return np.geomspace(0.3, 8.0, 72)


def _local_rate_table_te_grid() -> np.ndarray:
    return np.geomspace(0.15, 8.0, 96)


def _local_rate_fit_conditions(loaded: Any) -> dict[str, float]:
    pressure = _weighted_zone_value(loaded.chamber.zones, 'pressure_Pa')
    gas_temperature = _weighted_zone_value(loaded.chamber.zones, 'gas_temperature_K')
    return {
        'pressure_Pa': pressure,
        'gas_temperature_K': gas_temperature,
        'argon_density_m3': pressure / (K_B * max(gas_temperature, 1.0)),
    }


def _electron_impact_channels(loaded: Any) -> list[dict[str, Any]]:
    channels: list[dict[str, Any]] = []
    mechanism = loaded.mechanism
    for rxn in mechanism.gas_reactions:
        model = mechanism.model(rxn.rate_model_key)
        if str(model.get('backend', '')).lower() != 'electron_impact_xsec':
            continue
        cs_id = str(model.get('cross_section_id'))
        cs = mechanism.cross_sections[cs_id]
        if cs.target_species != 'Ar':
            continue
        energy_model = mechanism.model(rxn.energy_model_key) if rxn.energy_model_key else {}
        kind = str(cs.kind).lower()
        is_ionization = kind == 'ionization' or rxn.products.get('Ar_plus', 0.0) > rxn.reactants.get('Ar_plus', 0.0)
        channels.append(
            {
                'id': rxn.reaction_id,
                'local_reaction_id': rxn.reaction_id,
                'local_cross_section_id': cs_id,
                'kind': 'ionization' if is_ionization else 'excitation',
                'electron_energy_loss_eV': float(
                    energy_model.get(
                        'energy_loss_eV',
                        cs.energy_loss_eV if cs.energy_loss_eV is not None else cs.threshold_eV,
                    )
                ),
                'elastic': False,
                'electron_stoich': {'lhs': 1, 'rhs': 2 if is_ionization else 1},
                'species_stoich_lhs': [1, 0],
                'species_stoich_rhs': [0, 1] if is_ionization else [1, 0],
            }
        )
    for cs_id, cs in sorted(mechanism.momentum_transfer_cross_sections.items()):
        if cs.target_species != 'Ar':
            continue
        channels.append(
            {
                'id': f'{cs_id}_elastic',
                'local_reaction_id': None,
                'local_cross_section_id': cs_id,
                'kind': 'elastic_momentum',
                'electron_energy_loss_eV': 0.0,
                'elastic': True,
                'electron_stoich': {'lhs': 1, 'rhs': 1},
                'species_stoich_lhs': [1, 0],
                'species_stoich_rhs': [1, 0],
            }
        )
    return channels


def build_local_rate_fit_model(loaded: Any) -> dict[str, Any]:
    """Build a PyGMol chemistry model from local LXCat/two-term rate samples.

    The result is still an external benchmark model: local rates are projected
    onto PyGMol's Arrhenius interface, and no core chemistry files are changed.
    """

    base = load_benchmark_model()
    built = build_case(loaded)
    conditions = _local_rate_fit_conditions(loaded)
    te_grid = _local_rate_fit_te_grid()
    channels = _electron_impact_channels(loaded)
    rate_samples: dict[str, list[float]] = {item['local_cross_section_id']: [] for item in channels}
    for te_eV in te_grid:
        request = EEDFRequest(
            time_s=0.0,
            zone_id=loaded.chamber.zones[0].zone_id,
            composition={'Ar': conditions['argon_density_m3']},
            electron_density_m3=1.0e16,
            mean_energy_eV=1.5 * float(te_eV),
            reduced_field_Td=None,
            gas_temperature_K=conditions['gas_temperature_K'],
            pressure_Pa=conditions['pressure_Pa'],
            metadata={'fit_source': 'pygmol_local_rate_fit'},
        )
        result = built.eedf_backend.evaluate(request)
        for cs_id in rate_samples:
            rate_samples[cs_id].append(float(result.rate_coefficients.get(cs_id, 0.0)))

    reactions: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    for channel in channels:
        cs_id = channel['local_cross_section_id']
        arrhenius, metrics = fit_pygmol_arrhenius(te_grid, np.asarray(rate_samples[cs_id], dtype=float))
        reaction = {
            'id': channel['id'],
            'equation': (
                'e + Ar -> 2e + Ar+'
                if channel['kind'] == 'ionization'
                else 'e + Ar -> e + Ar'
            ),
            'arrhenius': arrhenius,
            'electron_energy_loss_eV': channel['electron_energy_loss_eV'],
            'elastic': channel['elastic'],
            'electron_stoich': channel['electron_stoich'],
            'species_stoich_lhs': channel['species_stoich_lhs'],
            'species_stoich_rhs': channel['species_stoich_rhs'],
            'local_reaction_id': channel['local_reaction_id'],
            'local_cross_section_id': cs_id,
            'local_channel_kind': channel['kind'],
            'fit_metrics': metrics,
        }
        reactions.append(reaction)
        fit_summaries.append(
            {
                'reaction_id': channel['id'],
                'local_cross_section_id': cs_id,
                'local_channel_kind': channel['kind'],
                **metrics,
            }
        )

    return {
        'model': {
            'id': 'pygmol_argon_local_lxcat_rate_fit',
            'scope': 'external_benchmark_only',
            'rate_alignment_with_local_case': 'local_lxcat_arrhenius_fit',
            'rate_source': 'Local Ar LXCat/two-term EEDF rates fitted to PyGMol Arrhenius form',
            'purpose': 'Same-electron-collision-rate PyGMol comparison without changing plasma_global core physics.',
            'guardrails': [
                'Do not import this generated model from plasma_global core modules.',
                'Do not treat the fitted Arrhenius coefficients as production chemistry data.',
                'Use this only to isolate rate-physics differences from geometry, wall-loss, and PyGMol equation-form differences.',
            ],
            'fit_temperature_eV': {
                'te_min': float(te_grid[0]),
                'te_max': float(te_grid[-1]),
                'n': int(te_grid.size),
                'pygmol_temperature_relation': 'local_mean_energy_eV = 1.5 * pygmol_Te_eV',
            },
            'fit_conditions': conditions,
            'fit_summary': fit_summaries,
        },
        'species': base['species'],
        'surface_return_matrix': base['surface_return_matrix'],
        'reactions': reactions,
    }


def build_local_rate_table_model(loaded: Any) -> dict[str, Any]:
    """Build a PyGMol model whose electron rates are supplied by local tables."""

    model = build_local_rate_fit_model(loaded)
    built = build_case(loaded)
    conditions = _local_rate_fit_conditions(loaded)
    te_grid = _local_rate_table_te_grid()
    rate_by_reaction_id: dict[str, list[float]] = {reaction['id']: [] for reaction in model['reactions']}
    cs_by_reaction_id = {reaction['id']: reaction['local_cross_section_id'] for reaction in model['reactions']}
    for te_eV in te_grid:
        request = EEDFRequest(
            time_s=0.0,
            zone_id=loaded.chamber.zones[0].zone_id,
            composition={'Ar': conditions['argon_density_m3']},
            electron_density_m3=1.0e16,
            mean_energy_eV=1.5 * float(te_eV),
            reduced_field_Td=None,
            gas_temperature_K=conditions['gas_temperature_K'],
            pressure_Pa=conditions['pressure_Pa'],
            metadata={'fit_source': 'pygmol_local_rate_table'},
        )
        result = built.eedf_backend.evaluate(request)
        for reaction_id, cs_id in cs_by_reaction_id.items():
            rate_by_reaction_id[reaction_id].append(float(result.rate_coefficients.get(cs_id, 0.0)))

    model['model']['id'] = 'pygmol_argon_local_lxcat_rate_table'
    model['model']['rate_alignment_with_local_case'] = 'local_lxcat_rate_table'
    model['model']['rate_source'] = 'Local Ar LXCat/two-term EEDF rate table sampled for PyGMol runtime interpolation'
    model['model']['purpose'] = 'Same-electron-collision-rate PyGMol comparison using tabulated local rates without changing plasma_global core physics.'
    model['model']['rate_table'] = {
        'te_eV': [float(item) for item in te_grid],
        'pygmol_temperature_relation': 'local_mean_energy_eV = 1.5 * pygmol_Te_eV',
        'hold': 'edge',
        'interpolation': 'log_rate_vs_log_Te',
        'rate_by_reaction_id': rate_by_reaction_id,
    }
    model['model']['guardrails'] = [
        'Do not import this generated model from plasma_global core modules.',
        'The table interpolation is an external PyGMol benchmark adapter, not production chemistry input.',
        'Use this to isolate electron-collision-rate differences; geometry, wall-loss, flow, and equation-form differences remain.',
    ]
    return model


def _step_power_w(step: Any) -> float:
    return sum(_as_float(command.get('value_W', 0.0)) for command in step.power_ports.values())


def _first_feed_sccm(recipe: Any, species_id: str) -> float:
    for step in recipe.steps:
        for flows in step.gas_inlets.values():
            if species_id in flows:
                return _as_float(flows[species_id])
    return 0.0


def _weighted_zone_value(zones: list[Any], attr: str) -> float:
    total_v = sum(float(z.volume_m3) for z in zones)
    if total_v <= 0.0:
        return 0.0
    return sum(float(z.volume_m3) * float(getattr(z, attr)) for z in zones) / total_v


def equivalent_cylinder(total_volume_m3: float, total_area_m2: float, preferred_radius_m: float) -> tuple[float, float]:
    def area_error(radius: float) -> float:
        return 2.0 * math.pi * radius * radius + 2.0 * total_volume_m3 / radius - total_area_m2

    roots: list[float] = []
    r_min = max(0.005, preferred_radius_m * 0.25)
    r_max = max(1.0, preferred_radius_m * 4.0)
    previous_r = r_min
    previous_f = area_error(previous_r)
    for i in range(1, 801):
        r = r_min + (r_max - r_min) * i / 800
        f = area_error(r)
        if f == 0.0:
            roots.append(r)
        elif f * previous_f < 0.0:
            lo, hi = previous_r, r
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if area_error(lo) * area_error(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
        previous_r, previous_f = r, f

    radius = min(roots, key=lambda candidate: abs(candidate - preferred_radius_m)) if roots else preferred_radius_m
    length = total_volume_m3 / (math.pi * radius * radius)
    return radius, length


def case_geometry(loaded: Any) -> dict[str, float]:
    total_volume = sum(float(zone.volume_m3) for zone in loaded.chamber.zones)
    total_area = sum(float(surface.area_m2) for surface in loaded.chamber.surfaces)
    wafer_areas = [float(surface.area_m2) for surface in loaded.chamber.surfaces if surface.kind == 'wafer']
    preferred_radius = math.sqrt((wafer_areas[0] if wafer_areas else total_area / 6.0) / math.pi)
    radius, length = equivalent_cylinder(total_volume, total_area, preferred_radius)
    return {
        'volume_m3': total_volume,
        'area_m2': total_area,
        'radius_m': radius,
        'length_m': length,
    }


def _case_power_series(recipe: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    times: list[float] = []
    powers: list[float] = []
    for step in recipe.steps:
        power = _step_power_w(step)
        times.extend([float(step.t_start_s), float(step.t_end_s)])
        powers.extend([power, power])
    return tuple(times), tuple(powers)


def _local_absorbed_power_series(local_rows: list[dict[str, str]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pairs = [
        (_as_float(row.get('time_s')), max(_as_float(row.get('total_absorbed_power_W')), 0.0))
        for row in local_rows
        if row.get('time_s') not in (None, '')
    ]
    pairs = sorted({time_s: power for time_s, power in pairs}.items())
    if not pairs:
        raise ValueError('local absorbed-power mode requires observables rows with time_s and total_absorbed_power_W')
    return tuple(float(time_s) for time_s, _ in pairs), tuple(float(power) for _, power in pairs)


def _normalize_power_mode(power_mode: str) -> str:
    normalized = str(power_mode or POWER_MODE_RECIPE).lower().replace('_', '-')
    if normalized not in {POWER_MODE_RECIPE, POWER_MODE_LOCAL_ABSORBED}:
        raise ValueError(f'Unsupported PyGMol power mode: {power_mode!r}')
    return normalized


def _normalize_wall_loss_mode(wall_loss_mode: str) -> str:
    normalized = str(wall_loss_mode or WALL_LOSS_MODE_PYGMOL).lower().replace('_', '-')
    if normalized not in {WALL_LOSS_MODE_PYGMOL, WALL_LOSS_MODE_LOCAL_COEFFICIENT}:
        raise ValueError(f'Unsupported PyGMol wall-loss mode: {wall_loss_mode!r}')
    return normalized


def _local_global_ion_loss_coefficient_s(loaded: Any, row: dict[str, str]) -> float:
    ion = loaded.mechanism.species_by_id.get('Ar_plus')
    ion_mass = float(getattr(ion, 'mass_kg', 39.948 * 1.66053906660e-27)) if ion is not None else 39.948 * 1.66053906660e-27
    weighted_loss = 0.0
    weighted_charge = 0.0
    for zone in loaded.chamber.zones:
        zone_id = zone.zone_id
        ne = max(_as_float(row.get(f'ne_{zone_id}_m3')), 0.0)
        if ne <= 0.0:
            continue
        family = str(row.get(f'ion_loss_family_{zone_id}', '')).lower()
        if family == 'bohm':
            area = max(_as_float(row.get(f'ion_loss_area_{zone_id}_m2')), 0.0)
            h_factor = max(_as_float(row.get(f'ion_loss_h_factor_{zone_id}')), 0.0)
            mean_e = max(_as_float(row.get(f'mean_energy_{zone_id}_eV')), 0.05)
            k_zone = BOHM_FLUX_COEFF * h_factor * area / max(float(zone.volume_m3), 1.0e-30)
            k_zone *= math.sqrt(mean_e * E_CHARGE / max(ion_mass, 1.0e-30))
        elif family == 'ambipolar_diffusion':
            k_zone = max(_as_float(row.get(f'ambipolar_loss_rate_{zone_id}_s')), 0.0)
        else:
            k_zone = 0.0
        weight = ne * float(zone.volume_m3)
        weighted_loss += k_zone * weight
        weighted_charge += weight
    return weighted_loss / max(weighted_charge, 1.0e-30)


def _local_wall_loss_coefficient_series(loaded: Any, local_rows: list[dict[str, str]]) -> dict[str, Any]:
    pairs = [
        (_as_float(row.get('time_s')), max(_local_global_ion_loss_coefficient_s(loaded, row), 0.0))
        for row in local_rows
        if row.get('time_s') not in (None, '')
    ]
    pairs = sorted({time_s: coeff for time_s, coeff in pairs}.items())
    if not pairs:
        raise ValueError('local wall-loss mode requires observables rows with ion-loss diagnostics')
    return {
        'time_s': [float(time_s) for time_s, _ in pairs],
        'coefficient_s_inv': [float(coeff) for _, coeff in pairs],
        'definition': 'global charge-weighted local ion wall-loss coefficient, applied to PyGMol positive-ion wall loss as -k_wall(t) n_i',
    }


def pygmol_argon_chemistry(model_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    model_spec = model_spec or load_benchmark_model()
    species = model_spec['species']
    reactions = model_spec['reactions']
    return {
        'species_ids': [item['id'] for item in species],
        'species_charges': [item['charge'] for item in species],
        'species_masses': [item['mass_amu'] for item in species],
        'species_lj_sigma_coefficients': [item['lj_sigma'] for item in species],
        'species_surface_sticking_coefficients': [item['surface_sticking'] for item in species],
        'species_surface_return_matrix': model_spec['surface_return_matrix'],
        'reactions_ids': [item['id'] for item in reactions],
        'reactions_strings': [item['equation'] for item in reactions],
        'reactions_arrh_a': [item['arrhenius']['a'] for item in reactions],
        'reactions_arrh_b': [item['arrhenius']['b'] for item in reactions],
        'reactions_arrh_c': [item['arrhenius']['c_eV'] for item in reactions],
        'reactions_el_energy_losses': [item['electron_energy_loss_eV'] for item in reactions],
        'reactions_elastic_flags': [item['elastic'] for item in reactions],
        'reactions_electron_stoich_lhs': [item['electron_stoich']['lhs'] for item in reactions],
        'reactions_electron_stoich_rhs': [item['electron_stoich']['rhs'] for item in reactions],
        'reactions_arbitrary_stoich_lhs': [0 for _ in reactions],
        'reactions_arbitrary_stoich_rhs': [0 for _ in reactions],
        'reactions_species_stoichiomatrix_lhs': [item['species_stoich_lhs'] for item in reactions],
        'reactions_species_stoichiomatrix_rhs': [item['species_stoich_rhs'] for item in reactions],
    }


def _pygmol_parameters(
    loaded: Any,
    geometry: dict[str, float],
    *,
    local_rows: list[dict[str, str]] | None = None,
    power_mode: str = POWER_MODE_RECIPE,
) -> dict[str, Any]:
    power_mode = _normalize_power_mode(power_mode)
    if power_mode == POWER_MODE_LOCAL_ABSORBED:
        if local_rows is None:
            raise ValueError('local absorbed-power mode requires local observables rows')
        t_power, power = _local_absorbed_power_series(local_rows)
    else:
        t_power, power = _case_power_series(loaded.recipe)
    return {
        'radius': geometry['radius_m'],
        'length': geometry['length_m'],
        'pressure': _weighted_zone_value(loaded.chamber.zones, 'pressure_Pa'),
        'power': power,
        't_power': t_power,
        'feeds': {'Ar': _first_feed_sccm(loaded.recipe, 'Ar')},
        'temp_e': 3.0,
        'temp_n': _weighted_zone_value(loaded.chamber.zones, 'gas_temperature_K'),
        't_end': float(loaded.recipe.steps[-1].t_end_s),
    }


def _pygmol_model_class(benchmark_model: dict[str, Any]):
    from pygmol.equations import ElectronEnergyEquations
    from pygmol.model import Model

    model_meta = benchmark_model.get('model', {})
    rate_table = model_meta.get('rate_table')
    wall_loss_table = model_meta.get('pygmol_adapter', {}).get('wall_loss_coefficient_s')
    if not rate_table and not wall_loss_table:
        return Model

    class LocalRateTableElectronEnergyEquations(ElectronEnergyEquations):
        def __init__(self, chemistry, plasma_params):
            if rate_table:
                self._local_rate_te = np.asarray(rate_table['te_eV'], dtype=float)
                self._local_rate_log_te = np.log(np.maximum(self._local_rate_te, 1.0e-30))
                self._local_rate_by_id = {
                    str(reaction_id): np.asarray(values, dtype=float)
                    for reaction_id, values in rate_table['rate_by_reaction_id'].items()
                }
                self._local_rate_log_by_id = {
                    reaction_id: np.log(np.clip(values, 1.0e-300, None))
                    for reaction_id, values in self._local_rate_by_id.items()
                }
            else:
                self._local_rate_te = np.asarray([], dtype=float)
                self._local_rate_log_te = np.asarray([], dtype=float)
                self._local_rate_log_by_id = {}
            if wall_loss_table:
                self._local_wall_time_s = np.asarray(wall_loss_table['time_s'], dtype=float)
                self._local_wall_coeff_s_inv = np.asarray(wall_loss_table['coefficient_s_inv'], dtype=float)
            else:
                self._local_wall_time_s = np.asarray([], dtype=float)
                self._local_wall_coeff_s_inv = np.asarray([], dtype=float)
            self._current_time_s = 0.0
            super().__init__(chemistry, plasma_params)

        @property
        def ode_system_rhs(self):
            base_func = super().ode_system_rhs

            def func(t, y):
                self._current_time_s = float(t)
                return base_func(t, y)

            return func

        def get_reaction_rate_coefficients(self, y, temp_e=None):
            coefs = super().get_reaction_rate_coefficients(y, temp_e=temp_e)
            if not self._local_rate_log_by_id:
                return coefs
            if temp_e is None:
                temp_e = self.get_electron_temperature(y)
            te = float(np.clip(temp_e, self._local_rate_te[0], self._local_rate_te[-1]))
            log_te = math.log(max(te, 1.0e-30))
            for idx, reaction_id in enumerate(self.chemistry.reactions_ids):
                if not self.mask_r_electron[idx]:
                    continue
                log_values = self._local_rate_log_by_id.get(str(reaction_id))
                if log_values is None:
                    continue
                coefs[idx] = float(math.exp(np.interp(log_te, self._local_rate_log_te, log_values)))
            return coefs

        def get_surface_loss_rates(self, y, wall_fluxes=None):
            rates = super().get_surface_loss_rates(y, wall_fluxes=wall_fluxes)
            if self._local_wall_coeff_s_inv.size == 0:
                return rates
            n = self.get_density_vector(y)
            k_wall = float(np.interp(
                self._current_time_s,
                self._local_wall_time_s,
                self._local_wall_coeff_s_inv,
            ))
            rates = np.array(rates, dtype=float, copy=True)
            rates[self.mask_sp_positive] = -max(k_wall, 0.0) * n[self.mask_sp_positive]
            return rates

    class LocalRateTableModel(Model):
        def _initialize_equations(self):
            self.equations = LocalRateTableElectronEnergyEquations(self.chemistry, self.plasma_params)

        def get_surface_loss_rates(self):
            if self.solution_primary is None:
                return super().get_surface_loss_rates()
            import pandas as pd

            rows = []
            for time_s, y in zip(self.t, self.solution_primary):
                self.equations._current_time_s = float(time_s)
                rows.append(self.equations.get_surface_loss_rates(y).copy())
            data = np.column_stack([self.t, np.asarray(rows, dtype=float)])
            return pd.DataFrame(data, columns=['t'] + [str(sp_id) for sp_id in self.chemistry.species_ids])

    return LocalRateTableModel


def run_pygmol(
    loaded: Any,
    geometry: dict[str, float],
    benchmark_model: dict[str, Any],
    *,
    local_rows: list[dict[str, str]] | None = None,
    power_mode: str = POWER_MODE_RECIPE,
) -> tuple[Any, dict[str, str]]:
    try:
        model_class = _pygmol_model_class(benchmark_model)
    except ImportError as exc:
        raise RuntimeError('PyGMol is required. Install it with: py -m pip install ".[compare]"') from exc

    try:
        pygmol_version = version('pygmol')
    except PackageNotFoundError:
        pygmol_version = 'unknown'

    model = model_class(
        pygmol_argon_chemistry(benchmark_model),
        _pygmol_parameters(loaded, geometry, local_rows=local_rows, power_mode=power_mode),
    )
    model.run(initial_densities={'Ar': 1.0, 'Ar+': 1.0e-9})
    return model, {'pygmol_version': pygmol_version}


def _local_step_rows(rows: list[dict[str, str]], step_id: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get('step_id') == step_id]


def _pygmol_step_rows(solution: Any, t_start: float, t_end: float) -> list[dict[str, float]]:
    eps = max(1.0e-15, abs(t_end - t_start) * 1.0e-9)
    mask = (solution['t'] >= t_start - eps) & (solution['t'] <= t_end + eps)
    return solution.loc[mask].to_dict('records')


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ratio(candidate: float, reference: float) -> float | None:
    return None if reference == 0.0 else candidate / reference


def _add_ratio_fields(row: dict[str, Any], pygmol_key: str, local_key: str, ratio_key: str) -> None:
    ratio = _ratio(_as_float(row.get(pygmol_key)), _as_float(row.get(local_key)))
    row[ratio_key] = '' if ratio is None else ratio


def _pygmol_wall_loss_coefficient_s(final_row: dict[str, Any], area_over_volume: float) -> float:
    density = _as_float(final_row.get('Ar+'))
    flux = abs(_as_float(final_row.get('Ar+_wall_flux_m2_s')))
    if density <= 0.0:
        return 0.0
    return flux * area_over_volume / density


def build_comparison_rows(loaded: Any, local_rows: list[dict[str, str]], pygmol_model: Any) -> list[dict[str, Any]]:
    solution = pygmol_model.get_solution()
    surface_rates = pygmol_model.get_surface_loss_rates()
    solution = solution.merge(surface_rates[['t', 'Ar+']], on='t', how='left', suffixes=('', '_surface_loss_m3_s'))
    if 'Ar+_surface_loss_m3_s' not in solution.columns:
        solution['Ar+_surface_loss_m3_s'] = solution['Ar+']
    area_over_volume = pygmol_model.equations.area / pygmol_model.equations.volume
    solution['Ar+_wall_flux_m2_s'] = solution['Ar+_surface_loss_m3_s'].abs() / area_over_volume

    rows: list[dict[str, Any]] = []
    for step in loaded.recipe.steps:
        step_id = step.step_id
        local_step = _local_step_rows(local_rows, step_id)
        pygmol_step = _pygmol_step_rows(solution, float(step.t_start_s), float(step.t_end_s))
        if not local_step or not pygmol_step:
            continue

        local_final = local_step[-1]
        pygmol_final = pygmol_step[-1]
        row: dict[str, Any] = {
            'step_id': step_id,
            't_start_s': float(step.t_start_s),
            't_end_s': float(step.t_end_s),
            'local_final_ne_m3': _as_float(local_final.get('electron_density_m3')),
            'pygmol_final_ne_m3': _as_float(pygmol_final.get('e')),
            'local_mean_ne_m3': _mean([_as_float(item.get('electron_density_m3')) for item in local_step]),
            'pygmol_mean_ne_m3': _mean([_as_float(item.get('e')) for item in pygmol_step]),
            'local_final_mean_energy_eV': _as_float(local_final.get('mean_electron_energy_eV')),
            'pygmol_final_Te_eV': _as_float(pygmol_final.get('T_e')),
            'pygmol_final_mean_energy_equiv_eV': 1.5 * _as_float(pygmol_final.get('T_e')),
            'local_mean_absorbed_power_W': _mean([_as_float(item.get('total_absorbed_power_W')) for item in local_step]),
            'pygmol_mean_absorbed_power_W': _mean([_as_float(item.get('P')) for item in pygmol_step]),
            'local_final_wafer_ion_flux_m2_s': _as_float(local_final.get('ion_flux_wafer_m2_s')),
            'pygmol_final_cylinder_wall_Arplus_flux_m2_s': _as_float(pygmol_final.get('Ar+_wall_flux_m2_s')),
            'local_final_global_ion_loss_rate_s': _local_global_ion_loss_coefficient_s(loaded, local_final),
            'pygmol_final_Arplus_wall_loss_rate_s': _pygmol_wall_loss_coefficient_s(pygmol_final, area_over_volume),
        }
        _add_ratio_fields(row, 'pygmol_final_ne_m3', 'local_final_ne_m3', 'ratio_final_ne_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_mean_ne_m3', 'local_mean_ne_m3', 'ratio_mean_ne_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_final_mean_energy_equiv_eV', 'local_final_mean_energy_eV', 'ratio_final_mean_energy_equiv_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_mean_absorbed_power_W', 'local_mean_absorbed_power_W', 'ratio_mean_absorbed_power_pygmol_over_local')
        _add_ratio_fields(row, 'pygmol_final_Arplus_wall_loss_rate_s', 'local_final_global_ion_loss_rate_s', 'ratio_final_wall_loss_rate_pygmol_over_local')
        rows.append(row)
    return rows


def ensure_local_outputs(case_path: Path, rerun_local: bool) -> tuple[Any, Path]:
    loaded = load_case_from_yaml(case_path)
    output_dir = Path(loaded.resolved_paths.output_dir)
    observables_path = output_dir / 'observables.csv'
    if rerun_local or not observables_path.exists():
        from plasma_global.workflows.runner import run_from_yaml

        run_from_yaml(case_path)
    return loaded, observables_path


def _normalize_rate_mode(rate_mode: str) -> str:
    normalized = str(rate_mode or RATE_MODE_SURROGATE).lower().replace('_', '-')
    if normalized not in {RATE_MODE_SURROGATE, RATE_MODE_LOCAL_FIT, RATE_MODE_LOCAL_TABLE}:
        raise ValueError(f'Unsupported PyGMol rate mode: {rate_mode!r}')
    return normalized


def _output_stem(
    rate_mode: str,
    power_mode: str = POWER_MODE_RECIPE,
    wall_loss_mode: str = WALL_LOSS_MODE_PYGMOL,
) -> str:
    if rate_mode == RATE_MODE_LOCAL_TABLE:
        stem = 'comparison_pygmol_local_rate_table'
    elif rate_mode == RATE_MODE_LOCAL_FIT:
        stem = 'comparison_pygmol_local_rates'
    elif rate_mode == RATE_MODE_SURROGATE:
        stem = 'comparison_pygmol'
    else:
        raise ValueError(f'Unsupported PyGMol rate mode: {rate_mode!r}')
    if power_mode == POWER_MODE_LOCAL_ABSORBED:
        stem += '_local_power'
    if wall_loss_mode == WALL_LOSS_MODE_LOCAL_COEFFICIENT:
        stem += '_local_wall'
    return stem


def _rate_fit_error_summary(model_spec: dict[str, Any]) -> dict[str, float | None]:
    model = model_spec.get('model', {})
    if model.get('rate_alignment_with_local_case') == 'local_lxcat_rate_table':
        return {
            'rate_table_sample_count': len(model.get('rate_table', {}).get('te_eV', []) or []) or None,
            'rate_fit_rms_log10_error_max': None,
            'rate_fit_max_abs_log10_error_max': None,
        }
    channels = model.get('fit_summary', []) or []
    summary = {
        'rate_table_sample_count': len(model.get('rate_table', {}).get('te_eV', []) or []) or None,
        'rate_fit_rms_log10_error_max': None,
        'rate_fit_max_abs_log10_error_max': None,
    }
    if not channels:
        return summary
    summary.update({
        'rate_fit_rms_log10_error_max': max(float(item['rms_log10_error']) for item in channels),
        'rate_fit_max_abs_log10_error_max': max(float(item['max_abs_log10_error']) for item in channels),
    })
    return summary


def build_report(
    case_path: Path = DEFAULT_CASE,
    *,
    rerun_local: bool = False,
    write_solution_csv: bool = True,
    rate_mode: str = RATE_MODE_SURROGATE,
    power_mode: str = POWER_MODE_RECIPE,
    wall_loss_mode: str = WALL_LOSS_MODE_PYGMOL,
) -> dict[str, Any]:
    rate_mode = _normalize_rate_mode(rate_mode)
    power_mode = _normalize_power_mode(power_mode)
    wall_loss_mode = _normalize_wall_loss_mode(wall_loss_mode)
    loaded, observables_path = ensure_local_outputs(case_path.resolve(), rerun_local)
    local_rows = _read_csv_rows(observables_path)
    output_dir = observables_path.parent
    geometry = case_geometry(loaded)
    if rate_mode == RATE_MODE_LOCAL_TABLE:
        benchmark_model = build_local_rate_table_model(loaded)
    elif rate_mode == RATE_MODE_LOCAL_FIT:
        benchmark_model = build_local_rate_fit_model(loaded)
    else:
        benchmark_model = load_benchmark_model()
    if wall_loss_mode == WALL_LOSS_MODE_LOCAL_COEFFICIENT:
        benchmark_model.setdefault('model', {}).setdefault('pygmol_adapter', {})['wall_loss_coefficient_s'] = _local_wall_loss_coefficient_series(loaded, local_rows)
    stem = _output_stem(rate_mode, power_mode, wall_loss_mode)
    generated_model_path = None
    if rate_mode in {RATE_MODE_LOCAL_FIT, RATE_MODE_LOCAL_TABLE}:
        generated_model_path = output_dir / f'{stem}_model.yaml'
        _write_yaml(generated_model_path, benchmark_model)
    pygmol_model, package_metadata = run_pygmol(
        loaded,
        geometry,
        benchmark_model,
        local_rows=local_rows,
        power_mode=power_mode,
    )
    comparison_rows = build_comparison_rows(loaded, local_rows, pygmol_model)

    summary_path = output_dir / f'{stem}_summary.csv'
    fieldnames = [
        'step_id',
        't_start_s',
        't_end_s',
        'local_final_ne_m3',
        'pygmol_final_ne_m3',
        'ratio_final_ne_pygmol_over_local',
        'local_mean_ne_m3',
        'pygmol_mean_ne_m3',
        'ratio_mean_ne_pygmol_over_local',
        'local_final_mean_energy_eV',
        'pygmol_final_Te_eV',
        'pygmol_final_mean_energy_equiv_eV',
        'ratio_final_mean_energy_equiv_pygmol_over_local',
        'local_mean_absorbed_power_W',
        'pygmol_mean_absorbed_power_W',
        'ratio_mean_absorbed_power_pygmol_over_local',
        'local_final_wafer_ion_flux_m2_s',
        'pygmol_final_cylinder_wall_Arplus_flux_m2_s',
        'local_final_global_ion_loss_rate_s',
        'pygmol_final_Arplus_wall_loss_rate_s',
        'ratio_final_wall_loss_rate_pygmol_over_local',
    ]
    _write_csv(summary_path, comparison_rows, fieldnames)
    solution_path = output_dir / f'{stem}_solution.csv'
    if write_solution_csv:
        pygmol_model.get_solution().to_csv(solution_path, index=False)
    metadata_path = output_dir / f'{stem}_metadata.yaml'
    if rate_mode == RATE_MODE_LOCAL_TABLE:
        rate_policy = {
            'status': 'same_electron_collision_rates_from_local_table',
            'local_case': 'native local Ar LXCat/two-term workflow remains unchanged',
            'pygmol_case': 'uses an external benchmark adapter that overrides PyGMol electron-impact coefficients from local rate tables',
            'reason': 'This is the closest same-rate comparison available without modifying PyGMol itself or the plasma_global core.',
        }
        model_note = (
            'PyGMol equations are subclassed inside the external benchmark tool so electron-impact rate coefficients are interpolated from local LXCat/two-term rate tables at runtime. '
            'Only electron-collision rates are aligned.'
        )
        comparison_name = 'argon_lxcat_vs_pygmol_local_lxcat_rate_table'
    elif rate_mode == RATE_MODE_LOCAL_FIT:
        rate_policy = {
            'status': 'same_electron_collision_rates_projected_to_pygmol_arrhenius',
            'local_case': 'native local Ar LXCat/two-term workflow remains unchanged',
            'pygmol_case': 'uses generated external-benchmark-only Arrhenius fits to local electron-impact rate coefficients',
            'reason': 'This isolates electron-collision-rate differences while keeping geometry, wall, and PyGMol equation-form differences visible.',
        }
        model_note = (
            'PyGMol receives a generated external model whose electron-impact rates are fitted from the local LXCat/two-term rates as functions of PyGMol Te. '
            'The fit is not exact; see fit_summary in the generated model YAML.'
        )
        comparison_name = 'argon_lxcat_vs_pygmol_local_lxcat_rate_fit'
    else:
        rate_policy = {
            'status': 'not_same_rate_physics_by_design',
            'local_case': 'native local Ar LXCat/two-term workflow remains unchanged',
            'pygmol_case': 'uses external-benchmark-only compact Arrhenius surrogate',
            'reason': 'Forcing the core case to use PyGMol surrogate rates would make a misleading production model.',
        }
        model_note = (
            'PyGMol solves a 0D heavy-species/electron-energy global model. '
            'The electron-impact rates are intentionally isolated in the external PyGMol benchmark model file and are not the same LXCat cross-section integration used by the local case.'
        )
        comparison_name = 'argon_lxcat_vs_pygmol_minimal_argon'
    metadata = {
        'comparison': comparison_name,
        'local_case': str(Path(loaded.resolved_paths.source_config)),
        'pygmol': {
            **package_metadata,
            'project_url': 'https://github.com/hanicinecm/pygmol',
            'pypi_url': 'https://pypi.org/project/pygmol/',
            'benchmark_model_file': str(generated_model_path or DEFAULT_BENCHMARK_MODEL),
            'benchmark_model_id': benchmark_model['model']['id'],
            'rate_alignment_with_local_case': benchmark_model['model']['rate_alignment_with_local_case'],
            'rate_source': benchmark_model['model']['rate_source'],
            'power_mode': power_mode,
            'wall_loss_mode': wall_loss_mode,
            'guardrails': benchmark_model['model']['guardrails'],
            'model_note': model_note,
        },
        'rate_policy': rate_policy,
        'equivalent_cylinder': geometry,
        'limitations': [
            'Local model is two-zone; PyGMol comparison is a single equivalent cylinder.',
            'Local electron energy is EEDF mean energy; PyGMol reports electron temperature.',
            'Local wafer ion flux and PyGMol total cylinder-wall Ar+ flux are not identical observables.',
            'Same-rate modes align electron-impact rate coefficients only; they do not align wall-loss or flow equations.',
            'local-absorbed power mode reuses local total_absorbed_power_W as the PyGMol power waveform.',
            'local-coefficient wall-loss mode replaces PyGMol positive-ion surface loss with the local global k_wall(t).',
        ],
    }
    _write_yaml(metadata_path, metadata)

    powered_rows = [row for row in comparison_rows if _as_float(row.get('local_mean_absorbed_power_W')) > 0.0]
    ratios = [_as_float(row.get('ratio_final_ne_pygmol_over_local'), default=float('nan')) for row in powered_rows]
    finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio) and ratio > 0.0]
    passed = bool(finite_ratios) and all(1.0e-3 <= ratio <= 1.0e3 for ratio in finite_ratios)
    return {
        'tool': 'pygmol_argon',
        'case': str(case_path.resolve()),
        'rate_mode': rate_mode,
        'power_mode': power_mode,
        'wall_loss_mode': wall_loss_mode,
        'output_dir': str(output_dir),
        'comparison_file': str(summary_path),
        'solution_file': str(solution_path) if write_solution_csv else None,
        'metadata_file': str(metadata_path),
        'generated_model_file': str(generated_model_path) if generated_model_path else None,
        'passed': passed,
        'summary': {
            'powered_step_count': len(powered_rows),
            'powered_ne_ratio_min_pygmol_over_local': min(finite_ratios) if finite_ratios else None,
            'powered_ne_ratio_max_pygmol_over_local': max(finite_ratios) if finite_ratios else None,
            'rate_alignment_with_local_case': benchmark_model['model']['rate_alignment_with_local_case'],
            'power_mode': power_mode,
            'wall_loss_mode': wall_loss_mode,
            **_rate_fit_error_summary(benchmark_model),
        },
        'rows': comparison_rows,
        'assessment': (
            'PyGMol is treated as an executable global-model benchmark. Passing means powered-step electron densities '
            'are within three orders of magnitude. In local-table mode, electron-impact rates are interpolated from local tables; '
            'in local-fit mode they are aligned through PyGMol Arrhenius fits. Power and wall-loss modes may also be aligned, but flow, geometry, and equation-form differences remain.'
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='External-tool PyGMol comparison for the local pure-Ar LXCat case.')
    parser.add_argument('--case', type=Path, default=DEFAULT_CASE)
    parser.add_argument('--rerun-local', action='store_true')
    parser.add_argument('--no-solution-csv', action='store_true')
    parser.add_argument('--rate-mode', choices=[RATE_MODE_SURROGATE, RATE_MODE_LOCAL_FIT, RATE_MODE_LOCAL_TABLE], default=RATE_MODE_SURROGATE)
    parser.add_argument('--power-mode', choices=[POWER_MODE_RECIPE, POWER_MODE_LOCAL_ABSORBED], default=POWER_MODE_RECIPE)
    parser.add_argument('--wall-loss-mode', choices=[WALL_LOSS_MODE_PYGMOL, WALL_LOSS_MODE_LOCAL_COEFFICIENT], default=WALL_LOSS_MODE_PYGMOL)
    parser.add_argument('--write', type=Path, default=None, help='Optional YAML report path; defaults to <output_dir>/comparison_pygmol_report.yaml')
    args = parser.parse_args(argv)

    try:
        report = build_report(
            args.case,
            rerun_local=args.rerun_local,
            write_solution_csv=not args.no_solution_csv,
            rate_mode=args.rate_mode,
            power_mode=args.power_mode,
            wall_loss_mode=args.wall_loss_mode,
        )
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    out = args.write or (
        Path(report['output_dir'])
        / f'{_output_stem(report["rate_mode"], report["power_mode"], report["wall_loss_mode"])}_report.yaml'
    )
    _write_yaml(out, report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
