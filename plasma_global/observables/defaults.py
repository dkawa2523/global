from __future__ import annotations

import numpy as np


BASE_SUMMARY_KEYS = (
    'electron_density_m3',
    'mean_electron_energy_eV',
    'self_bias_V',
    'plasma_potential_V',
    'mean_ion_energy_wafer_eV',
    'ion_flux_wafer_m2_s',
    'radical_to_ion_flux_ratio_wafer',
    'F_to_C_radical_flux_ratio_wafer',
    'film_wafer_m',
    'etch_deposition_balance_wafer',
)

PORT_SUMMARY_SUFFIXES = (
    'absorbed_power_W',
    'delivered_power_W',
    'frequency_Hz',
    'source_voltage_V',
    'gap_voltage_V',
    'voltage_V',
    'voltage_rms_V',
    'rf_voltage_rms_V',
    'coil_voltage_rms_V',
    'current_A',
    'current_rms_A',
    'rf_current_rms_A',
    'coil_current_rms_A',
    'self_bias_V',
    'plasma_potential_V',
    'coupling_efficiency',
    'reduced_field_Td',
    'effective_field_Td',
    'plasma_resistance_ohm',
    'plasma_resistance_Ohm',
)


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _is_summary_port_key(key: str) -> bool:
    if not key.startswith('port_'):
        return False
    return any(key.endswith(f'_{suffix}') for suffix in PORT_SUMMARY_SUFFIXES)


def _summary_keys(records: list[dict]) -> list[str]:
    keys = list(BASE_SUMMARY_KEYS)
    dynamic_port_keys = sorted({k for rec in records for k in rec if _is_summary_port_key(str(k))})
    for key in dynamic_port_keys:
        if key not in keys:
            keys.append(key)
    return keys


def _stepwise_summary(records: list[dict]) -> dict[str, dict]:
    by_step: dict[str, list[dict]] = {}
    for rec in records:
        step_id = str(rec.get('step_id', 'unknown'))
        by_step.setdefault(step_id, []).append(rec)
    summary: dict[str, dict] = {}
    keys_of_interest = _summary_keys(records)
    for step_id, rows in by_step.items():
        block: dict[str, float | int] = {
            'n_points': len(rows),
            't_start_s': _safe_float(rows[0].get('time_s')),
            't_end_s': _safe_float(rows[-1].get('time_s')),
        }
        for key in keys_of_interest:
            values = [float(r[key]) for r in rows if key in r and r[key] is not None]
            if values:
                block[f'mean_{key}'] = float(np.mean(values))
                block[f'final_{key}'] = float(values[-1])
                block[f'min_{key}'] = float(np.min(values))
                block[f'max_{key}'] = float(np.max(values))
        warning_keys = sorted({k for r in rows for k in r.keys() if k.startswith('warning_')})
        for key in warning_keys:
            vals = [int(float(r.get(key, 0.0) or 0.0)) for r in rows]
            block[f'count_{key}'] = int(sum(vals))
        summary[step_id] = block
    return summary


def summarize_solution(solution) -> dict:
    obs = solution.diagnostics.get('observables', []) or []
    out = {
        'success': bool(solution.success),
        'status': int(solution.status),
        'message': str(solution.message),
        'n_times': int(solution.t.size),
        't_start_s': float(solution.t[0]) if solution.t.size else None,
        't_end_s': float(solution.t[-1]) if solution.t.size else None,
        'nfev': solution.diagnostics.get('nfev'),
        'njev': solution.diagnostics.get('njev'),
        'nlu': solution.diagnostics.get('nlu'),
    }
    if obs:
        final = obs[-1]
        for key in _summary_keys(obs):
            if key in final and final[key] is not None:
                out[f'final_{key}'] = float(final[key])
        out['step_summary'] = _stepwise_summary(obs)
        warning_keys = sorted({k for r in obs for k in r.keys() if k.startswith('warning_')})
        out['warning_counts'] = {key: int(sum(int(float(r.get(key, 0.0) or 0.0)) for r in obs)) for key in warning_keys}
    return out


def observables_dataframe(solution) -> list[dict]:
    return solution.diagnostics.get('observables', [])
