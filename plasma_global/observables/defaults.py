from __future__ import annotations


BASE_SUMMARY_KEYS = (
    'electron_density_m3',
    'mean_electron_energy_eV',
    'self_bias_V',
    'plasma_potential_V',
    'total_absorbed_power_W',
)


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _allowed_final_observable_key(key: str) -> bool:
    if key in BASE_SUMMARY_KEYS:
        return True
    return (
        (key.startswith('ne_') and key.endswith('_m3'))
        or (key.startswith('mean_energy_') and key.endswith('_eV'))
        or (key.startswith('pabs_') and key.endswith('_W'))
        or (key.startswith('Tg_') and key.endswith('_K'))
        or (key.startswith('pressure_') and key.endswith('_Pa'))
        or (key.startswith('EoverN_') and key.endswith('_Td'))
        or (key.startswith('ion_flux_') and key.endswith('_m2_s'))
        or (key.startswith('film_') and key.endswith('_m'))
    )


def summarize_solution(solution, observables: list[dict] | None = None, chemistry_provenance: dict | None = None) -> dict:
    obs = observables or []
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
    for key in ('solver_event_count', 'steady_state_event_count'):
        value = solution.diagnostics.get(key)
        if value is not None:
            out[key] = int(value)
    event_counts = solution.diagnostics.get('event_counts') or {}
    if event_counts:
        out['event_counts'] = {str(k): int(v) for k, v in event_counts.items()}
    if chemistry_provenance:
        out['chemistry_provenance'] = chemistry_provenance
    if obs:
        final = obs[-1]
        for key in sorted(str(k) for k in final):
            if not _allowed_final_observable_key(key):
                continue
            if key in final and final[key] is not None:
                value = _safe_float(final[key])
                if value is not None:
                    out[f'final_{key}'] = value
    return out
