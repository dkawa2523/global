"""Solver scale and absolute-tolerance helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def state_scale(system: Any, y: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(np.asarray(y, dtype=float)), 1.0)
    if 'gas_densities' in system.state_layout.slices:
        gas_slice = system.state_layout.slice('gas_densities')
        scale[gas_slice] = np.maximum(np.abs(y[gas_slice]), system.floor_density)
    if 'electron_energy' in system.state_layout.slices:
        e_slice = system.state_layout.slice('electron_energy')
        scale[e_slice] = np.maximum(np.abs(y[e_slice]), system.floor_energy)
    if 'gas_temperature' in system.state_layout.slices:
        t_slice = system.state_layout.slice('gas_temperature')
        scale[t_slice] = np.maximum(np.abs(y[t_slice]), 50.0)
    for spec in system.mechanism.state_variables:
        for idx in system.state_layout.extra_state_index.get(spec.state_id, {}).values():
            scale[idx] = max(abs(float(y[idx])), float(spec.scale))
    return scale


def solver_atol_vector(system: Any) -> np.ndarray:
    base_atol = float(system.run_config.numerics.atol)
    atol = np.full(system.state_layout.size, base_atol, dtype=float)
    if 'gas_densities' in system.state_layout.slices:
        atol[system.state_layout.slice('gas_densities')] = max(base_atol, system.floor_density)
    if 'electron_energy' in system.state_layout.slices:
        atol[system.state_layout.slice('electron_energy')] = max(base_atol, system.floor_energy)
    if 'gas_temperature' in system.state_layout.slices:
        atol[system.state_layout.slice('gas_temperature')] = max(base_atol, 1.0e-6)
    if 'surface_coverages' in system.state_layout.slices:
        atol[system.state_layout.slice('surface_coverages')] = max(base_atol, 1.0e-12)
    for spec in system.mechanism.state_variables:
        floor = max(base_atol, abs(float(spec.scale)) * 1.0e-12)
        for idx in system.state_layout.extra_state_index.get(spec.state_id, {}).values():
            atol[idx] = floor
    return atol


def relative_rhs_norm_s_inv(system: Any, y: np.ndarray, dydt: np.ndarray) -> float:
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(dydt)):
        return float('inf')
    return float(np.max(np.abs(dydt) / state_scale(system, y)))


__all__ = ['relative_rhs_norm_s_inv', 'solver_atol_vector', 'state_scale']
