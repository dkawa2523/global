from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from plasma_global.chemistry.models import E_CHARGE, K_B
from plasma_global.eedf.base import EEDFRequest, EEDFResult
from plasma_global.eedf.swarm_backend import SWARM_MODEL_REGISTRY, SwarmEEDFBackend
from plasma_global.eedf.swarm_base import SwarmModel

M_E = 9.1093837015e-31
TD_TO_VM2 = 1.0e-21


@dataclass
class _MixtureProfile:
    target_fractions: dict[str, float]
    target_densities_m3: dict[str, float]
    total_target_density_m3: float
    momentum_sigma_mix: np.ndarray
    inelastic_sigma_mix: np.ndarray
    energy_loss_sigma_mix_eV: np.ndarray


@dataclass
class _MixtureSwarmTable:
    key: tuple[Any, ...]
    field_grid_Td: np.ndarray
    mean_energy_by_field_eV: np.ndarray
    k_by_field: dict[str, np.ndarray]
    mobility_by_field: np.ndarray
    diffusion_by_field: np.ndarray
    drift_velocity_by_field: np.ndarray
    mean_energy_grid_eV: np.ndarray
    field_by_mean_Td: np.ndarray
    k_by_mean: dict[str, np.ndarray]
    mobility_by_mean: np.ndarray
    diffusion_by_mean: np.ndarray
    drift_velocity_by_mean: np.ndarray


class Boltzmann2TermSwarmModel(SwarmModel):
    """Cross-section-driven two-term-like swarm closure.

    The implementation is intentionally modular:
    - actual tabulated cross sections are loaded from chemistry manifest files,
    - the swarm solver is separated from the EEDF backend interface,
    - mixture-specific tables are cached and can be replaced by alternative
      swarm models without touching the plasma chemistry code.

    The present solver uses a steady 1D energy-space recurrence with actual
    momentum-transfer and inelastic cross sections, and it closes the field
    dependence through an electron power-balance root solve. It is an internal
    approximate closure for compact studies and backend-contract testing, not
    a replacement for mature external swarm solvers.
    """

    def prepare(self, mechanism: Any, chamber: Any, run_config: Any, swarm_config: Any | None = None) -> None:
        super().prepare(mechanism, chamber, run_config, swarm_config)
        cfg = getattr(swarm_config, 'boltzmann_2term', None)
        energy_cfg = getattr(cfg, 'energy_grid', None)
        field_cfg = getattr(cfg, 'reduced_field_grid_Td', None)
        self.energy_min_eV = float(getattr(energy_cfg, 'min_eV', 1.0e-3) if energy_cfg else 1.0e-3)
        self.energy_max_eV = float(getattr(energy_cfg, 'max_eV', 160.0) if energy_cfg else 160.0)
        self.n_energy = int(getattr(energy_cfg, 'n', 360) if energy_cfg else 360)
        self.field_min_Td = float(getattr(field_cfg, 'min', 0.2) if field_cfg else 0.2)
        self.field_max_Td = float(getattr(field_cfg, 'max', 2500.0) if field_cfg else 2500.0)
        self.n_field = int(getattr(field_cfg, 'n', 48) if field_cfg else 48)
        self.max_shape_iter = int(getattr(cfg, 'max_shape_iterations', 48) if cfg else 48)
        self.max_field_iter = int(getattr(cfg, 'max_field_iterations', 42) if cfg else 42)
        self.cache_max_entries = int(getattr(getattr(swarm_config, 'cache', None), 'max_entries', 12) if swarm_config else 12)
        self.mixture_fraction_decimals = int(getattr(getattr(swarm_config, 'cache', None), 'fraction_decimals', 3) if swarm_config else 3)
        self.evaluation_mode = str(getattr(swarm_config, 'closure', 'auto') or 'auto').lower() if swarm_config else 'auto'
        key_species = getattr(swarm_config, 'mixture_key_species', None) if swarm_config else None
        if key_species:
            self.mixture_key_species = [str(s) for s in key_species]
        else:
            self.mixture_key_species = sorted({cs.target_species for cs in self.mechanism.cross_sections.values() if cs.target_species})
        self.energy_grid_eV = np.geomspace(self.energy_min_eV, self.energy_max_eV, self.n_energy)
        self.d_energy_eV = np.gradient(self.energy_grid_eV)
        self.electron_speed = np.sqrt(2.0 * self.energy_grid_eV * E_CHARGE / M_E)
        self.field_grid_Td = np.geomspace(self.field_min_Td, self.field_max_Td, self.n_field)
        self._sigma_on_grid = {cs_id: cs.sigma_interp(self.energy_grid_eV) for cs_id, cs in self.mechanism.cross_sections.items()}
        self._loss_eV_by_id = {cs_id: float(cs.energy_loss_eV if cs.energy_loss_eV is not None else cs.threshold_eV) for cs_id, cs in self.mechanism.cross_sections.items()}
        self._momentum_by_target: dict[str, list[str]] = {}
        self._inelastic_by_target: dict[str, list[str]] = {}
        for cs_id, cs in self.mechanism.cross_sections.items():
            target = cs.target_species or '__unassigned__'
            kind = str(cs.kind).lower()
            if kind in {'elastic', 'elastic_momentum', 'momentum_transfer', 'mt'}:
                self._momentum_by_target.setdefault(target, []).append(cs_id)
            else:
                self._inelastic_by_target.setdefault(target, []).append(cs_id)
        self._target_species = sorted(set(self._momentum_by_target) | set(self._inelastic_by_target))
        self._cache: OrderedDict[tuple[Any, ...], _MixtureSwarmTable] = OrderedDict()

    def _synthetic_momentum_sigma(self, target_species: str) -> np.ndarray:
        sp = self.mechanism.species_by_id.get(target_species)
        mass = max(getattr(sp, 'mass_amu', 28.0), 1.0) if sp is not None else 28.0
        sigma0 = 2.0e-20 * math.sqrt(mass / 28.0)
        eps = self.energy_grid_eV
        return sigma0 * (1.0 + eps / 5.0) ** -0.35

    def _target_densities(self, composition: dict[str, float]) -> dict[str, float]:
        densities: dict[str, float] = {}
        for target in self._target_species:
            if target == '__unassigned__':
                continue
            densities[target] = max(float(composition.get(target, 0.0)), 0.0)
        if not densities:
            for sp_id, val in composition.items():
                sp = self.mechanism.species_by_id.get(sp_id)
                if sp is not None and sp.phase == 'gas' and sp.charge == 0:
                    densities[sp_id] = max(float(val), 0.0)
        return densities

    def _build_mixture_profile(self, request: EEDFRequest) -> _MixtureProfile:
        target_dens = self._target_densities(request.composition)
        total = sum(target_dens.values())
        if total <= 0.0:
            total = max(float(request.pressure_Pa) / (K_B * max(float(request.gas_temperature_K), 1.0)), 1.0e18)
            if not target_dens:
                target_dens = {target: total / max(len(self._target_species), 1) for target in self._target_species if target != '__unassigned__'}
            else:
                scale = total / max(sum(target_dens.values()), 1.0)
                target_dens = {k: v * scale for k, v in target_dens.items()}
        fractions = {k: v / total for k, v in target_dens.items() if v > 0.0}
        sigma_mt = np.zeros_like(self.energy_grid_eV)
        sigma_inel = np.zeros_like(self.energy_grid_eV)
        sigma_loss = np.zeros_like(self.energy_grid_eV)
        for target, frac in fractions.items():
            mt_ids = self._momentum_by_target.get(target, [])
            if mt_ids:
                for cs_id in mt_ids:
                    sigma_mt += frac * self._sigma_on_grid[cs_id]
            else:
                sigma_mt += frac * self._synthetic_momentum_sigma(target)
            for cs_id in self._inelastic_by_target.get(target, []):
                sigma = self._sigma_on_grid[cs_id]
                sigma_inel += frac * sigma
                sigma_loss += frac * sigma * self._loss_eV_by_id[cs_id]
        if np.max(sigma_mt) <= 0.0:
            sigma_mt += self._synthetic_momentum_sigma('__fallback__')
        return _MixtureProfile(
            target_fractions=fractions,
            target_densities_m3=target_dens,
            total_target_density_m3=total,
            momentum_sigma_mix=sigma_mt,
            inelastic_sigma_mix=sigma_inel,
            energy_loss_sigma_mix_eV=sigma_loss,
        )

    def _mixture_key(self, request: EEDFRequest, profile: _MixtureProfile) -> tuple[Any, ...]:
        fracs = profile.target_fractions
        key_species = self.mixture_key_species or sorted(fracs)
        frac_key = tuple((sp, round(float(fracs.get(sp, 0.0)), self.mixture_fraction_decimals)) for sp in key_species)
        mode_key = self.evaluation_mode
        return (mode_key, frac_key)

    def _shape_distribution(self, h: float, profile: _MixtureProfile) -> np.ndarray:
        eps = self.energy_grid_eV
        de = self.d_energy_eV
        nu_m_norm = profile.momentum_sigma_mix / max(float(np.mean(profile.momentum_sigma_mix)), 1.0e-30)
        loss_norm = profile.energy_loss_sigma_mix_eV / max(float(np.mean(profile.energy_loss_sigma_mix_eV[profile.energy_loss_sigma_mix_eV > 0.0])) if np.any(profile.energy_loss_sigma_mix_eV > 0.0) else 1.0, 1.0e-12)
        D = (h + 0.02) * np.sqrt(eps + 0.03) / (1.0 + 0.35 * nu_m_norm)
        adv = 0.8 + 0.6 * nu_m_norm + 1.4 * loss_norm / max(h, 1.0e-6)
        W = np.clip(adv * de / np.maximum(D, 1.0e-14), 1.0e-9, 90.0)
        logf = np.zeros_like(eps)
        logf[1:] = -np.cumsum(W[:-1])
        # Mild phase-space weighting keeps the low-energy density finite.
        f = np.exp(logf - np.max(logf)) / np.sqrt(eps + 0.02)
        f /= max(np.trapezoid(f, eps), 1.0e-40)
        return f

    def _rate_coefficient(self, sigma: np.ndarray, f: np.ndarray) -> float:
        return float(np.trapezoid(sigma * self.electron_speed * f, self.energy_grid_eV))

    def _distribution_mean_energy(self, f: np.ndarray) -> float:
        return float(np.trapezoid(self.energy_grid_eV * f, self.energy_grid_eV))

    def _transport_from_distribution(self, f: np.ndarray, profile: _MixtureProfile, reduced_field_Td: float, total_density_m3: float) -> dict[str, float]:
        nu_m = total_density_m3 * self._rate_coefficient(profile.momentum_sigma_mix, f)
        mobility = E_CHARGE / max(M_E * nu_m, 1.0e-30)
        mean_energy_eV = self._distribution_mean_energy(f)
        diffusion = mobility * max(mean_energy_eV, 0.03)
        electric_field_V_m = max(reduced_field_Td, 0.0) * TD_TO_VM2 * total_density_m3
        drift_velocity = mobility * electric_field_V_m
        return {
            'mobility_m2_V_s': mobility,
            'diffusion_m2_s': diffusion,
            'drift_velocity_m_s': drift_velocity,
            'mean_energy_eV': mean_energy_eV,
        }

    def _collisional_power_loss_W_per_electron(self, f: np.ndarray, request: EEDFRequest, profile: _MixtureProfile) -> float:
        total = 0.0
        for cs_id, cs in self.mechanism.cross_sections.items():
            if str(cs.kind).lower() in {'elastic', 'elastic_momentum', 'momentum_transfer', 'mt'}:
                continue
            target = cs.target_species
            if not target:
                continue
            n_t = profile.target_densities_m3.get(target, 0.0)
            if n_t <= 0.0:
                continue
            k = self._rate_coefficient(self._sigma_on_grid[cs_id], f)
            total += n_t * k * self._loss_eV_by_id[cs_id] * E_CHARGE
        # Small elastic heating sink to avoid zero-loss pathologies.
        total += 0.03 * profile.total_target_density_m3 * self._rate_coefficient(profile.momentum_sigma_mix, f) * E_CHARGE
        return total

    def _heating_power_W_per_electron(self, f: np.ndarray, request: EEDFRequest, profile: _MixtureProfile, reduced_field_Td: float) -> float:
        total_density = max(profile.total_target_density_m3, float(request.pressure_Pa) / (K_B * max(float(request.gas_temperature_K), 1.0)), 1.0e18)
        transport = self._transport_from_distribution(f, profile, reduced_field_Td, total_density)
        E = max(reduced_field_Td, 0.0) * TD_TO_VM2 * total_density
        return E_CHARGE * transport['mobility_m2_V_s'] * E * E

    def _shape_mismatch(self, h: float, request: EEDFRequest, profile: _MixtureProfile, reduced_field_Td: float) -> tuple[float, np.ndarray]:
        f = self._shape_distribution(h, profile)
        p_heat = self._heating_power_W_per_electron(f, request, profile, reduced_field_Td)
        p_loss = self._collisional_power_loss_W_per_electron(f, request, profile)
        mismatch = math.log(max(p_heat, 1.0e-60) / max(p_loss, 1.0e-60))
        return mismatch, f

    def _solve_distribution_for_field(self, request: EEDFRequest, profile: _MixtureProfile, reduced_field_Td: float) -> np.ndarray:
        reduced_field_Td = max(float(reduced_field_Td), self.field_grid_Td[0])
        lo = 0.02
        hi = 220.0
        m_lo, f_lo = self._shape_mismatch(lo, request, profile, reduced_field_Td)
        m_hi, f_hi = self._shape_mismatch(hi, request, profile, reduced_field_Td)
        if abs(m_lo) < 1.0e-4:
            return f_lo
        if abs(m_hi) < 1.0e-4:
            return f_hi
        if m_lo * m_hi > 0.0:
            return f_lo if abs(m_lo) < abs(m_hi) else f_hi
        best = f_hi
        for _ in range(self.max_shape_iter):
            mid = math.sqrt(lo * hi)
            m_mid, f_mid = self._shape_mismatch(mid, request, profile, reduced_field_Td)
            best = f_mid
            if abs(m_mid) < 5.0e-4:
                break
            if m_lo * m_mid <= 0.0:
                hi = mid
                m_hi = m_mid
            else:
                lo = mid
                m_lo = m_mid
        return best

    def _solve_field_table(self, request: EEDFRequest, profile: _MixtureProfile) -> _MixtureSwarmTable:
        total_density = max(profile.total_target_density_m3, float(request.pressure_Pa) / (K_B * max(float(request.gas_temperature_K), 1.0)), 1.0e18)
        k_by_field = {cs_id: np.zeros_like(self.field_grid_Td) for cs_id in self.mechanism.cross_sections}
        mobility = np.zeros_like(self.field_grid_Td)
        diffusion = np.zeros_like(self.field_grid_Td)
        drift = np.zeros_like(self.field_grid_Td)
        mean_energy = np.zeros_like(self.field_grid_Td)
        for i, field in enumerate(self.field_grid_Td):
            f = self._solve_distribution_for_field(request, profile, float(field))
            mean_energy[i] = self._distribution_mean_energy(f)
            tr = self._transport_from_distribution(f, profile, float(field), total_density)
            mobility[i] = tr['mobility_m2_V_s']
            diffusion[i] = tr['diffusion_m2_s']
            drift[i] = tr['drift_velocity_m_s']
            for cs_id in self.mechanism.cross_sections:
                k_by_field[cs_id][i] = self._rate_coefficient(self._sigma_on_grid[cs_id], f)
        order = np.argsort(mean_energy)
        mean_sorted = np.asarray(mean_energy[order], dtype=float)
        field_sorted = np.asarray(self.field_grid_Td[order], dtype=float)
        # Enforce a strictly increasing mean-energy axis for interpolation.
        mean_sorted = np.maximum.accumulate(mean_sorted + np.linspace(0.0, 1.0e-6, mean_sorted.size))
        mobility_sorted = mobility[order]
        diffusion_sorted = diffusion[order]
        drift_sorted = drift[order]
        k_by_mean = {cs_id: table[order] for cs_id, table in k_by_field.items()}
        key = self._mixture_key(request, profile)
        return _MixtureSwarmTable(
            key=key,
            field_grid_Td=np.asarray(self.field_grid_Td, dtype=float),
            mean_energy_by_field_eV=np.asarray(mean_energy, dtype=float),
            k_by_field=k_by_field,
            mobility_by_field=np.asarray(mobility, dtype=float),
            diffusion_by_field=np.asarray(diffusion, dtype=float),
            drift_velocity_by_field=np.asarray(drift, dtype=float),
            mean_energy_grid_eV=mean_sorted,
            field_by_mean_Td=field_sorted,
            k_by_mean=k_by_mean,
            mobility_by_mean=mobility_sorted,
            diffusion_by_mean=diffusion_sorted,
            drift_velocity_by_mean=drift_sorted,
        )

    @staticmethod
    def _interp_slope(x: np.ndarray, y: np.ndarray, x0: float) -> float:
        if x.size < 2:
            return 0.0
        if x0 <= x[0]:
            i = 0
        elif x0 >= x[-1]:
            i = x.size - 2
        else:
            i = max(int(np.searchsorted(x, x0)) - 1, 0)
        dx = max(float(x[i + 1] - x[i]), 1.0e-30)
        return float((y[i + 1] - y[i]) / dx)

    def _lookup_table(self, request: EEDFRequest) -> _MixtureSwarmTable:
        profile = self._build_mixture_profile(request)
        key = self._mixture_key(request, profile)
        table = self._cache.get(key)
        if table is not None:
            self._cache.move_to_end(key)
            return table
        table = self._solve_field_table(request, profile)
        self._cache[key] = table
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)
        return table

    def _interp_by_field(self, field: float, table: _MixtureSwarmTable) -> EEDFResult:
        field_clip = float(np.clip(field, table.field_grid_Td[0], table.field_grid_Td[-1]))
        k_map = {cs_id: float(np.interp(field_clip, table.field_grid_Td, arr)) for cs_id, arr in table.k_by_field.items()}
        mean_e = float(np.interp(field_clip, table.field_grid_Td, table.mean_energy_by_field_eV))
        dk_map = {
            cs_id: self._interp_slope(table.mean_energy_grid_eV, table.k_by_mean[cs_id], mean_e)
            for cs_id in table.k_by_field
        }
        return EEDFResult(
            rate_coefficients=k_map,
            d_rate_d_mean_energy_eV=dk_map,
            transport={
                'mean_energy_eV': mean_e,
                'mobility_m2_V_s': float(np.interp(field_clip, table.field_grid_Td, table.mobility_by_field)),
                'diffusion_m2_s': float(np.interp(field_clip, table.field_grid_Td, table.diffusion_by_field)),
                'drift_velocity_m_s': float(np.interp(field_clip, table.field_grid_Td, table.drift_velocity_by_field)),
                'effective_field_Td': field_clip,
                'swarm_model': 'boltzmann_2term',
                'lookup_mode': 'field',
            },
        )

    def _interp_by_mean_energy(self, mean_energy_eV: float, table: _MixtureSwarmTable) -> EEDFResult:
        eps_clip = float(np.clip(mean_energy_eV, table.mean_energy_grid_eV[0], table.mean_energy_grid_eV[-1]))
        k_map = {cs_id: float(np.interp(eps_clip, table.mean_energy_grid_eV, arr)) for cs_id, arr in table.k_by_mean.items()}
        dk_map = {cs_id: self._interp_slope(table.mean_energy_grid_eV, arr, eps_clip) for cs_id, arr in table.k_by_mean.items()}
        return EEDFResult(
            rate_coefficients=k_map,
            d_rate_d_mean_energy_eV=dk_map,
            transport={
                'mean_energy_eV': eps_clip,
                'mobility_m2_V_s': float(np.interp(eps_clip, table.mean_energy_grid_eV, table.mobility_by_mean)),
                'diffusion_m2_s': float(np.interp(eps_clip, table.mean_energy_grid_eV, table.diffusion_by_mean)),
                'drift_velocity_m_s': float(np.interp(eps_clip, table.mean_energy_grid_eV, table.drift_velocity_by_mean)),
                'effective_field_Td': float(np.interp(eps_clip, table.mean_energy_grid_eV, table.field_by_mean_Td)),
                'swarm_model': 'boltzmann_2term',
                'lookup_mode': 'mean_energy',
            },
        )

    def evaluate(self, request: EEDFRequest) -> EEDFResult:
        table = self._lookup_table(request)
        mode = self.evaluation_mode
        if mode == 'local_field':
            use_field = True
        elif mode == 'mean_energy':
            use_field = False
        else:
            # In the global ODE system electron energy density is an evolved
            # state. The default closure therefore keeps reaction rates tied to
            # We/ne; explicit local_field mode is available for field-driven
            # studies.
            use_field = False
        if use_field:
            return self._interp_by_field(float(request.reduced_field_Td or 0.0), table)
        return self._interp_by_mean_energy(float(request.mean_energy_eV), table)


class Boltzmann2TermBackend(SwarmEEDFBackend):
    def __init__(self) -> None:
        super().__init__(forced_model_name='boltzmann_2term')


SWARM_MODEL_REGISTRY.register('boltzmann_2term', lambda **kwargs: Boltzmann2TermSwarmModel())
