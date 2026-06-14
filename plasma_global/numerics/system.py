from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from plasma_global.chemistry.models import MechanismBundle
from plasma_global.eedf.base import EEDFBackend
from plasma_global.electrical.base import ElectricalBackend
from plasma_global.electrical.coupling_adapter import ElectricalCouplingAdapter
from plasma_global.numerics.state_layout import StateLayout
from plasma_global.observables.adapter import ObservablesAdapter
from plasma_global.physics.gas_phase_core import GasPhaseCore
from plasma_global.physics.prescribed_electrons import build_prescribed_electron_profile
from plasma_global.physics.surface_core import SurfaceCore
from plasma_global.reactor.models import ChamberConfig, RecipeConfig
from plasma_global.reactor.surface_models import (
    bohm_h_factor,
    effective_ion_loss_frequency_s,
    ion_loss_enabled,
    ion_loss_family,
    ion_loss_uses_effective_frequency,
)


@dataclass
class GlobalPlasmaSystem:
    """Top-level transient multiphysics coordinator.

    This class intentionally keeps the public solver-facing interface stable
    (`initial_state`, `rhs`, `jacobian`, `compute_observables`, `state_labels`),
    while delegating most domain logic to dedicated components:

    - `GasPhaseCore`
    - `SurfaceCore`
    - `ElectricalCouplingAdapter`
    - `ObservablesAdapter`

    The goal is to preserve compatibility with the existing workflow while
    making the central coupling point smaller and easier to reason about.
    """

    mechanism: MechanismBundle
    chamber: ChamberConfig
    recipe: RecipeConfig
    run_config: Any
    eedf_backend: EEDFBackend
    electrical_backend: ElectricalBackend
    state_layout: StateLayout
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.zone_ids = list(self.state_layout.zone_ids)
        self.zone_index = {z: i for i, z in enumerate(self.zone_ids)}
        self.n_zones = len(self.zone_ids)
        self.gas_species_ids = list(self.state_layout.gas_species_ids)
        self.gas_species_index = {s: i for i, s in enumerate(self.gas_species_ids)}
        self.n_gas_species = len(self.gas_species_ids)
        self.surface_ids = list(self.state_layout.surface_index.keys())

        self.positivity_cfg = self.run_config.numerics.positivity or {}
        self.floor_density = float(self.positivity_cfg.get('floor_density_m3', 1.0))
        self.floor_energy = float(self.positivity_cfg.get('floor_energy_J_m3', 1.0e-12))
        self.clip_negative = bool(self.positivity_cfg.get('clip_negative', True))

        self.gas_species = [self.mechanism.species_by_id[s] for s in self.gas_species_ids]
        self.gas_charges = np.array([sp.charge for sp in self.gas_species], dtype=float)
        self.gas_masses = np.array([max(sp.mass_kg, 1.0e-30) for sp in self.gas_species], dtype=float)
        self.gas_species_tags = {sp.canonical_id: set(sp.state_tags) for sp in self.gas_species}
        self.surface_species_tags = {sp.canonical_id: set(sp.state_tags) for sp in self.mechanism.surface_species}
        self.positive_ion_local_indices = [i for i, sp in enumerate(self.gas_species) if sp.charge > 0]
        self.negative_ion_local_indices = [i for i, sp in enumerate(self.gas_species) if sp.charge < 0]
        self.radical_local_indices = [
            i for i, sp in enumerate(self.gas_species)
            if sp.charge == 0 and 'radical' in self.gas_species_tags.get(sp.canonical_id, set())
        ]
        self.halogen_elements = {'F', 'Cl', 'Br'}
        self.zone_wall_area = {
            z.zone_id: sum(s.area_m2 for s in self.chamber.surfaces_by_zone.get(z.zone_id, []))
            for z in self.chamber.zones
        }
        self.zone_ion_loss_family = self._zone_ion_loss_family()
        self.zone_ion_loss_area = {
            z.zone_id: sum(
                s.area_m2
                for s in self.chamber.surfaces_by_zone.get(z.zone_id, [])
                if ion_loss_enabled(s.models)
            )
            for z in self.chamber.zones
        }
        self.zone_char_length_m = {
            z.zone_id: max(z.volume_m3 / max(self.zone_wall_area.get(z.zone_id, 0.0), 1.0e-12), 1.0e-6)
            for z in self.chamber.zones
        }
        self.zone_ion_loss_characteristic_length_m = self._zone_bohm_characteristic_length_m()
        self.zone_ion_loss_h_factor = self._zone_bohm_h_factor()
        self.zone_effective_ion_loss_frequency_s = self._zone_effective_ion_loss_frequency_s()
        self.zone_ambipolar_loss_rate_s = dict(self.zone_effective_ion_loss_frequency_s)
        self.electron_density_closure = str(getattr(self.run_config.physics, 'electron_density_closure', 'quasi_neutral') or 'quasi_neutral').lower()
        self.prescribed_electron_profile = build_prescribed_electron_profile(self.run_config, self.zone_ids)

        self.gas_core = GasPhaseCore(self)
        self.surface_core = SurfaceCore(self)
        self.electrical_adapter = ElectricalCouplingAdapter(self)
        self.observables_adapter = ObservablesAdapter(self)

        # Backward-compatible aliases for downstream tooling.
        self.zone_residence_time_s = self.gas_core.zone_residence_time_s
        self.zone_wall_temperature = self.gas_core.zone_wall_temperature
        self.surface_free_site_species = self.surface_core.surface_free_site_species
        self.surface_site_occupancy = self.surface_core.surface_site_occupancy
        self.gas_reactions = self.gas_core.gas_reactions
        self.surface_reactions = self.surface_core.surface_reactions
        self._last_power = None
        self._last_zone_mean_energy_eV = {z: 3.0 for z in self.zone_ids}
        self._last_zone_electron_density_m3 = {z: self.floor_density for z in self.zone_ids}
        self.reset_numerical_diagnostics()

    def reset_numerical_diagnostics(self) -> None:
        self.diagnostics.update(
            {
                'projection_density_clip_count': 0,
                'projection_electron_energy_clip_count': 0,
                'projection_max_abs_delta': 0.0,
                'nonfinite_state_count': 0,
                'final_rhs_norm_inf': None,
                'final_relative_rhs_norm_s_inv': None,
                'solver_event_count': 0,
                'steady_state_event_count': 0,
            }
        )

    def _zone_ion_loss_family(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for zone in self.chamber.zones:
            families = {
                ion_loss_family(surface.models)
                for surface in self.chamber.surfaces_by_zone.get(zone.zone_id, [])
                if ion_loss_enabled(surface.models)
            }
            families.discard('disabled')
            if not families:
                out[zone.zone_id] = 'disabled'
            elif len(families) == 1:
                out[zone.zone_id] = next(iter(families))
            else:
                raise ValueError(
                    f'Zone {zone.zone_id} mixes ion-loss model families {sorted(families)}; '
                    'choose either Bohm-like or one effective-frequency wall-loss family to avoid double counting.'
                )
        return out

    def _zone_bohm_h_factor(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone in self.chamber.zones:
            surfaces = [
                surface for surface in self.chamber.surfaces_by_zone.get(zone.zone_id, [])
                if ion_loss_enabled(surface.models) and ion_loss_family(surface.models) == 'bohm'
            ]
            area = sum(surface.area_m2 for surface in surfaces)
            if area <= 0.0:
                out[zone.zone_id] = 0.0
                continue
            weighted = 0.0
            for surface in surfaces:
                char_length = float(surface.models.get('characteristic_length_m') or self.zone_ion_loss_characteristic_length_m[zone.zone_id])
                h = bohm_h_factor(
                    surface.models,
                    pressure_Pa=zone.pressure_Pa,
                    gas_temperature_K=zone.gas_temperature_K,
                    characteristic_length_m=char_length,
                )
                weighted += surface.area_m2 * h
            out[zone.zone_id] = weighted / area
        return out

    def _zone_bohm_characteristic_length_m(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone in self.chamber.zones:
            surfaces = [
                surface for surface in self.chamber.surfaces_by_zone.get(zone.zone_id, [])
                if ion_loss_enabled(surface.models) and ion_loss_family(surface.models) == 'bohm'
            ]
            area = sum(surface.area_m2 for surface in surfaces)
            if area <= 0.0:
                out[zone.zone_id] = 0.0
                continue
            default_length = zone.volume_m3 / max(area, 1.0e-30)
            weighted = 0.0
            for surface in surfaces:
                char_length = float(surface.models.get('characteristic_length_m') or default_length)
                weighted += surface.area_m2 * max(char_length, 1.0e-30)
            out[zone.zone_id] = weighted / area
        return out

    def _zone_effective_ion_loss_frequency_s(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for zone in self.chamber.zones:
            surfaces = [
                surface for surface in self.chamber.surfaces_by_zone.get(zone.zone_id, [])
                if ion_loss_enabled(surface.models) and ion_loss_uses_effective_frequency(ion_loss_family(surface.models))
            ]
            area = sum(surface.area_m2 for surface in surfaces)
            if area <= 0.0:
                out[zone.zone_id] = 0.0
                continue
            weighted = 0.0
            for surface in surfaces:
                rate = effective_ion_loss_frequency_s(surface.models, volume_m3=zone.volume_m3, area_m2=surface.area_m2)
                weighted += surface.area_m2 * rate
            out[zone.zone_id] = weighted / area
        return out

    def initial_state(self) -> np.ndarray:
        y0 = self.state_layout.make_state()
        self.gas_core.initialize_state(y0)
        self.surface_core.initialize_state(y0)
        return self.project_state(y0, count_diagnostics=True)

    def current_step(self, time_s: float) -> Any:
        last_idx = len(self.recipe.steps) - 1
        for idx, step in enumerate(self.recipe.steps):
            if idx < last_idx:
                if step.t_start_s <= time_s < step.t_end_s:
                    return step
            elif step.t_start_s <= time_s <= step.t_end_s + 1.0e-30:
                return step
        return self.recipe.steps[-1]

    def project_state(self, y: np.ndarray, *, count_diagnostics: bool = False) -> np.ndarray:
        y = np.array(y, dtype=float, copy=True)
        before = y.copy() if count_diagnostics else None
        y = self.gas_core.project_state(y)
        y = self.surface_core.project_state(y)
        if count_diagnostics and before is not None:
            self._record_projection_diagnostics(before, y)
        return y

    def project_trajectory(self, Y: np.ndarray, *, count_diagnostics: bool = False) -> np.ndarray:
        Y = np.array(Y, dtype=float, copy=True)
        for i in range(Y.shape[1]):
            Y[:, i] = self.project_state(Y[:, i], count_diagnostics=count_diagnostics)
        return Y

    def rhs(self, time_s: float, y: np.ndarray) -> np.ndarray:
        self._record_nonfinite_state(y)
        dydt = np.zeros_like(y)
        step = self.current_step(time_s)
        coupled = self.electrical_adapter.evaluate(time_s, y, step)
        self._sync_coupling_cache(coupled)
        self.gas_core.apply_rhs(time_s, y, step, coupled, dydt)
        self.surface_core.apply_rhs(time_s, y, step, coupled, dydt)
        if self.clip_negative:
            self.gas_core.clip_negative_rhs(y, dydt)
            self.surface_core.clip_negative_rhs(y, dydt)
        return dydt

    def jacobian(self, time_s: float, y: np.ndarray):
        J = lil_matrix((y.size, y.size), dtype=float)
        step = self.current_step(time_s)
        coupled = self.electrical_adapter.evaluate(time_s, y, step)
        self._sync_coupling_cache(coupled)
        self.gas_core.apply_jacobian(time_s, y, step, coupled, J)
        self.surface_core.apply_jacobian(time_s, y, step, coupled, J)
        return csr_matrix(J)

    def _sync_coupling_cache(self, coupled) -> None:
        self._last_power = coupled.power
        self._last_zone_electron_density_m3 = dict(coupled.ne_by_zone)
        self._last_zone_mean_energy_eV = dict(coupled.mean_e_by_zone)

    def _record_nonfinite_state(self, y: np.ndarray) -> None:
        count = int(np.count_nonzero(~np.isfinite(y)))
        if count:
            self.diagnostics['nonfinite_state_count'] = int(self.diagnostics.get('nonfinite_state_count', 0) or 0) + count

    def _record_projection_diagnostics(self, before: np.ndarray, after: np.ndarray) -> None:
        self._record_nonfinite_state(before)
        self._record_nonfinite_state(after)

        if 'gas_densities' in self.state_layout.slices:
            gas_slice = self.state_layout.slice('gas_densities')
            gas_delta = after[gas_slice] - before[gas_slice]
            changed = np.isfinite(gas_delta) & (gas_delta != 0.0)
            self.diagnostics['projection_density_clip_count'] = (
                int(self.diagnostics.get('projection_density_clip_count', 0) or 0) + int(np.count_nonzero(changed))
            )

        if 'electron_energy' in self.state_layout.slices:
            e_slice = self.state_layout.slice('electron_energy')
            e_delta = after[e_slice] - before[e_slice]
            changed = np.isfinite(e_delta) & (e_delta != 0.0)
            self.diagnostics['projection_electron_energy_clip_count'] = (
                int(self.diagnostics.get('projection_electron_energy_clip_count', 0) or 0) + int(np.count_nonzero(changed))
            )

        delta = after - before
        finite_delta = np.abs(delta[np.isfinite(delta)])
        if finite_delta.size:
            max_delta = float(np.max(finite_delta))
            self.diagnostics['projection_max_abs_delta'] = max(
                float(self.diagnostics.get('projection_max_abs_delta', 0.0) or 0.0),
                max_delta,
            )

    def _state_scale(self, y: np.ndarray) -> np.ndarray:
        scale = np.maximum(np.abs(np.asarray(y, dtype=float)), 1.0)
        if 'gas_densities' in self.state_layout.slices:
            gas_slice = self.state_layout.slice('gas_densities')
            scale[gas_slice] = np.maximum(np.abs(y[gas_slice]), self.floor_density)
        if 'electron_energy' in self.state_layout.slices:
            e_slice = self.state_layout.slice('electron_energy')
            scale[e_slice] = np.maximum(np.abs(y[e_slice]), self.floor_energy)
        if 'gas_temperature' in self.state_layout.slices:
            t_slice = self.state_layout.slice('gas_temperature')
            scale[t_slice] = np.maximum(np.abs(y[t_slice]), 50.0)
        return scale

    def _relative_rhs_norm_s_inv(self, y: np.ndarray, dydt: np.ndarray) -> float:
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(dydt)):
            return float('inf')
        return float(np.max(np.abs(dydt) / self._state_scale(y)))

    def update_final_rhs_diagnostics(self, time_s: float, y: np.ndarray) -> None:
        dydt = self.rhs(float(time_s), y)
        if np.all(np.isfinite(dydt)):
            rhs_norm = float(np.max(np.abs(dydt))) if dydt.size else 0.0
        else:
            rhs_norm = float('inf')
        self.diagnostics['final_rhs_norm_inf'] = rhs_norm
        self.diagnostics['final_relative_rhs_norm_s_inv'] = self._relative_rhs_norm_s_inv(y, dydt)

    def prescribed_electron_density_by_zone(self, time_s: float) -> dict[str, float] | None:
        if self.prescribed_electron_profile is None:
            return None
        return self.prescribed_electron_profile.density_by_zone(float(time_s), self.zone_ids)

    def compute_observables(self, t: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
        return self.observables_adapter.compute(t, y)

    def state_labels(self) -> list[str]:
        labels: list[str] = [''] * self.state_layout.size
        for zone_id, mapping in self.state_layout.gas_index.items():
            for sp_id, idx in mapping.items():
                labels[idx] = f'n[{zone_id},{sp_id}]'
        for zone_id, idx in self.state_layout.electron_energy_index.items():
            labels[idx] = f'We[{zone_id}]'
        for zone_id, idx in self.state_layout.gas_temperature_index.items():
            labels[idx] = f'Tg[{zone_id}]'
        for surface_id, mapping in self.state_layout.surface_index.items():
            for sp_id, idx in mapping.items():
                labels[idx] = f'theta[{surface_id},{sp_id}]'
        for surface_id, mapping in self.state_layout.inventory_index.items():
            for key, idx in mapping.items():
                labels[idx] = f'inventory[{surface_id},{key}]'
        for surface_id, idx in self.state_layout.film_index.items():
            labels[idx] = f'film[{surface_id}]'
        return labels

    def scipy_events(self) -> list[Any]:
        events_cfg = self.run_config.numerics.events or {}
        steady_cfg = events_cfg.get('steady_state') if isinstance(events_cfg, dict) else None
        if not isinstance(steady_cfg, dict) or not bool(steady_cfg.get('enabled', False)):
            return []

        threshold = max(float(steady_cfg.get('relative_rhs_norm_s_inv', 1.0e-3)), 0.0)
        min_step_time_s = max(float(steady_cfg.get('min_step_time_s', 0.0)), 0.0)

        def steady_state_event(time_s: float, y: np.ndarray) -> float:
            step = self.current_step(float(time_s))
            step_start = float(getattr(step, 't_start_s', time_s))
            if float(time_s) < step_start + min_step_time_s:
                return 1.0
            dydt = self.rhs(float(time_s), y)
            return self._relative_rhs_norm_s_inv(y, dydt) - threshold

        steady_state_event.terminal = True
        steady_state_event.direction = -1.0
        steady_state_event.event_name = 'steady_state'
        return [steady_state_event]
