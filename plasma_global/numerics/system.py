from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plasma_global.chemistry.models import MechanismBundle
from plasma_global.eedf.base import EEDFBackend
from plasma_global.electrical.base import ElectricalBackend
from plasma_global.electrical.coupling_adapter import ElectricalCouplingAdapter
from plasma_global.numerics.state_layout import StateLayout
from plasma_global.physics.gas_phase_core import GasPhaseCore
from plasma_global.physics.prescribed_electrons import build_prescribed_electron_profile
from plasma_global.physics.surface_core import SurfaceCore
from plasma_global.physics.wall_loss import zone_ion_loss_properties
from plasma_global.reactor.models import ChamberConfig, RecipeConfig


@dataclass
class GlobalPlasmaSystem:
    """Top-level transient multiphysics coordinator.

    This class intentionally keeps the public solver-facing interface small
    (`initial_state`, `rhs`, `project_state`, `project_trajectory`,
    `state_labels`, `scipy_events`),
    while delegating most domain logic to dedicated components:

    - `GasPhaseCore`
    - `SurfaceCore`
    - `ElectricalCouplingAdapter`

    The goal is to preserve compatibility with the existing workflow while
    making the central coupling point smaller and easier to reason about.
    """

    mechanism: MechanismBundle
    chamber: ChamberConfig
    recipe: RecipeConfig
    run_config: Any
    resolved_paths: Any
    eedf_backend: EEDFBackend
    electrical_backend: ElectricalBackend
    state_layout: StateLayout

    def __post_init__(self) -> None:
        self._init_layout_metadata()
        self._init_positivity_config()
        self._init_species_metadata()
        self._init_wall_loss_metadata()
        self._init_electron_density_closure()
        self._init_components()

    def _init_layout_metadata(self) -> None:
        self.zone_ids = list(self.state_layout.zone_ids)
        self.zone_index = {z: i for i, z in enumerate(self.zone_ids)}
        self.n_zones = len(self.zone_ids)
        self.gas_species_ids = list(self.state_layout.gas_species_ids)
        self.gas_species_index = {s: i for i, s in enumerate(self.gas_species_ids)}
        self.n_gas_species = len(self.gas_species_ids)
        self.surface_ids = list(self.state_layout.surface_index.keys())

    def _init_positivity_config(self) -> None:
        self.positivity_cfg = self.run_config.numerics.positivity or {}
        self.floor_density = float(self.positivity_cfg.get('floor_density_m3', 1.0))
        self.floor_energy = float(self.positivity_cfg.get('floor_energy_J_m3', 1.0e-12))
        self.clip_negative = bool(self.positivity_cfg.get('clip_negative', True))

    def _init_species_metadata(self) -> None:
        self.gas_species = [self.mechanism.species_by_id[s] for s in self.gas_species_ids]
        self.gas_charges = np.array([sp.charge for sp in self.gas_species], dtype=float)
        self.gas_masses = np.array([max(sp.mass_kg, 1.0e-30) for sp in self.gas_species], dtype=float)
        self.surface_species_tags = {sp.canonical_id: set(sp.state_tags) for sp in self.mechanism.surface_species}
        self.positive_ion_local_indices = [i for i, sp in enumerate(self.gas_species) if sp.charge > 0]

    def _init_wall_loss_metadata(self) -> None:
        ion_loss = zone_ion_loss_properties(self.chamber)
        self.zone_ion_loss_family = ion_loss.family
        self.zone_ion_loss_area = ion_loss.area_m2
        self.zone_ion_loss_h_factor = ion_loss.h_factor
        self.zone_effective_ion_loss_frequency_s = ion_loss.effective_frequency_s

    def _init_electron_density_closure(self) -> None:
        self.electron_density_closure = str(getattr(self.run_config.physics, 'electron_density_closure', 'quasi_neutral') or 'quasi_neutral').lower()
        self.prescribed_electron_profile = build_prescribed_electron_profile(
            self.run_config,
            self.resolved_paths,
            self.zone_ids,
        )

    def _init_components(self) -> None:
        self.gas_core = GasPhaseCore(self)
        self.surface_core = SurfaceCore(self)
        self.electrical_adapter = ElectricalCouplingAdapter(self)

    @property
    def zone_wall_temperature(self) -> dict[str, float]:
        return self.gas_core.zone_wall_temperature

    def initial_state(self) -> np.ndarray:
        y0 = self.state_layout.make_state()
        self.gas_core.initialize_state(y0)
        self.surface_core.initialize_state(y0)
        return self.project_state(y0)

    def current_step(self, time_s: float) -> Any:
        last_idx = len(self.recipe.steps) - 1
        for idx, step in enumerate(self.recipe.steps):
            if idx < last_idx:
                if step.t_start_s <= time_s < step.t_end_s:
                    return step
            elif step.t_start_s <= time_s <= step.t_end_s + 1.0e-30:
                return step
        return self.recipe.steps[-1]

    def project_state(self, y: np.ndarray) -> np.ndarray:
        y = np.array(y, dtype=float, copy=True)
        y = self.gas_core.project_state(y)
        return self.surface_core.project_state(y)

    def project_trajectory(self, Y: np.ndarray) -> np.ndarray:
        Y = np.array(Y, dtype=float, copy=True)
        for i in range(Y.shape[1]):
            Y[:, i] = self.project_state(Y[:, i])
        return Y

    def rhs(self, time_s: float, y: np.ndarray) -> np.ndarray:
        dydt = np.zeros_like(y)
        step = self.current_step(time_s)
        coupled = self.electrical_adapter.evaluate(time_s, y, step)
        self.gas_core.apply_rhs(time_s, y, step, coupled, dydt)
        if self.surface_core.enabled:
            self.surface_core.apply_rhs(time_s, y, step, coupled, dydt)
        self._apply_positivity_guard(y, dydt)
        return dydt

    def _apply_positivity_guard(self, y: np.ndarray, dydt: np.ndarray) -> None:
        if not self.clip_negative:
            return
        self.gas_core.clip_negative_rhs(y, dydt)
        if self.surface_core.enabled:
            self.surface_core.clip_negative_rhs(y, dydt)

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

    def prescribed_electron_density_by_zone(self, time_s: float) -> dict[str, float] | None:
        if self.prescribed_electron_profile is None:
            return None
        return self.prescribed_electron_profile.density_by_zone(float(time_s), self.zone_ids)

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
