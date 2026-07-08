from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ProcessCore:
    system: Any

    @property
    def enabled(self) -> bool:
        return bool(self.system.state_layout.extra_state_index)

    def initialize_state(self, y0: np.ndarray) -> None:
        for spec in self.system.mechanism.state_variables:
            for idx in self.system.state_layout.extra_state_index.get(spec.state_id, {}).values():
                y0[idx] = float(spec.initial)

    def project_state(self, y: np.ndarray) -> np.ndarray:
        for spec in self.system.mechanism.state_variables:
            if spec.lower_bound is None:
                continue
            for idx in self.system.state_layout.extra_state_index.get(spec.state_id, {}).values():
                y[idx] = max(float(y[idx]), float(spec.lower_bound))
        return y

    def clip_negative_rhs(self, y: np.ndarray, dydt: np.ndarray) -> None:
        for spec in self.system.mechanism.state_variables:
            if spec.lower_bound is None:
                continue
            lower = float(spec.lower_bound)
            for idx in self.system.state_layout.extra_state_index.get(spec.state_id, {}).values():
                if y[idx] <= lower and dydt[idx] < 0.0:
                    dydt[idx] = 0.0

    def apply_rhs(self, _time_s: float, y: np.ndarray, _step: Any, coupled: Any, dydt: np.ndarray) -> None:
        for process in self.system.mechanism.processes:
            target = self.system.mechanism.state_variable_by_id[process.target]
            if process.kind == 'source':
                self._apply_source(process, target, dydt)
            elif process.kind == 'relaxation':
                self._apply_relaxation(process, target, y, dydt)
            elif process.kind == 'ion_flux_source':
                self._apply_ion_flux_source(process, target, coupled, dydt)

    def _owner_ids(self, process: Any, target: Any) -> list[str]:
        available = self.system.state_layout.extra_state_index.get(target.state_id, {})
        if target.scope == 'zone' and process.zones:
            return [zone_id for zone_id in process.zones if zone_id in available]
        if target.scope == 'surface' and process.surfaces:
            return [surface_id for surface_id in process.surfaces if surface_id in available]
        return list(available)

    def _apply_source(self, process: Any, target: Any, dydt: np.ndarray) -> None:
        value = float(process.parameters['value'])
        for owner_id in self._owner_ids(process, target):
            idx = self.system.state_layout.extra_state_index[target.state_id][owner_id]
            dydt[idx] += value

    def _apply_relaxation(self, process: Any, target: Any, y: np.ndarray, dydt: np.ndarray) -> None:
        tau_s = max(float(process.parameters['tau_s']), 1.0e-30)
        equilibrium = float(process.parameters.get('equilibrium', 0.0))
        for owner_id in self._owner_ids(process, target):
            idx = self.system.state_layout.extra_state_index[target.state_id][owner_id]
            dydt[idx] += (equilibrium - float(y[idx])) / tau_s

    def _apply_ion_flux_source(self, process: Any, target: Any, coupled: Any, dydt: np.ndarray) -> None:
        coefficient = float(process.parameters['coefficient'])
        for surface_id in self._owner_ids(process, target):
            surface = self.system.chamber.surface_by_id[surface_id]
            z = self.system.zone_index[surface.zone_id]
            flux = self.system.surface_core.surface_ion_flux_m2_s(
                surface_id,
                surface.zone_id,
                coupled.gas[z],
                coupled,
            )
            idx = self.system.state_layout.extra_state_index[target.state_id][surface_id]
            dydt[idx] += coefficient * flux
