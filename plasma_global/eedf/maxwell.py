from __future__ import annotations

import math

from plasma_global.eedf.base import EEDFRequest, EEDFResult, EEDFTransport
from plasma_global.eedf.swarm_backend import SwarmEEDFBackend
from plasma_global.eedf.swarm_base import SwarmModel


class MaxwellSwarmModel(SwarmModel):
    """Analytic Maxwellian swarm model.

    This model does not use transport cross sections; it is a cheap analytic
    backend for parity checks and development cases.
    """

    def evaluate(self, request: EEDFRequest) -> EEDFResult:
        eps = max(float(request.mean_energy_eV), 1.0e-3)
        k_map: dict[str, float] = {}
        dk_map: dict[str, float] = {}
        for cs_id, cs in self.mechanism.cross_sections.items():
            A = float(cs.prefactor_m3_s)
            p = float(cs.exponent)
            Eth = float(cs.threshold_eV)
            k = A * (eps ** p) * math.exp(-Eth / eps)
            dk = k * (p / eps + Eth / (eps * eps))
            k_map[cs_id] = k
            dk_map[cs_id] = dk
        return EEDFResult(
            rate_coefficients=k_map,
            d_rate_d_mean_energy_eV=dk_map,
            transport=EEDFTransport(
                mean_energy_eV=eps,
                mobility_m2_V_s=0.1 / max(request.pressure_Pa, 1.0),
                diffusion_m2_s=0.1 * eps / max(request.pressure_Pa, 1.0),
                effective_field_Td=float(request.reduced_field_Td or 0.0),
                lookup_mode='mean_energy',
            ),
        )


class MaxwellEEDFBackend(SwarmEEDFBackend):
    def __init__(self) -> None:
        super().__init__(forced_model_name='maxwell')
