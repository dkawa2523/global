from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from plasma_global.eedf.boltzmann_2term import Boltzmann2TermBackend
from plasma_global.eedf.maxwell import MaxwellEEDFBackend
from plasma_global.eedf.swarm_backend import SwarmEEDFBackend
from plasma_global.eedf.table import TableEEDFBackend
from plasma_global.electrical.ccp import CCPBackend
from plasma_global.electrical.dc_series import DCSeriesCircuitBackend
from plasma_global.electrical.direct_power import DirectPowerBackend
from plasma_global.electrical.external_table import ExternalCircuitTableBackend
from plasma_global.electrical.icp import ICPBackend
from plasma_global.electrical.rf_envelope import RFEnvelopeBackend
from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator


@dataclass(frozen=True)
class BackendSpec:
    builder: Callable[..., Any]
    description: str


@dataclass(frozen=True)
class BackendCatalog:
    entries: dict[str, BackendSpec]

    def build(self, name: str, **kwargs: Any) -> Any:
        if name not in self.entries:
            raise KeyError(f"Unknown backend: {name}. Available: {self.names()}")
        return self.entries[name].builder(**kwargs)

    def names(self) -> list[str]:
        return sorted(self.entries)

    def details(self) -> dict[str, dict[str, str]]:
        return {
            name: {'description': spec.description}
            for name, spec in sorted(self.entries.items())
        }


EEDF_REGISTRY = BackendCatalog(
    {
        'swarm': BackendSpec(lambda **kwargs: SwarmEEDFBackend(), 'Swarm wrapper with replaceable swarm model backend.'),
        'maxwell': BackendSpec(lambda **kwargs: MaxwellEEDFBackend(), 'Analytic Maxwellian closure.'),
        'boltzmann_2term': BackendSpec(lambda **kwargs: Boltzmann2TermBackend(), 'Internal two-term Boltzmann swarm backend.'),
        'rate_table': BackendSpec(lambda **kwargs: TableEEDFBackend(), 'Interpolated external rate-table backend.'),
    }
)

ELECTRICAL_REGISTRY = BackendCatalog(
    {
        'direct_power': BackendSpec(lambda **kwargs: DirectPowerBackend(), 'Direct absorbed-power prescription.'),
        'dc_series_circuit': BackendSpec(
            lambda **kwargs: DCSeriesCircuitBackend(),
            'Reduced voltage-source plus ballast-resistor circuit for DC and pulsed DC cases.',
        ),
        'external_circuit_table': BackendSpec(
            lambda **kwargs: ExternalCircuitTableBackend(),
            'One-way coupling from measured or SPICE-generated circuit waveform CSV data.',
        ),
        'rf_envelope': BackendSpec(lambda **kwargs: RFEnvelopeBackend(), 'Cycle-averaged HF/LF RF power and bias envelope model.'),
        'ccp': BackendSpec(lambda **kwargs: CCPBackend(), 'Reduced CCP / bias backend.'),
        'icp': BackendSpec(lambda **kwargs: ICPBackend(), 'Reduced ICP source-coupling backend.'),
    }
)

INTEGRATOR_REGISTRY = BackendCatalog(
    {
        'scipy_bdf': BackendSpec(
            lambda *, run_config: SciPyBDFIntegrator(
                rtol=run_config.numerics.rtol,
                atol=run_config.numerics.atol,
                first_step=run_config.numerics.first_step,
                max_step=run_config.numerics.max_step,
            ),
            'SciPy solve_ivp(method="BDF") backend.',
        )
    }
)


__all__ = [
    'BackendCatalog',
    'BackendSpec',
    'EEDF_REGISTRY',
    'ELECTRICAL_REGISTRY',
    'INTEGRATOR_REGISTRY',
]
