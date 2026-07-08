"""Built-in EEDF backend registry."""

from __future__ import annotations

from plasma_global.backends import BackendCatalog, BackendSpec
from plasma_global.eedf.maxwell import MaxwellEEDFBackend
from plasma_global.eedf.swarm_backend import SwarmEEDFBackend


EEDF_REGISTRY = BackendCatalog(
    {
        'swarm': BackendSpec(
            lambda **kwargs: SwarmEEDFBackend(),
            'Swarm wrapper for swarm.model_name: table; boltzmann_2term is experimental.',
            maturity='configuration_dependent',
            intended_use='Prepared HDF5 swarm tables for production-style cases; internal boltzmann_2term for compact studies.',
            caveat='The boltzmann_2term model is approximate and should not replace a mature external swarm solver.',
        ),
        'maxwell': BackendSpec(
            lambda **kwargs: MaxwellEEDFBackend(),
            'Analytic Maxwellian closure.',
            maturity='development',
            intended_use='Smoke tests, rough estimates, and parity checks.',
            caveat='Transport and rates are analytic approximations, not calibrated electron kinetics.',
        ),
    }
)


__all__ = ['EEDF_REGISTRY']
