"""Built-in time-integrator registry."""

from __future__ import annotations

from plasma_global.backends import BackendCatalog, BackendSpec
from plasma_global.numerics.scipy_backend import SciPyBDFIntegrator


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
            maturity='stable',
            intended_use='Default stiff ODE integration for global-model recipe steps.',
            caveat='No analytic Jacobian is supplied by default.',
        )
    }
)


__all__ = ['INTEGRATOR_REGISTRY']
