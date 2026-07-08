"""Build prepared runtime objects from already loaded case inputs."""

from __future__ import annotations

from typing import cast

from plasma_global.eedf.base import EEDFBackend
from plasma_global.eedf.registry import EEDF_REGISTRY
from plasma_global.electrical.base import ElectricalBackend
from plasma_global.electrical.registry import ELECTRICAL_REGISTRY
from plasma_global.numerics.registry import INTEGRATOR_REGISTRY
from plasma_global.numerics.solver_base import TimeIntegrator
from plasma_global.numerics.state_layout import build_state_layout
from plasma_global.workflows.types import BuiltCase, LoadedCase


def _build_backends(loaded: LoadedCase) -> tuple[EEDFBackend, ElectricalBackend, TimeIntegrator]:
    run_config = loaded.run_config
    eedf = cast(EEDFBackend, EEDF_REGISTRY.build(run_config.physics.eedf_backend))
    electrical = cast(ElectricalBackend, ELECTRICAL_REGISTRY.build(run_config.physics.electrical_backend))
    integrator = cast(TimeIntegrator, INTEGRATOR_REGISTRY.build(run_config.physics.integrator, run_config=run_config))
    return eedf, electrical, integrator


def _prepare_backends(loaded: LoadedCase, eedf: EEDFBackend, electrical: ElectricalBackend) -> None:
    eedf.prepare(
        mechanism=loaded.mechanism,
        chamber=loaded.chamber,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
    )
    electrical.prepare(
        chamber=loaded.chamber,
        recipe=loaded.recipe,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
    )


def build_case(loaded: LoadedCase) -> BuiltCase:
    from plasma_global.numerics.system import GlobalPlasmaSystem

    eedf, electrical, integrator = _build_backends(loaded)
    _prepare_backends(loaded, eedf, electrical)
    layout = build_state_layout(loaded.mechanism, loaded.chamber, loaded.run_config)
    system = GlobalPlasmaSystem(
        mechanism=loaded.mechanism,
        chamber=loaded.chamber,
        recipe=loaded.recipe,
        run_config=loaded.run_config,
        resolved_paths=loaded.resolved_paths,
        eedf_backend=eedf,
        electrical_backend=electrical,
        state_layout=layout,
    )
    return BuiltCase(
        loaded=loaded,
        eedf_backend=eedf,
        electrical_backend=electrical,
        integrator=integrator,
        state_layout=layout,
        system=system,
    )


__all__ = ['build_case']
