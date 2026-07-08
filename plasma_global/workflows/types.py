"""Workflow data containers shared by case loading and case building."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from plasma_global.chemistry.models import MechanismBundle
from plasma_global.config.models import ResolvedPaths, RunConfig
from plasma_global.eedf.base import EEDFBackend
from plasma_global.electrical.base import ElectricalBackend
from plasma_global.numerics.solver_base import TimeIntegrator
from plasma_global.reactor.models import ChamberConfig, RecipeConfig

if TYPE_CHECKING:
    from plasma_global.numerics.state_layout import StateLayout
    from plasma_global.numerics.system import GlobalPlasmaSystem


@dataclass
class LoadedCase:
    run_config: RunConfig
    resolved_paths: ResolvedPaths
    chamber: ChamberConfig
    recipe: RecipeConfig
    mechanism: MechanismBundle
    validation_messages: list[dict[str, Any]]


@dataclass
class BuiltCase:
    loaded: LoadedCase
    eedf_backend: EEDFBackend
    electrical_backend: ElectricalBackend
    integrator: TimeIntegrator
    state_layout: StateLayout
    system: GlobalPlasmaSystem


__all__ = ['BuiltCase', 'LoadedCase']
