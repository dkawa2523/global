"""Orchestrate one compiled physical-runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from plasma_global.core._runtime_assembly import (
    RuntimeAssembly,
    RuntimeEvaluation,
)
from plasma_global.core._runtime_electrons import RuntimeElectronEvaluator
from plasma_global.core._runtime_sources import RuntimeSourceEvaluator
from plasma_global.core._runtime_state import RuntimeStatePreparer
from plasma_global.core.domain import RecipeSegment

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel


@dataclass(frozen=True, slots=True)
class RuntimeEvaluator:
    """Coordinate state preparation, physical terms, and RHS assembly."""

    model: CompiledGlobalModel

    def evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool = True,
    ) -> RuntimeEvaluation:
        """Evaluate the private diagnostic payload used by the public facade."""

        result = self._evaluate(
            time_s,
            state,
            segment,
            collect_ledger=collect_ledger,
            derivative_only=False,
        )
        if not isinstance(result, RuntimeEvaluation):
            raise RuntimeError("diagnostic evaluation returned only a derivative")
        return result

    def evaluate_derivative(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        domain_atol: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate only dy/dt without materializing diagnostic wrappers."""

        result = self._evaluate(
            time_s,
            state,
            segment,
            collect_ledger=False,
            derivative_only=True,
            domain_atol=domain_atol,
        )
        if not isinstance(result, np.ndarray):
            raise RuntimeError("derivative evaluation returned diagnostic data")
        return result

    def _evaluate(
        self,
        time_s: float,
        state: np.ndarray,
        segment: RecipeSegment,
        *,
        collect_ledger: bool,
        derivative_only: bool,
        domain_atol: np.ndarray | None = None,
    ) -> RuntimeEvaluation | np.ndarray:
        prepared = RuntimeStatePreparer(self.model).prepare(
            time_s, state, segment, domain_atol
        )
        power = RuntimeElectronEvaluator(self.model).evaluate_power(
            time_s, segment, prepared
        )
        sources = RuntimeSourceEvaluator(self.model)
        transport = sources.evaluate_transport(segment, prepared)
        reactions = sources.evaluate_reactions_and_walls(
            time_s, prepared, power, collect_ledger=collect_ledger
        )
        surface = sources.evaluate_surface_terms(
            segment, prepared, reactions, collect_ledger=collect_ledger
        )
        energy = sources.evaluate_energy_terms(segment, prepared, power)
        derivative, ledger_by_zone = RuntimeAssembly(
            model=self.model,
            extension_rhs=sources.extension_rhs,
            prepared=prepared,
            power=power,
            transport=transport,
            reactions=reactions,
            surface=surface,
            energy=energy,
            collect_ledger=collect_ledger,
        ).assemble()

        if derivative_only:
            derivative.setflags(write=False)
            return derivative
        return RuntimeEvaluation(
            derivative=derivative,
            electron_states=power.electron_states,
            kinetics_by_zone=power.kinetics_by_zone,
            gas_temperature_K_by_zone={
                zone.zone_id: float(prepared.gas_temperature_by_zone[index])
                for index, zone in enumerate(self.model.zones)
            },
            charge_residual_m3_by_zone=prepared.charge_residual_by_zone,
            ledger_by_zone=ledger_by_zone,
            power_coupling=power.coupling,
        )


__all__ = ["RuntimeEvaluator"]
