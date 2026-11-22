"""Public exception hierarchy for configuration and simulation failures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plasma_global.core.result import SimulationResult


class PlasmaGlobalError(Exception):
    """Base class for expected, user-facing failures."""


class CaseValidationError(PlasmaGlobalError, ValueError):
    """The input case is syntactically or semantically invalid."""


class ChemistryError(CaseValidationError):
    """The chemistry mechanism cannot be compiled safely."""


class ModelDomainError(PlasmaGlobalError, RuntimeError):
    """The evolving state left the validity domain of a selected closure."""


class CouplingConvergenceError(PlasmaGlobalError, RuntimeError):
    """An algebraic electrical/EEDF coupling failed to converge."""


class IntegrationError(PlasmaGlobalError, RuntimeError):
    """The time integrator failed or produced a materially invalid state."""

    def __init__(
        self,
        message: str,
        *,
        partial_result: SimulationResult | None = None,
        failure_time_s: float | None = None,
        failure_segment_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_result = partial_result
        self.failure_time_s = failure_time_s
        self.failure_segment_id = failure_segment_id


class ModelConfigurationError(CaseValidationError):
    """The supplied model cannot define one unambiguous physical system."""


class StateDomainError(ModelDomainError):
    """An evolving state is outside the validity domain of its model."""


class QuasineutralityError(StateDomainError):
    """Heavy-particle charge implies a negative electron density."""


__all__ = [
    "CaseValidationError",
    "ChemistryError",
    "CouplingConvergenceError",
    "IntegrationError",
    "ModelConfigurationError",
    "ModelDomainError",
    "PlasmaGlobalError",
    "QuasineutralityError",
    "StateDomainError",
]
