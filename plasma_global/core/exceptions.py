"""Domain-specific failures for the compact global-model core."""

from __future__ import annotations

from plasma_global.errors import (
    CaseValidationError,
    ModelDomainError,
    PlasmaGlobalError,
)


class CoreModelError(PlasmaGlobalError):
    """Base class for failures raised by the compiled core."""


class ModelConfigurationError(CaseValidationError, CoreModelError):
    """The supplied model cannot define one unambiguous physical system."""


class StateDomainError(ModelDomainError, CoreModelError):
    """An ODE state is outside the physical domain of the selected model."""


class QuasineutralityError(StateDomainError):
    """Heavy-particle charge implies a negative quasineutral electron density."""


__all__ = [
    "CoreModelError",
    "ModelConfigurationError",
    "QuasineutralityError",
    "StateDomainError",
]
