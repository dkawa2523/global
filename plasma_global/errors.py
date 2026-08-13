"""Public exception hierarchy for configuration and simulation failures."""

from __future__ import annotations


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


class CoreModelError(PlasmaGlobalError):
    """Base class for failures raised while compiling or evaluating a model."""


class ModelConfigurationError(CaseValidationError, CoreModelError):
    """The supplied model cannot define one unambiguous physical system."""


class StateDomainError(ModelDomainError, CoreModelError):
    """An evolving state is outside the validity domain of its model."""


class QuasineutralityError(StateDomainError):
    """Heavy-particle charge implies a negative electron density."""


class MigrationError(PlasmaGlobalError, ValueError):
    """A legacy case cannot be converted without an explicit user choice."""


__all__ = [
    "CaseValidationError",
    "ChemistryError",
    "CoreModelError",
    "CouplingConvergenceError",
    "IntegrationError",
    "MigrationError",
    "ModelConfigurationError",
    "ModelDomainError",
    "PlasmaGlobalError",
    "QuasineutralityError",
    "StateDomainError",
]
