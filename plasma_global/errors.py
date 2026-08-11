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


class MigrationError(PlasmaGlobalError, ValueError):
    """A legacy case cannot be converted without an explicit user choice."""


__all__ = [
    "CaseValidationError",
    "ChemistryError",
    "CouplingConvergenceError",
    "IntegrationError",
    "MigrationError",
    "ModelDomainError",
    "PlasmaGlobalError",
]
