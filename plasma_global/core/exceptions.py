"""Compatibility imports for the public model exception hierarchy.

Model implementations depend on :mod:`plasma_global.errors`, the shared lower
layer.  This module remains as a stable import path for existing callers.
"""

from plasma_global.errors import (
    CoreModelError,
    ModelConfigurationError,
    QuasineutralityError,
    StateDomainError,
)

__all__ = [
    "CoreModelError",
    "ModelConfigurationError",
    "QuasineutralityError",
    "StateDomainError",
]
