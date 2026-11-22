"""Public declarations, preparation, simulation, and result publication."""

from __future__ import annotations

from plasma_global.build import PreparedCase, compile_case
from plasma_global.catalog import list_models
from plasma_global.core.result import SimulationResult
from plasma_global.input.load import case_from_mapping, load_case
from plasma_global.input.schema import CaseSpec
from plasma_global.output import (
    ArtifactReport,
    ArtifactStatus,
    ResultPaths,
    write_result,
)


def simulate(case: CaseSpec | PreparedCase) -> SimulationResult:
    """Simulate a declaration or reuse an already prepared input snapshot."""
    from plasma_global.build import simulate_case

    return simulate_case(case)


__all__ = [
    "ArtifactReport",
    "ArtifactStatus",
    "CaseSpec",
    "PreparedCase",
    "ResultPaths",
    "case_from_mapping",
    "compile_case",
    "list_models",
    "load_case",
    "simulate",
    "write_result",
]
