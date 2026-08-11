"""The complete public application API."""

from __future__ import annotations

from pathlib import Path

from plasma_global.core.result import SimulationResult
from plasma_global.input.load import load_case as _load_case
from plasma_global.input.schema import CaseSpec
from plasma_global.output import ResultPaths


def load_case(path: str | Path) -> CaseSpec:
    """Load and validate one schema-v3 case."""

    return _load_case(path)


def simulate(case: CaseSpec) -> SimulationResult:
    """Compile and simulate an already loaded case."""

    from plasma_global.build import simulate_case

    return simulate_case(case)


def write_result(result: SimulationResult, out: str | Path) -> ResultPaths:
    """Write the fixed ``result.h5`` and ``summary.yaml`` artifacts."""

    from plasma_global.output import write_result as _write_result

    return _write_result(result, out)


__all__ = ["load_case", "simulate", "write_result"]
