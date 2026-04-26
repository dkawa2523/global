from __future__ import annotations

from typing import Any

from plasma_global.eedf.base import EEDFRequest, EEDFResult


class SwarmModel:
    """Abstract interface for swarm / EEDF closures.

    A SwarmModel consumes resolved electron-collision data and returns rate
    coefficients and transport properties. The GlobalPlasmaSystem interacts only
    with EEDFResult, so any future swarm closure (Boltzmann multi-term, Monte
    Carlo swarm, tabulated databases, neural surrogate, etc.) can be swapped in
    without changing the plasma chemistry code.
    """

    def prepare(self, mechanism: Any, chamber: Any, run_config: Any, swarm_config: Any | None = None) -> None:
        self.mechanism = mechanism
        self.chamber = chamber
        self.run_config = run_config
        self.swarm_config = swarm_config

    def evaluate(self, request: EEDFRequest) -> EEDFResult:  # pragma: no cover - interface
        raise NotImplementedError
