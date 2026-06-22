from __future__ import annotations

from plasma_global.eedf.base import EEDFBackend, EEDFRequest, EEDFResult


def build_swarm_model(name: str):
    if name == 'boltzmann_2term':
        from plasma_global.eedf.boltzmann_2term import Boltzmann2TermSwarmModel

        return Boltzmann2TermSwarmModel()
    if name == 'maxwell':
        from plasma_global.eedf.maxwell import MaxwellSwarmModel

        return MaxwellSwarmModel()
    if name == 'table':
        from plasma_global.eedf.table import TabulatedSwarmModel

        return TabulatedSwarmModel()
    raise KeyError(f"Unknown swarm model: {name}. Available: ['boltzmann_2term', 'maxwell', 'table']")


class SwarmEEDFBackend(EEDFBackend):
    def __init__(self, forced_model_name: str | None = None) -> None:
        self.forced_model_name = forced_model_name

    def prepare(self, mechanism, chamber, run_config, resolved_paths) -> None:
        super().prepare(mechanism, chamber, run_config, resolved_paths)
        swarm_config = run_config.swarm
        model_name = self.forced_model_name or swarm_config.model_name or 'boltzmann_2term'
        self.swarm_model = build_swarm_model(model_name)
        self.swarm_model.prepare(
            mechanism=mechanism,
            chamber=chamber,
            run_config=run_config,
            resolved_paths=resolved_paths,
            swarm_config=swarm_config,
        )

    def evaluate(self, request: EEDFRequest) -> EEDFResult:
        if not hasattr(self, 'swarm_model'):
            self.prepare(self.mechanism, self.chamber, self.run_config, self.resolved_paths)
        return self.swarm_model.evaluate(request)
