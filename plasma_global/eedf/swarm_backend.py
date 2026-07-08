from __future__ import annotations

from plasma_global.eedf.base import EEDFBackend, EEDFRequest, EEDFResult

SWARM_MODEL_NAMES = ('boltzmann_2term', 'table')


def build_swarm_model(name: str):
    if name == 'boltzmann_2term':
        from plasma_global.eedf.boltzmann_2term import Boltzmann2TermSwarmModel

        return Boltzmann2TermSwarmModel()
    if name == 'table':
        from plasma_global.eedf.table import TabulatedSwarmModel

        return TabulatedSwarmModel()
    raise KeyError(f'Unknown swarm model: {name}. Available: {list(SWARM_MODEL_NAMES)}')


class SwarmEEDFBackend(EEDFBackend):
    def prepare(self, mechanism, chamber, run_config, resolved_paths) -> None:
        super().prepare(mechanism, chamber, run_config, resolved_paths)
        swarm_config = run_config.swarm
        model_name = swarm_config.model_name or 'table'
        self.swarm_model = build_swarm_model(model_name)
        self.swarm_model.prepare(
            mechanism=mechanism,
            chamber=chamber,
            run_config=run_config,
            resolved_paths=resolved_paths,
            swarm_config=swarm_config,
        )

    def evaluate(self, request: EEDFRequest) -> EEDFResult:
        return self.swarm_model.evaluate(request)

    def provenance(self) -> dict:
        provenance = getattr(self.swarm_model, 'provenance', None)
        return provenance() if callable(provenance) else {}
