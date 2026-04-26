from __future__ import annotations

from plasma_global.core.registry import Registry
from plasma_global.eedf.base import EEDFBackend, EEDFRequest, EEDFResult


SWARM_MODEL_REGISTRY = Registry()


class SwarmEEDFBackend(EEDFBackend):
    def __init__(self, forced_model_name: str | None = None) -> None:
        self.forced_model_name = forced_model_name

    def prepare(self, mechanism, chamber, run_config) -> None:
        super().prepare(mechanism, chamber, run_config)
        swarm_config = getattr(run_config, 'swarm', None)
        model_name = self.forced_model_name or getattr(swarm_config, 'model_name', None) or 'boltzmann_2term'
        self.swarm_model = SWARM_MODEL_REGISTRY.build(model_name)
        self.swarm_model.prepare(mechanism=mechanism, chamber=chamber, run_config=run_config, swarm_config=swarm_config)

    def evaluate(self, request: EEDFRequest) -> EEDFResult:
        if not hasattr(self, 'swarm_model'):
            self.prepare(self.mechanism, self.chamber, self.run_config)
        return self.swarm_model.evaluate(request)
