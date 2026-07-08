from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plasma_global.numerics.solver_base import SolverResult
from plasma_global.workflows import runner


def test_run_from_yaml_fails_fast_on_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    step = SimpleNamespace(step_id='bad_step', t_start_s=0.0, t_end_s=1.0e-9)
    loaded = SimpleNamespace(recipe=SimpleNamespace(steps=[step]))

    class FakeSystem:
        def initial_state(self) -> np.ndarray:
            return np.array([1.0])

        def project_state(self, y: np.ndarray) -> np.ndarray:
            return y

    class FailingIntegrator:
        def solve(self, **kwargs) -> SolverResult:
            return SolverResult(
                t=np.array([0.0]),
                y=np.array([[1.0]]),
                success=False,
                status=-1,
                message='forced failure',
            )

    monkeypatch.setattr(runner, 'load_case_from_yaml', lambda _path: loaded)
    monkeypatch.setattr(
        runner,
        'build_case',
        lambda _loaded: SimpleNamespace(loaded=loaded, system=FakeSystem(), integrator=FailingIntegrator()),
    )

    with pytest.raises(RuntimeError, match="bad_step.*forced failure"):
        runner.run_from_yaml('case.yaml')
