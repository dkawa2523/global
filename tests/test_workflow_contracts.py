from __future__ import annotations

from types import SimpleNamespace

from plasma_global.numerics.system import GlobalPlasmaSystem


def test_recipe_boundary_selects_next_step() -> None:
    fake_system = SimpleNamespace(
        recipe=SimpleNamespace(
            steps=[
                SimpleNamespace(step_id='first', t_start_s=0.0, t_end_s=1.0),
                SimpleNamespace(step_id='second', t_start_s=1.0, t_end_s=2.0),
            ]
        )
    )

    assert GlobalPlasmaSystem.current_step(fake_system, 1.0).step_id == 'second'
