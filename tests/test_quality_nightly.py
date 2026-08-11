from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from plasma_global import load_case, simulate
from plasma_global.audit import audit_case

ROOT = Path(__file__).parents[1]
SMOKE_CASE = ROOT / "examples" / "v3" / "cases" / "smoke.yaml"


@pytest.mark.nightly
def test_smoke_case_stays_finite_within_resource_budget() -> None:
    import resource

    started = time.perf_counter()
    case = load_case(SMOKE_CASE)
    result = simulate(case)
    audit = audit_case(case)
    elapsed_s = time.perf_counter() - started
    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    assert result.status.success
    assert result.state.dtype == np.dtype(np.float64)
    assert result.state.shape == (result.n_times, result.n_states)
    assert np.isfinite(result.time_s).all()
    assert np.isfinite(result.state).all()
    assert audit.passed
    for quantity in (
        "particle_ledger_normalized",
        "electron_energy_ledger_normalized",
        "heavy_energy_ledger_normalized",
    ):
        assert audit.conservation_max_abs_residual[quantity] < 1.0e-12
    assert elapsed_s <= 30.0
    assert max_rss_bytes <= 1024**3

    report = ROOT / ".quality-reports" / "nightly-performance.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "case": SMOKE_CASE.relative_to(ROOT).as_posix(),
                "elapsed_s": elapsed_s,
                "conservation_max_abs_residual": dict(
                    audit.conservation_max_abs_residual
                ),
                "max_rss_bytes": max_rss_bytes,
                "n_states": result.n_states,
                "n_times": result.n_times,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
