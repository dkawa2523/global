from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "benchmarks" / "v2_baseline.yaml"


def test_v2_migration_baseline_has_stable_schema_and_finite_records() -> None:
    assert BASELINE.is_file()
    document = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))

    assert set(document) == {
        "schema_version",
        "captured_at",
        "purpose",
        "environment",
        "capture_completeness",
        "legacy_suite",
        "cases",
        "notes",
    }
    assert document["schema_version"] == 2
    assert str(document["captured_at"]) == str(date(2026, 8, 9))
    assert "migration comparison only" in document["purpose"]

    environment = document["environment"]
    assert set(environment) == {"python", "platform"}
    assert all(isinstance(value, str) and value for value in environment.values())

    completeness = document["capture_completeness"]
    assert completeness == {
        "legacy_test_outcome": "recorded",
        "final_case_values": "recorded",
        "single_run_elapsed_time": "recorded_orientation_only",
        "conservation_balances": "not_recorded",
        "external_table_read_counts": "not_recorded",
        "rhs_evaluation_timing": "not_recorded",
    }
    assert all(
        completeness[name] == "not_recorded"
        for name in (
            "conservation_balances",
            "external_table_read_counts",
            "rhs_evaluation_timing",
        )
    )

    suite = document["legacy_suite"]
    assert set(suite) == {"passed", "deselected", "coverage_percent"}
    assert isinstance(suite["passed"], int)
    assert suite["passed"] > 0
    assert isinstance(suite["deselected"], int)
    assert suite["deselected"] >= 0
    assert 0 <= suite["coverage_percent"] <= 100

    assert set(document["cases"]) == {
        "crane_two_reaction_argon",
        "smoke",
        "zdplaskin_example2",
    }
    for record in document["cases"].values():
        assert set(record) == {"elapsed_s", "nfev", "saved_points", "final"}
        assert math.isfinite(record["elapsed_s"])
        assert record["elapsed_s"] >= 0.0
        assert isinstance(record["nfev"], int)
        assert record["nfev"] > 0
        assert isinstance(record["saved_points"], int)
        assert record["saved_points"] > 0
        assert record["final"]
        assert all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in record["final"].values()
        )

    assert document["notes"]
    assert all(isinstance(note, str) and note for note in document["notes"])
