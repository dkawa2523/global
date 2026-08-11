from __future__ import annotations

import pytest

from tools import quality
from tools.quality_gate.baselines import _assert_findings_not_worse
from tools.quality_gate.checks import _policy_diff_violations
from tools.quality_gate.context import Finding, QualityFailure
from tools.quality_gate.measurements import _preserve_secret_reviews


def test_public_quality_commands_remain_available() -> None:
    assert callable(quality.fast_main)
    assert callable(quality.pr_main)
    assert callable(quality.nightly_main)
    assert callable(quality.baseline_main)


def test_policy_diff_detects_suppression_and_removed_assertion() -> None:
    suppression = "# " + "no" + "qa"
    diff = "\n".join(
        (
            "+++ b/tests/test_example.py",
            f"+{suppression}",
            "-assert conserved",
        )
    )

    assert _policy_diff_violations(diff) == [
        suppression,
        "removed test contract: assert conserved",
    ]


def test_finding_comparison_rejects_new_issue() -> None:
    finding = Finding("ruff|F821|module.py|run|undefined", "module.py:1 F821")

    with pytest.raises(QualityFailure, match="New or worsened ruff findings"):
        _assert_findings_not_worse(
            "ruff",
            [finding],
            {"ruff": {"findings": {}}},
        )


def test_secret_baseline_update_preserves_review_decision() -> None:
    finding = {"type": "checksum", "hashed_secret": "fingerprint"}
    current = {"results": {"tests/example.py": [finding.copy()]}}
    previous_finding = {**finding, "is_secret": False}
    previous = {"results": {"tests\\example.py": [previous_finding]}}

    merged = _preserve_secret_reviews(current, previous)

    assert merged["results"]["tests/example.py"][0]["is_secret"] is False
