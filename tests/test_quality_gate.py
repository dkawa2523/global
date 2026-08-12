from __future__ import annotations

import pytest

from tools import quality
from tools.quality_gate.baselines import _assert_findings_not_worse
from tools.quality_gate.context import Finding, QualityFailure
from tools.quality_gate.measurements import (
    _preserve_secret_reviews,
    _secret_fingerprints,
    _tracked_files,
)
from tools.quality_gate.test_policy import (
    _policy_diff_violations,
    _snapshot,
    _snapshot_violations,
)


def _policy_violations(previous: str, current: str) -> list[str]:
    return _snapshot_violations(
        _snapshot(previous, namespace="tests/example.py"),
        _snapshot(current, namespace="tests/example.py"),
    )


def test_public_quality_commands_remain_available() -> None:
    assert callable(quality.fast_main)
    assert callable(quality.pr_main)
    assert callable(quality.nightly_main)
    assert callable(quality.baseline_main)


def test_policy_diff_detects_suppression_and_removed_assertion() -> None:
    directives = (
        "# " + "no" + "qa",
        "# ruff: " + "noqa",
        "# flake8: " + "noqa",
        "# pyrefly: " + "ignore",
        "# pragma: " + "no branch",
        "# pragma: " + "allowlist secret",
    )
    diff = "\n".join(
        (
            "+++ b/tests/test_example.py",
            *(f"+{directive}" for directive in directives),
            "-def test_removed():",
        )
    )

    assert _policy_diff_violations(diff) == list(directives)


def test_secret_suppressions_are_rejected_outside_python() -> None:
    allowlist = "# pragma: " + "allowlist secret"
    whitelist = "# pragma: " + "whitelist nextline secret"
    diff = "\n".join(
        (
            "+++ b/config.yaml",
            f"+token: value  {allowlist}",
            "+" + "# " + "noqa",
            "+++ b/data.csv",
            f"+{whitelist}",
        )
    )

    assert _policy_diff_violations(diff) == [
        f"token: value  {allowlist}",
        whitelist,
    ]


def test_policy_diff_leaves_skip_detection_to_the_ast_snapshot() -> None:
    diff = "\n".join(
        (
            "+++ b/tools/quality_gate/example.py",
            '+SKIP_NAMES = {"pytest.skip", "pytest.mark.xfail"}',
        )
    )

    assert _policy_diff_violations(diff) == []


def test_raises_policy_allows_strengthening_and_argument_updates() -> None:
    previous = """
def test_contract():
    with pytest.raises(ValueError, match="cache.max_entries"):
        run()
"""
    current = r"""
def test_contract():
    with pytest.raises(TypeError):
        prepare()
    with pytest.raises(ValueError, match=r"cache\.max_entries"):
        prepared = build_config(mode="strict")
        run(prepared)
"""

    assert _policy_violations(previous, current) == []


def test_raises_policy_allows_same_call_additions_and_nested_arguments() -> None:
    previous = """
def test_contract(config):
    with pytest.raises(ValueError, match="invalid"):
        run(config)
"""
    added = """
def test_contract(config):
    with pytest.raises(TypeError):
        run(other)
    with pytest.raises(ValueError, match="invalid"):
        run(config)
"""
    nested_argument = """
def test_contract(config):
    with pytest.raises(ValueError, match="invalid"):
        run(build_config(config))
"""
    renamed_api = """
def test_contract(config):
    with pytest.raises(ValueError, match="invalid"):
        execute(config)
"""

    assert _policy_violations(previous, added) == []
    assert _policy_violations(previous, nested_argument) == []
    assert _policy_violations(previous, renamed_api) == []


def test_raises_policy_rejects_binding_and_body_weakening() -> None:
    previous = """
ERROR = ValueError
CASES = ["specific"]

@pytest.mark.parametrize("message", CASES)
class TestErrors:
    def test_contract(self, message):
        with pytest.raises(ERROR, match=message):
            run()
"""
    weakened = """
ERROR = Exception
CASES = [".*"]

@pytest.mark.parametrize("message", CASES)
class TestErrors:
    def test_contract(self, message):
        with pytest.raises(ERROR, match=message):
            raise ValueError("specific")
"""

    violations = _policy_violations(previous, weakened)

    assert len(violations) == 1
    assert "pytest.raises" in violations[0]

    substituted = previous.replace(
        "        with pytest.raises(ERROR, match=message):\n            run()",
        "        with pytest.raises(ERROR, match=message):\n"
        "            run(other)\n"
        "        with pytest.raises(Exception, match='.*'):\n"
        "            run()",
    )
    assert "pytest.raises" in _policy_violations(previous, substituted)[0]


def test_plain_assertion_policy_rejects_removal_and_constant_truth() -> None:
    previous = """
def test_contract(actual):
    assert actual == 4
    assert actual > 0
"""
    changed_expectation = """
def test_contract(actual):
    assert actual == 5
    assert actual > 1
"""
    removed = """
def test_contract(actual):
    assert actual == 4
"""
    constant_true = """
def test_contract(actual):
    assert actual == 4
    assert True
"""
    removed_test = """
def helper(actual):
    assert actual == 4
    assert actual > 0
"""
    call_comparison = """
def test_contract(actual):
    assert generate() == generate()
    assert actual > 0
"""
    constant_number = """
def test_contract(actual):
    assert 1
    assert not False
"""

    assert _policy_violations(previous, changed_expectation) == []
    assert "weakened assertion" in _policy_violations(previous, removed)[0]
    assert any(
        "constant-true" in item for item in _policy_violations(previous, constant_true)
    )
    assert "removed test contract" in _policy_violations(previous, removed_test)[0]
    assert all(
        "constant-true" not in item
        for item in _policy_violations(previous, call_comparison)
    )
    assert any(
        "constant-true" in item
        for item in _policy_violations(previous, constant_number)
    )


def test_plain_assertion_policy_preserves_comparison_contracts() -> None:
    previous = """
def test_contract(actual, expected):
    assert actual == expected
"""
    changed_expected = """
def test_contract(actual, expected):
    assert actual == revised_expected
"""
    changed_api = """
def test_contract(result, expected):
    assert result.values == revised_expected
"""
    previous_api = """
def test_contract(result, expected):
    assert result.value == expected
"""
    strengthened = """
def test_contract(actual, expected):
    assert actual == expected and validate(actual)
"""
    weakened = """
def test_contract(actual, expected):
    assert actual is not None
"""
    weakened_operator = """
def test_contract(actual, expected):
    assert actual != expected
"""

    assert _policy_violations(previous, changed_expected) == []
    assert _policy_violations(previous_api, changed_api) == []
    assert _policy_violations(previous, strengthened) == []
    assert "weakened assertion" in _policy_violations(previous, weakened)[0]
    assert "weakened assertion" in _policy_violations(previous, weakened_operator)[0]


def test_plain_assertion_policy_preserves_numeric_bounds() -> None:
    previous = """
def test_contract(actual):
    assert actual > 0
    assert actual < 10
    assert 0 < actual
    assert 10 > actual
"""
    strengthened = """
def test_contract(actual):
    assert actual > 1
    assert actual < 9
    assert 1 < actual
    assert 9 > actual
"""
    weakened = """
def test_contract(actual):
    assert actual > -1000
    assert actual < 1000
    assert -1000 < actual
    assert 1000 > actual
"""

    assert _policy_violations(previous, strengthened) == []
    assert "weakened assertion" in _policy_violations(previous, weakened)[0]

    chained = """
def test_contract(actual):
    assert 0 < actual < 10
"""
    narrowed_chain = """
def test_contract(actual):
    assert 1 < actual < 9
"""
    expanded_chain = """
def test_contract(actual):
    assert -1000 < actual < 1000
"""
    removed_upper_bound = """
def test_contract(actual):
    assert actual > 0
"""

    assert _policy_violations(chained, narrowed_chain) == []
    assert "weakened assertion" in _policy_violations(chained, expanded_chain)[0]
    assert "weakened assertion" in _policy_violations(chained, removed_upper_bound)[0]


def test_plain_assertion_policy_preserves_literal_membership() -> None:
    previous = """
ALLOWED = {1, 2}

def test_contract(actual):
    assert actual in [1, 2]
    assert actual not in {3, 4}
    assert actual in ALLOWED
"""
    strengthened = """
ALLOWED = {1, 2}

def test_contract(actual):
    assert actual in [1]
    assert actual not in {3, 4, 5}
    assert actual in ALLOWED
"""
    weakened = """
ALLOWED = {1, 2, 3}

def test_contract(actual):
    assert actual in [1, 2, 3]
    assert actual not in {3}
    assert actual in ALLOWED
"""

    assert _policy_violations(previous, strengthened) == []
    assert "weakened assertion" in _policy_violations(previous, weakened)[0]


def test_numerical_policy_rejects_removal_tolerance_and_self_comparison() -> None:
    previous = """
import numpy as np

def test_contract(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(actual, expected)
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12, nan_ok=False)
"""
    strengthened = """
import numpy as np

def test_contract(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)
    np.testing.assert_array_equal(actual, expected)
    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-13, nan_ok=False)
"""
    weakened = """
import numpy as np

def test_contract(actual, expected):
    np.testing.assert_allclose(actual, actual, rtol=1e-2, atol=1e-3)
    assert actual == pytest.approx(actual, rel=1.0, abs=1.0, nan_ok=True)
"""

    assert _policy_violations(previous, strengthened) == []
    assert len(_policy_violations(previous, weakened)) == 3

    substituted = """
import numpy as np

def test_contract(actual, expected, other):
    np.testing.assert_allclose(other, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(actual, expected, rtol=1.0, atol=1.0)
    np.testing.assert_array_equal(other, expected)
    assert other == pytest.approx(expected, rel=1e-12, abs=1e-15)
    assert actual == pytest.approx(expected, rel=1.0, abs=1.0, nan_ok=True)
"""
    assert len(_policy_violations(previous, substituted)) == 3


def test_numerical_policy_allows_same_subject_additions_and_api_updates() -> None:
    previous = """
import numpy as np

def test_contract(result, expected):
    np.testing.assert_allclose(result.value, expected, rtol=1e-8)
    assert result.value == pytest.approx(expected, rel=1e-9)
"""
    added = """
import numpy as np

def test_contract(result, expected, reference):
    np.testing.assert_allclose(result.value, reference, rtol=1e-10)
    np.testing.assert_allclose(result.value, expected, rtol=1e-8)
    assert result.value == pytest.approx(reference, rel=1e-11)
    assert result.value == pytest.approx(expected, rel=1e-9)
"""
    changed_api = """
import numpy as np

def test_contract(result, expected):
    np.testing.assert_allclose(result.values, expected, rtol=1e-8)
    assert result.values == pytest.approx(expected, rel=1e-9)
"""

    assert _policy_violations(previous, added) == []
    assert _policy_violations(previous, changed_api) == []


def test_tolerance_policy_handles_positional_and_mapping_arguments() -> None:
    positional = """
import numpy as np

def test_contract(actual, expected):
    np.testing.assert_allclose(actual, expected, 1e-9, 1e-12, False)
    assert actual == pytest.approx(expected, 1e-9, 1e-12, False)
"""
    weakened_positional = """
import numpy as np

def test_contract(actual, expected):
    np.testing.assert_allclose(actual, expected, 1.0, 1.0, True)
    assert actual == pytest.approx(expected, 1.0, 1.0, True)
"""
    mapped = """
import numpy as np

APPROX_OPTIONS = {"rel": 1e-9, "abs": 1e-12, "nan_ok": False}

def test_contract(actual, expected):
    allclose_options = {"rtol": 1e-9, "atol": 1e-12, "equal_nan": False}
    np.testing.assert_allclose(actual, expected, **allclose_options)
    assert actual == pytest.approx(expected, **APPROX_OPTIONS)
"""
    weakened_mapping = """
import numpy as np

APPROX_OPTIONS = {"rel": 1.0, "abs": 1.0, "nan_ok": True}

def test_contract(actual, expected):
    allclose_options = {"rtol": 1.0, "atol": 1.0, "equal_nan": True}
    np.testing.assert_allclose(actual, expected, **allclose_options)
    assert actual == pytest.approx(expected, **APPROX_OPTIONS)
"""
    unknown = """
def test_contract(actual, expected, options):
    assert actual == pytest.approx(expected, **options)
"""
    changed_unknown = """
def test_contract(actual, expected, options, other_options):
    assert actual == pytest.approx(expected, **other_options)
"""
    mutated = """
OPTIONS = {}
OPTIONS["rel"] = 1e-9

def test_contract(actual, expected):
    assert actual == pytest.approx(expected, **OPTIONS)
"""
    weakened_mutation = """
OPTIONS = {}
OPTIONS["rel"] = 1.0

def test_contract(actual, expected):
    assert actual == pytest.approx(expected, **OPTIONS)
"""

    assert len(_policy_violations(positional, weakened_positional)) == 2
    assert len(_policy_violations(mapped, weakened_mapping)) == 2
    assert _policy_violations(unknown, unknown) == []
    assert "numerical assertion" in _policy_violations(unknown, changed_unknown)[0]
    assert "numerical assertion" in _policy_violations(mutated, weakened_mutation)[0]


def test_approx_policy_models_abs_only_and_explicit_defaults() -> None:
    abs_only = """
def test_contract(actual, expected):
    assert actual == pytest.approx(expected, abs=1e-6)
"""
    relative_added = """
def test_contract(actual, expected):
    assert actual == pytest.approx(expected, rel=1e-7, abs=1e-6)
"""
    default = """
def test_contract(actual, expected):
    assert actual == pytest.approx(expected)
"""
    explicit_none = """
def test_contract(actual, expected):
    assert actual == pytest.approx(expected, rel=None, abs=None)
"""

    assert "numerical assertion" in _policy_violations(abs_only, relative_added)[0]
    assert _policy_violations(default, explicit_none) == []


def test_skip_policy_resolves_import_aliases() -> None:
    source = """
import pytest as pt
from unittest import expectedFailure as known_failure, skip as disabled

@pt.mark.skip(reason="disabled")
def test_one():
    pass

@disabled("disabled")
def test_two():
    pt.importorskip("optional")

@known_failure
def test_three():
    pass
"""

    snapshot = _snapshot(source, namespace="tests/example.py")

    assert len(snapshot.skips) == 4


def test_skip_policy_resolves_local_import_aliases() -> None:
    source = """
def test_one():
    import pytest as pt
    pt.skip("disabled")
    pt = object()

def test_two():
    from pytest import xfail as expected_failure
    expected_failure("disabled")

def test_three():
    from unittest import expectedFailure as known_failure

    @known_failure
    def check():
        pass

import pytest as pt

def test_shadowed_parameter(pt):
    pt.skip("application method")

def test_unittest_case(self):
    self.skipTest("disabled")

def test_dynamic_alias():
    import pytest as pt
    getattr(pt, "skip")("disabled")
    getattr(pt, "xfail")("disabled")
"""

    snapshot = _snapshot(source, namespace="tests/example.py")

    assert len(snapshot.skips) == 6


def test_finding_comparison_rejects_new_issue() -> None:
    finding = Finding("ruff|F821|module.py|run|undefined", "module.py:1 F821")

    with pytest.raises(QualityFailure, match="New or worsened ruff findings"):
        _assert_findings_not_worse(
            "ruff",
            [finding],
            {"ruff": {"findings": {}}},
        )


def test_secret_baseline_update_preserves_review_decision() -> None:
    fingerprint = "fingerprint"
    finding = {"type": "checksum", "hashed_secret": fingerprint}
    current = {"results": {"tests/example.py": [finding.copy()]}}
    previous_finding = {**finding, "is_secret": False}
    previous = {"results": {"tests\\example.py": [previous_finding]}}

    merged = _preserve_secret_reviews(current, previous)

    assert merged["results"]["tests/example.py"][0]["is_secret"] is False


def test_secret_fingerprints_preserve_duplicate_counts() -> None:
    fingerprint_key = "hashed_" + chr(115) + "ecret"
    finding = {"type": "checksum", fingerprint_key: "fingerprint"}
    single = {"results": {"tests/example.py": [finding]}}
    duplicate = {"results": {"tests/example.py": [finding, finding]}}

    assert _secret_fingerprints(duplicate) - _secret_fingerprints(single) == {
        "tests/example.py|checksum|fingerprint": 1
    }


def test_secret_scan_includes_untracked_nonignored_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*arguments: str) -> str:
        calls.append(arguments)
        return "tracked.py\nuntracked.py\n.secrets.baseline\n"

    monkeypatch.setattr("tools.quality_gate.measurements._git", fake_git)

    assert _tracked_files() == ["tracked.py", "untracked.py"]
    assert calls == [("ls-files", "--cached", "--others", "--exclude-standard")]
