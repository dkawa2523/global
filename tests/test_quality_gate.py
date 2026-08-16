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
    _skip_diff_violations,
)


def _new_file_diff(path: str, source: str) -> str:
    lines = source.splitlines()
    additions = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{additions}\n"
    )


def test_public_quality_commands_remain_available() -> None:
    assert callable(quality.fast_main)
    assert callable(quality.pr_main)
    assert callable(quality.baseline_main)
    assert callable(quality.nightly_main)


@pytest.mark.parametrize(
    "directive",
    [
        "# " + "noqa",
        "# ruff: " + "noqa",
        "# no" + "sec",
        "# type: " + "ignore[assignment]",
        "# pyrefly: " + "ignore[bad-assignment]",
        "# pragma: no " + "cover",
        "# pragma: no " + "mutate",
    ],
)
def test_policy_diff_rejects_new_python_suppressions(directive: str) -> None:
    diff = _new_file_diff("plasma_global/example.py", f"value = 1  {directive}\n")

    assert _policy_diff_violations(diff) == [f"value = 1  {directive}"]


@pytest.mark.parametrize(
    "directive",
    [
        "# pragma: " + "allowlist secret",
        "# pragma: " + "allowlist nextline secret",
        "# pragma: " + "whitelist secret",
    ],
)
def test_policy_diff_rejects_secret_allowlists_in_any_file(directive: str) -> None:
    diff = _new_file_diff("docs/example.md", directive)

    assert _policy_diff_violations(diff) == [directive]


def test_policy_diff_ignores_context_and_removed_directives() -> None:
    marker = "#"
    diff = (
        "diff --git a/example.py b/example.py\n"
        "--- a/example.py\n"
        "+++ b/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        f"-old = 1  {marker} no" + "qa\n"
        f" unchanged = 2  {marker} no" + "sec\n"
        "+new = 3\n"
    )

    assert _policy_diff_violations(diff) == []


def test_skip_diff_rejects_direct_skip_sites() -> None:
    path = "tests/example.py"
    alias = "pt"
    skip_name = "sk" + "ip"
    xfail_name = "x" + "fail"
    source = f"""import pytest as {alias}

@{alias}.mark.{skip_name}(reason="disabled")
def test_one():
    pass

def test_two():
    {alias}.{xfail_name}("disabled")
"""

    violations = _skip_diff_violations(_new_file_diff(path, source))

    assert len(violations) == 2
    assert any(f"{alias}.mark.{skip_name}" in item for item in violations)
    assert any(f"{alias}.{xfail_name}" in item for item in violations)


@pytest.mark.parametrize(
    "source",
    [
        "pytest" + "mark = [pytest.mark." + "sk" + "ip]",
        "pytest.param(1, marks=pytest.mark." + "x" + 'fail(reason="known"))',
        "disabled = pytest.mark." + "sk" + "ip",
        "from pytest import " + "x" + "fail as expected_failure",
        "self.sk" + "ip" + 'Test("disabled")',
    ],
)
def test_skip_diff_rejects_supported_disabling_syntax(source: str) -> None:
    path = "tests/example.py"

    assert len(_skip_diff_violations(_new_file_diff(path, source))) == 1


def test_skip_diff_ignores_existing_skip_when_another_line_is_added() -> None:
    path = "tests/example.py"
    diff = f"""\
diff --git a/{path} b/{path}
--- a/{path}
+++ b/{path}
@@ -4,0 +5,1 @@
+VALUE = 1
"""

    assert _skip_diff_violations(diff) == []


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
