"""Reject newly added quality suppressions and disabled tests."""

from __future__ import annotations

import re
from collections.abc import Iterator

from tools.quality_gate.context import _base_ref, _die, _git

FORBIDDEN_PYTHON_ADDITION = re.compile(
    r"#\s*(?:(?:ruff|flake8):\s*)?noqa\b"
    r"|#\s*nosec\b"
    r"|#\s*type:\s*ignore(?:\[[^]]*\])?(?=\s|$)"
    r"|#\s*pyrefly:\s*ignore(?:-errors|\[[^]]*\])?(?=\s|$)"
    r"|#\s*pragma:\s*no\s+(?:cover|mutate)\b",
    re.IGNORECASE,
)
FORBIDDEN_SECRET_ADDITION = re.compile(
    r"#\s*pragma:\s*(?:allowlist|whitelist)\s+(?:nextline\s+)?secret\b",
    re.IGNORECASE,
)
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
FORBIDDEN_TEST_DISABLE_ADDITION = re.compile(
    r"\bpytestmark\s*="
    r"|\b(?:pytest|[A-Za-z_]\w*)\.mark\.(?:skip|skipif|xfail)\b"
    r"|\b(?:pytest|[A-Za-z_]\w*)\.(?:skip|xfail|importorskip)\s*\("
    r"|\bself\.skipTest\s*\("
    r"|\b(?:unittest\.)?(?:expectedFailure|skip|skipIf|skipUnless)\b"
    r"|^\s*from\s+(?:pytest|unittest)\s+import\b.*"
    r"\b(?:expectedFailure|importorskip|skip|skipIf|skipUnless|xfail)\b",
    re.IGNORECASE,
)


def _diff_additions(diff: str) -> Iterator[tuple[str, int, str]]:
    current_file = ""
    current_line = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current_file = ""
            in_hunk = False
            continue
        if line.startswith("+++ b/"):
            current_file = line[6:].replace("\\", "/")
            continue
        if matched := HUNK_HEADER.match(line):
            current_line = int(matched.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            yield current_file, current_line, line[1:]
            current_line += 1
        elif line.startswith("-") or line.startswith("\\"):
            continue
        else:
            current_line += 1


def _policy_diff_violations(diff: str) -> list[str]:
    return [
        addition.strip()
        for path, _, addition in _diff_additions(diff)
        if FORBIDDEN_SECRET_ADDITION.search(addition)
        or (path.endswith(".py") and FORBIDDEN_PYTHON_ADDITION.search(addition))
    ]


def _skip_diff_violations(diff: str) -> list[str]:
    """Reject direct test-disabling syntax on newly added test lines."""

    return [
        f"forbidden skip/xfail addition: {path}:{line}: {addition.strip()}"
        for path, line, addition in _diff_additions(diff)
        if path.startswith("tests/")
        and path.endswith(".py")
        and FORBIDDEN_TEST_DISABLE_ADDITION.search(addition)
    ]


def _check_policy_additions() -> None:
    base = _base_ref()
    if base is None:
        return
    diff = _git("diff", "--unified=0", base, "--")
    issues = [*_policy_diff_violations(diff), *_skip_diff_violations(diff)]
    if issues:
        _die(
            "Forbidden quality-policy additions:\n"
            + "\n".join(f"  {issue}" for issue in issues)
        )
