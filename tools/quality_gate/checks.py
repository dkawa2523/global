"""Independent policy checks used by the fast and pull-request gates."""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.quality_gate.context import (
    PYREFLY_BASELINE_PATH,
    REPORT_DIR,
    ROOT,
    SECRETS_BASELINE_PATH,
    SOURCE_PATHS,
    Finding,
    _base_ref,
    _capture,
    _die,
    _git,
    _run,
)
from tools.quality_gate.measurements import _secret_fingerprints, _secret_scan

FATAL_RUFF_CODES = {"F821", "F822", "F823"}
FORBIDDEN_ADDITION = re.compile(
    r"(?:#\s*(?:noqa|nosec|type:\s*ignore|pragma:\s*no\s+(?:cover|mutate))"
    r"|pytest\.(?:mark\.)?(?:skip|xfail)\b"
    r"|@pytest\.mark\.(?:skip|xfail)\b)"
)
FORBIDDEN_TEST_REMOVAL = re.compile(
    r"(?:\bassert\b|\bpytest\.raises\b|\b(?:async\s+)?def\s+test_)"
)


def _check_fatal_findings(ruff: Sequence[Finding], bandit: Sequence[Finding]) -> None:
    fatal = [
        item.display
        for item in ruff
        if any(f"|{code}|" in item.key for code in FATAL_RUFF_CODES)
        or "|E9" in item.key
    ]
    fatal.extend(item.display for item in bandit if ":HIGH|" in item.key)
    if fatal:
        _die("Non-baselinable findings:\n" + "\n".join(f"  {item}" for item in fatal))


def _check_format(paths: Sequence[str]) -> None:
    _run(["ruff", "format", "--check", *paths])


def _check_pyrefly(baseline_path: Path = PYREFLY_BASELINE_PATH) -> None:
    _run(
        [
            "pyrefly",
            "check",
            f"--baseline={baseline_path}",
            *SOURCE_PATHS,
        ]
    )


def _check_imports() -> None:
    _run(["lint-imports"])


def _pip_audit() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    requirements = REPORT_DIR / "audit-requirements.txt"
    _run(
        [
            "uv",
            "export",
            "--quiet",
            "--frozen",
            "--extra",
            "dev",
            "--no-emit-project",
            "--no-hashes",
            "--output-file",
            str(requirements),
        ]
    )
    _run(
        [
            "pip-audit",
            "--requirement",
            str(requirements),
            "--no-deps",
            "--disable-pip",
        ]
    )


def _pytest_with_coverage() -> tuple[float, int]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_xml = REPORT_DIR / "coverage.xml"
    coverage_json = REPORT_DIR / "coverage.json"
    output = _capture(
        [
            "pytest",
            "--cov=plasma_global",
            "--cov-branch",
            f"--cov-report=xml:{coverage_xml}",
            f"--cov-report=json:{coverage_json}",
            "--cov-report=term",
        ],
        env={"HYPOTHESIS_PROFILE": "ci"},
    )
    print(f"\n> pytest --cov=plasma_global --cov-branch\n{output}", flush=True)
    payload = json.loads(coverage_json.read_text(encoding="utf-8"))
    coverage = float(payload["totals"]["percent_covered"])
    matched = re.search(r"(\d+) passed", output)
    if matched is None:
        _die("Could not determine the pytest pass count")
    return coverage, int(matched.group(1))


def _count_assertions() -> int:
    count = 0
    for path in (ROOT / "tests").glob("**/*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        count += sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    return count


def _check_diff_coverage() -> None:
    base = _base_ref()
    if base is None:
        print("No Git base is available; diff coverage is not applicable.")
        return
    changed_source = _git(
        "diff", "--name-only", "--diff-filter=ACMR", base, "--", "plasma_global/*.py"
    ).strip()
    if not changed_source:
        print("No changed plasma_global files; diff coverage is not applicable.")
        return
    _run(
        [
            "diff-cover",
            str(REPORT_DIR / "coverage.xml"),
            f"--compare-branch={base}",
            "--fail-under=90",
        ]
    )


def _deleted_tests(base: str) -> list[str]:
    return [
        line
        for line in _git("diff", "--name-status", base, "--", "tests").splitlines()
        if line.startswith("D\t")
    ]


def _removed_test_contract(current_file: str, line: str) -> bool:
    return (
        current_file.startswith("tests/")
        and line.startswith("-")
        and not line.startswith("---")
        and FORBIDDEN_TEST_REMOVAL.search(line[1:]) is not None
    )


def _policy_diff_violations(diff: str) -> list[str]:
    violations = []
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].replace("\\", "/")
        elif line.startswith("+") and not line.startswith("+++"):
            if FORBIDDEN_ADDITION.search(line[1:]):
                violations.append(line[1:].strip())
        elif _removed_test_contract(current_file, line):
            violations.append(f"removed test contract: {line[1:].strip()}")
    return violations


def _check_policy_additions() -> None:
    base = _base_ref()
    if base is None:
        return
    deleted_tests = _deleted_tests(base)
    if deleted_tests:
        _die("Deleting tests is forbidden:\n  " + "\n  ".join(deleted_tests))
    violations = _policy_diff_violations(
        _git("diff", "--unified=0", base, "--", "*.py")
    )
    if violations:
        _die(
            "Forbidden quality suppressions were added:\n"
            + "\n".join(f"  {line}" for line in violations)
        )


def _check_complexity(current: dict[str, int], baseline: dict[str, Any]) -> None:
    allowed = {
        key: int(value) for key, value in baseline["radon"]["complexity"].items()
    }
    failures = [
        f"{symbol}: {score} > {allowed.get(symbol, 10)}"
        for symbol, score in current.items()
        if score > allowed.get(symbol, 10)
    ]
    if failures:
        _die("New or worsened cyclomatic complexity:\n  " + "\n  ".join(failures))


def _check_vulture(current: Sequence[str], baseline: dict[str, Any]) -> None:
    excess = Counter(current) - Counter(baseline["vulture"]["findings"])
    if excess:
        _die(
            "New dead-code candidates:\n  "
            + "\n  ".join(f"{count} x {item}" for item, count in excess.items())
        )


def _check_secrets() -> None:
    if not SECRETS_BASELINE_PATH.is_file():
        _die(".secrets.baseline is missing")
    current = _secret_fingerprints(_secret_scan())
    baseline = _secret_fingerprints(
        json.loads(SECRETS_BASELINE_PATH.read_text(encoding="utf-8"))
    )
    extra = current - baseline
    if extra:
        _die("New secret candidates:\n  " + "\n  ".join(sorted(extra)))
