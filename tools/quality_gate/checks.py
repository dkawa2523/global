"""Static, test, security, and baseline checks used by quality commands."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

from tools.quality_gate.context import (
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

MIN_COMBINED_COVERAGE = 85.0


def _check_fatal_bandit(bandit: Sequence[Finding]) -> None:
    fatal = [item.display for item in bandit if ":HIGH|" in item.key]
    if fatal:
        _die(
            "High-severity Bandit findings:\n"
            + "\n".join(f"  {item}" for item in fatal)
        )


def _check_no_findings(tool: str, findings: Sequence[Finding]) -> None:
    if findings:
        _die(
            f"{tool} findings:\n" + "\n".join(f"  {item.display}" for item in findings)
        )


def _check_format(paths: Sequence[str]) -> None:
    _run(
        ["ruff", "format", "--config", str(ROOT / "pyproject.toml"), "--check", *paths]
    )


def _check_pyrefly() -> None:
    _run(
        [
            "pyrefly",
            "check",
            "--config",
            str(ROOT / "pyproject.toml"),
            *SOURCE_PATHS,
        ]
    )


def _check_imports() -> None:
    _run(["lint-imports", "--config", str(ROOT / "pyproject.toml")])


def _pip_audit() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    requirements = REPORT_DIR / "audit-requirements.txt"
    _run(
        [
            "uv",
            "export",
            "--quiet",
            "--frozen",
            "--all-extras",
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


def _pytest_with_coverage() -> float:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    coverage_xml = REPORT_DIR / "coverage.xml"
    coverage_json = REPORT_DIR / "coverage.json"
    output = _capture(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pyproject.toml"),
            "-o",
            "addopts=",
            "-m",
            "not nightly",
            "tests",
            "--cov=plasma_global",
            "--cov-branch",
            f"--cov-config={ROOT / 'pyproject.toml'}",
            f"--cov-report=xml:{coverage_xml}",
            f"--cov-report=json:{coverage_json}",
            "--cov-report=term",
        ],
        env={"HYPOTHESIS_PROFILE": "ci"},
    )
    print(f"\n> pytest --cov=plasma_global --cov-branch\n{output}", flush=True)
    payload = json.loads(coverage_json.read_text(encoding="utf-8"))
    coverage = float(payload["totals"]["percent_covered"])
    return coverage


def _check_combined_coverage(coverage: float) -> None:
    """Enforce coverage.py's combined statement-and-branch percentage."""

    if coverage + 1.0e-9 < MIN_COMBINED_COVERAGE:
        _die(
            f"Combined statement-and-branch coverage is below the fixed floor: "
            f"{coverage:.6f}% < {MIN_COMBINED_COVERAGE:.1f}%"
        )


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


def _check_complexity(current: dict[str, int], baseline: dict[str, Any]) -> None:
    allowed = {
        key: int(value)
        for key, value in baseline.get("complexity_exceptions", {}).items()
    }
    failures = [
        f"{symbol}: {score} > {allowed.get(symbol, 10)}"
        for symbol, score in current.items()
        if score > allowed.get(symbol, 10)
    ]
    if failures:
        _die("New or worsened cyclomatic complexity:\n  " + "\n  ".join(failures))


def _check_vulture(current: Sequence[str]) -> None:
    if current:
        _die("Dead-code candidates:\n  " + "\n  ".join(current))


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
