"""Orchestration for the four public quality commands."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from collections.abc import Callable
from typing import Any

from tools.quality_gate.baselines import (
    _assert_findings_not_worse,
    _baseline_payload,
    _check_native_baseline_sizes,
    _load_baseline,
    _validate_baseline_monotonic,
    _write_json,
)
from tools.quality_gate.checks import (
    _check_complexity,
    _check_diff_coverage,
    _check_fatal_findings,
    _check_format,
    _check_imports,
    _check_policy_additions,
    _check_pyrefly,
    _check_secrets,
    _check_vulture,
    _count_assertions,
    _pip_audit,
    _pytest_with_coverage,
)
from tools.quality_gate.context import (
    ALL_PYTHON_PATHS,
    BASELINE_PATH,
    PYREFLY_BASELINE_PATH,
    QUALITY_DIR,
    REPORT_DIR,
    ROOT,
    SECRETS_BASELINE_PATH,
    SOURCE_PATHS,
    QualityFailure,
    _capture,
    _changed_python_files,
    _die,
    _run,
)
from tools.quality_gate.measurements import (
    _bandit_findings,
    _preserve_secret_reviews,
    _radon_complexity,
    _ruff_findings,
    _secret_scan,
    _vulture_findings,
)


def _run_fast() -> None:
    baseline = _load_baseline()
    changed = _changed_python_files()
    paths = changed or list(ALL_PYTHON_PATHS)
    print(f"Checking {len(paths)} changed Python paths/files.")
    _check_format(paths)
    ruff = _ruff_findings(paths)
    _check_fatal_findings(ruff, [])
    _assert_findings_not_worse("ruff", ruff, baseline)
    _check_pyrefly()
    _run(["pytest"], env={"HYPOTHESIS_PROFILE": "ci"})


def _check_test_regression(
    coverage: float, passed: int, baseline: dict[str, Any]
) -> None:
    baseline_coverage = float(baseline["coverage"]["branch_percent"])
    if coverage + 1.0e-9 < baseline_coverage:
        _die(f"Branch coverage regressed: {coverage:.6f}% < {baseline_coverage:.6f}%")
    baseline_tests = baseline["tests"]
    if passed < int(baseline_tests["passed"]):
        _die(f"Passed-test count regressed: {passed} < {baseline_tests['passed']}")
    assertions = _count_assertions()
    if assertions < int(baseline_tests["assertions"]):
        _die(
            f"Assertion count regressed: {assertions} < {baseline_tests['assertions']}"
        )


def _run_pr() -> None:
    baseline = _load_baseline()
    _validate_baseline_monotonic(baseline)
    _check_native_baseline_sizes()
    _check_format(ALL_PYTHON_PATHS)
    ruff = _ruff_findings(ALL_PYTHON_PATHS)
    bandit = _bandit_findings()
    _check_fatal_findings(ruff, bandit)
    _assert_findings_not_worse("ruff", ruff, baseline)
    _assert_findings_not_worse("bandit", bandit, baseline)
    _check_pyrefly()
    _check_imports()
    coverage, passed = _pytest_with_coverage()
    _check_test_regression(coverage, passed, baseline)
    _check_diff_coverage()
    _check_complexity(_radon_complexity(), baseline)
    _check_vulture(_vulture_findings(), baseline)
    _pip_audit()
    _check_secrets()
    _check_policy_additions()


def _updated_pyrefly_baseline() -> str:
    temporary = REPORT_DIR / "pyrefly-baseline.json"
    if temporary.exists():
        temporary.unlink()
    _run(
        [
            "pyrefly",
            "check",
            f"--baseline={temporary}",
            "--update-baseline",
            *SOURCE_PATHS,
        ],
        # A newly written baseline produces exit 1 when diagnostics are present.
        allowed={0, 1},
    )
    if not temporary.is_file():
        _die("Pyrefly did not create its native baseline")
    return str(temporary)


def _update_baseline() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    _check_format(ALL_PYTHON_PATHS)
    ruff = _ruff_findings(ALL_PYTHON_PATHS)
    bandit = _bandit_findings()
    _check_fatal_findings(ruff, bandit)
    _check_imports()
    _pip_audit()
    coverage, passed = _pytest_with_coverage()
    payload = _baseline_payload(
        ruff=ruff,
        bandit=bandit,
        complexity=_radon_complexity(),
        vulture=_vulture_findings(),
        coverage=coverage,
        passed=passed,
        assertions=_count_assertions(),
    )
    secret_payload = _secret_scan()
    if SECRETS_BASELINE_PATH.is_file():
        previous_secrets = json.loads(SECRETS_BASELINE_PATH.read_text(encoding="utf-8"))
        secret_payload = _preserve_secret_reviews(secret_payload, previous_secrets)
    temporary_pyrefly = _updated_pyrefly_baseline()
    _validate_baseline_monotonic(payload)
    _write_json(BASELINE_PATH, payload)
    _write_json(SECRETS_BASELINE_PATH, secret_payload)
    shutil.copyfile(temporary_pyrefly, PYREFLY_BASELINE_PATH)
    print(
        "Updated quality/baseline.json, quality/pyrefly-baseline.json, "
        "and .secrets.baseline explicitly."
    )


def _run_mutation_gate() -> None:
    # Preload NumPy because extension modules cannot be reloaded after mutmut's
    # coverage pass unloads ordinary Python modules.
    invocation = [
        sys.executable,
        "-c",
        "import numpy; from mutmut.__main__ import cli; cli()",
    ]
    _run([*invocation, "run"])
    output = _capture([*invocation, "results"])
    if output.strip():
        print(output, end="" if output.endswith("\n") else "\n")
    unacceptable = re.compile(
        r": (?:survived|no tests|timeout|suspicious|not checked|segfault|"
        r"check was interrupted by user)$",
        re.MULTILINE,
    )
    if unacceptable.search(output):
        _die("Mutation gate has surviving, untested, or incomplete mutants")


def _run_nightly_native() -> None:
    _run_pr()
    for seed in (0, 1, 2):
        _run(
            [
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-m",
                "property",
                f"--hypothesis-seed={seed}",
            ],
            env={"HYPOTHESIS_PROFILE": "nightly"},
        )
    _run(["pytest", "-q", "-o", "addopts=", "-m", "nightly"])
    _run_mutation_gate()


def _delegate_nightly_to_wsl() -> None:
    if shutil.which("wsl.exe") is None:
        _die(
            "mutmut requires fork support. Install WSL, or run quality-nightly "
            "on the Linux GitHub Actions job."
        )
    linux_root = _capture(
        ["wsl.exe", "bash", "-lc", f"wslpath -a {shlex.quote(str(ROOT))}"]
    ).strip()
    if not linux_root:
        _die("WSL could not translate the repository path")
    wsl_environment = (
        'UV_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/plasma-global/uv" '
        'UV_PROJECT_ENVIRONMENT="${XDG_CACHE_HOME:-$HOME/.cache}/plasma-global/venv"'
    )
    command = (
        f"cd {shlex.quote(linux_root)} && "
        f"{wsl_environment} uv sync --frozen --extra dev && "
        f"{wsl_environment} uv run --frozen quality-nightly --native"
    )
    _run(["wsl.exe", "bash", "-lc", command])


def _entrypoint(operation: Callable[[], None]) -> None:
    try:
        operation()
    except QualityFailure as exc:
        print(f"\nQUALITY FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def fast_main() -> None:
    """Run the changed-file developer gate."""

    _entrypoint(_run_fast)


def pr_main() -> None:
    """Run the complete pull-request gate."""

    _entrypoint(_run_pr)


def baseline_main() -> None:
    """Explicitly replace the tracked legacy baselines after validation."""

    _entrypoint(_update_baseline)


def nightly_main() -> None:
    """Run Linux-only stress, seeded property, and mutation checks."""

    native = "--native" in sys.argv[1:]
    if os.name == "nt" and not native:
        _entrypoint(_delegate_nightly_to_wsl)
        return
    if os.name == "nt":
        _die("--native quality-nightly is only supported on Linux/WSL")
    _entrypoint(_run_nightly_native)
