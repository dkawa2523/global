"""Orchestration for the four public quality commands."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
from collections.abc import Callable

from tools.quality_gate.baselines import (
    _assert_findings_not_worse,
    _baseline_payload,
    _check_native_baseline_sizes,
    _load_baseline,
    _validate_baseline_monotonic,
    _write_json,
)
from tools.quality_gate.checks import (
    _check_combined_coverage,
    _check_complexity,
    _check_diff_coverage,
    _check_fatal_bandit,
    _check_format,
    _check_imports,
    _check_no_findings,
    _check_pyrefly,
    _check_secrets,
    _check_vulture,
    _pip_audit,
    _pytest_with_coverage,
)
from tools.quality_gate.config_contract import _check_quality_config
from tools.quality_gate.context import (
    ALL_PYTHON_PATHS,
    BASELINE_PATH,
    QUALITY_DIR,
    REPORT_DIR,
    ROOT,
    SECRETS_BASELINE_PATH,
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
from tools.quality_gate.test_policy import _check_policy_additions


def _run_fast() -> None:
    changed = _changed_python_files()
    paths = changed or list(ALL_PYTHON_PATHS)
    print(f"Checking {len(paths)} changed Python paths/files.")
    _check_format(paths)
    ruff = _ruff_findings(paths)
    _check_no_findings("Ruff", ruff)
    _check_pyrefly()
    _run(
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
        ],
        env={"HYPOTHESIS_PROFILE": "ci"},
    )


def _run_pr() -> None:
    _check_quality_config()
    baseline = _load_baseline()
    _validate_baseline_monotonic(baseline)
    _check_native_baseline_sizes()
    _check_format(ALL_PYTHON_PATHS)
    ruff = _ruff_findings(ALL_PYTHON_PATHS)
    bandit = _bandit_findings()
    _check_fatal_bandit(bandit)
    _check_no_findings("Ruff", ruff)
    _assert_findings_not_worse("bandit", bandit, baseline)
    _check_pyrefly()
    _check_imports()
    coverage = _pytest_with_coverage()
    _check_combined_coverage(coverage)
    _check_diff_coverage()
    _check_complexity(_radon_complexity(), baseline)
    _check_vulture(_vulture_findings())
    _pip_audit()
    _check_secrets()
    _check_policy_additions()


def _update_baseline() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    _check_format(ALL_PYTHON_PATHS)
    ruff = _ruff_findings(ALL_PYTHON_PATHS)
    bandit = _bandit_findings()
    _check_fatal_bandit(bandit)
    _check_no_findings("Ruff", ruff)
    _check_imports()
    _pip_audit()
    coverage = _pytest_with_coverage()
    _check_combined_coverage(coverage)
    complexity = _radon_complexity()
    _check_vulture(_vulture_findings())
    payload = _baseline_payload(
        bandit=bandit,
        complexity=complexity,
    )
    secret_payload = _secret_scan()
    if SECRETS_BASELINE_PATH.is_file():
        previous_secrets = json.loads(SECRETS_BASELINE_PATH.read_text(encoding="utf-8"))
        secret_payload = _preserve_secret_reviews(secret_payload, previous_secrets)
    _validate_baseline_monotonic(payload)
    _write_json(BASELINE_PATH, payload)
    _write_json(SECRETS_BASELINE_PATH, secret_payload)
    print("Updated quality/baseline.json and .secrets.baseline explicitly.")


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
                sys.executable,
                "-m",
                "pytest",
                "-c",
                str(ROOT / "pyproject.toml"),
                "-q",
                "-o",
                "addopts=",
                "-m",
                "property",
                f"--hypothesis-seed={seed}",
                "tests",
            ],
            env={"HYPOTHESIS_PROFILE": "nightly"},
        )
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROOT / "pyproject.toml"),
            "-q",
            "-o",
            "addopts=",
            "-m",
            "nightly",
            "tests",
        ]
    )
    _run_mutation_gate()


def _delegate_nightly_to_wsl() -> None:
    if shutil.which("wsl.exe") is None:
        _die(
            "mutmut requires fork support. Install WSL, or run "
            "`python -m tools.quality nightly --native` on Linux."
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
        f"{wsl_environment} uv run --frozen python -m tools.quality nightly --native"
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
    """Explicitly replace the tracked exception files after validation."""

    _entrypoint(_update_baseline)


def nightly_main() -> None:
    """Run Linux-only stress, seeded property, and mutation checks."""

    native = "--native" in sys.argv[1:]
    if os.name == "nt" and not native:
        _entrypoint(_delegate_nightly_to_wsl)
        return
    if os.name == "nt":
        _die("quality nightly --native is only supported on Linux/WSL")
    _entrypoint(_run_nightly_native)
