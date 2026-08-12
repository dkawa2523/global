"""Read, compare, and explicitly write legacy quality baselines."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.quality_gate.context import (
    BASELINE_PATH,
    PYREFLY_BASELINE_PATH,
    SECRETS_BASELINE_PATH,
    Finding,
    _base_ref,
    _die,
    _git,
    _relative_path,
    _source_for_path,
    _symbol_for_source,
)
from tools.quality_gate.measurements import _secret_fingerprints

INITIAL_MEASUREMENT = {
    "bandit_findings": 25,
    "branch_coverage_percent": 80.739627,
    "complex_functions": 47,
    "detect_secrets_findings": 1,
    "pip_audit_vulnerabilities": 0,
    "pyrefly_errors": 56,
    "pyrefly_warnings_and_errors": 266,
    "pytest_passed": 195,
    "ruff_findings_raw": 735,
    "ruff_test_assert_findings": 556,
    "vulture_findings": 0,
}


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.is_file():
        _die("quality baseline is missing; run `uv run quality-baseline` explicitly")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _allowed_findings(baseline: dict[str, Any], tool: str) -> Counter[str]:
    return Counter(
        {key: int(value) for key, value in baseline[tool]["findings"].items()}
    )


def _assert_findings_not_worse(
    tool: str,
    findings: Sequence[Finding],
    baseline: dict[str, Any],
) -> None:
    current = Counter(item.key for item in findings)
    excess = current - _allowed_findings(baseline, tool)
    if not excess:
        return
    display_by_key = {item.key: item.display for item in findings}
    details = "\n".join(
        f"  {count} x {display_by_key[key]}" for key, count in excess.items()
    )
    _die(f"New or worsened {tool} findings:\n{details}")


def _read_json_at_ref(relative_path: str, source_ref: str) -> dict[str, Any] | None:
    output = _git("show", f"{source_ref}:{relative_path}", allowed={0, 128})
    if not output.strip():
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        _die(f"Base branch has an invalid {relative_path}: {exc}")


def _read_base_json(relative_path: str) -> dict[str, Any] | None:
    base = _base_ref()
    return None if base is None else _read_json_at_ref(relative_path, base)


def _pyrefly_fingerprint(
    item: dict[str, Any],
    sources: dict[str, str | None],
    source_ref: str | None,
) -> str:
    path = _relative_path(str(item.get("path", "")))
    if path not in sources:
        sources[path] = _source_for_path(path, source_ref=source_ref)
    symbol = _symbol_for_source(sources[path], int(item.get("line", 0) or 0))
    message = str(item.get("concise_description", item.get("description", "")))
    return "|".join(
        (
            path,
            symbol,
            str(item.get("name", "")),
            str(item.get("severity", "")),
            " ".join(message.split()),
        )
    )


def _pyrefly_fingerprints(
    payload: dict[str, Any], *, source_ref: str | None = None
) -> Counter[str]:
    sources: dict[str, str | None] = {}
    return Counter(
        _pyrefly_fingerprint(item, sources, source_ref)
        for item in payload.get("errors", [])
    )


def _native_fingerprints(
    relative_path: str,
    payload: dict[str, Any],
    *,
    source_ref: str | None = None,
) -> Counter[str]:
    if relative_path == ".secrets.baseline":
        return _secret_fingerprints(payload)
    return _pyrefly_fingerprints(payload, source_ref=source_ref)


def _baseline_complexity_failures(
    current: dict[str, Any], previous: dict[str, Any]
) -> list[str]:
    failures = []
    for symbol, score in current["radon"]["complexity"].items():
        old_score = int(previous["radon"]["complexity"].get(symbol, 10))
        if score > old_score:
            failures.append(f"complexity baseline worsened: {symbol}")
    return failures


def _validate_baseline_monotonic(current: dict[str, Any]) -> None:
    previous = _read_base_json("quality/baseline.json")
    if previous is None:
        return
    failures = []
    for tool in ("ruff", "bandit"):
        old = Counter(previous[tool]["findings"])
        new = Counter(current[tool]["findings"])
        if new - old:
            failures.append(f"{tool} baseline contains new findings")
    failures.extend(_baseline_complexity_failures(current, previous))
    if float(current["coverage"]["branch_percent"]) < float(
        previous["coverage"]["branch_percent"]
    ):
        failures.append("branch coverage baseline was lowered")
    for name in ("passed", "assertions"):
        if int(current["tests"][name]) < int(previous["tests"][name]):
            failures.append(f"test {name} baseline was lowered")
    if Counter(current["vulture"]["findings"]) - Counter(
        previous["vulture"]["findings"]
    ):
        failures.append("vulture baseline contains new findings")
    if failures:
        _die("Baseline weakening is forbidden:\n  " + "\n  ".join(failures))


def _check_native_baseline_sizes() -> None:
    base = _base_ref()
    if base is None:
        return
    for relative_path, current_path in (
        ("quality/pyrefly-baseline.json", PYREFLY_BASELINE_PATH),
        (".secrets.baseline", SECRETS_BASELINE_PATH),
    ):
        excess = _native_baseline_excess(relative_path, current_path, base)
        if excess:
            _die(
                f"{relative_path} accepts findings that are absent from the base branch"
            )


def _native_baseline_excess(
    relative_path: str, current_path: Path, base: str
) -> Counter[str]:
    previous = _read_json_at_ref(relative_path, base) or {}
    current = json.loads(current_path.read_text(encoding="utf-8"))
    return _native_fingerprints(relative_path, current) - _native_fingerprints(
        relative_path, previous, source_ref=base
    )


def _baseline_payload(
    *,
    ruff: Sequence[Finding],
    bandit: Sequence[Finding],
    complexity: dict[str, int],
    vulture: Sequence[str],
    coverage: float,
    passed: int,
    assertions: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "initial_measurement": INITIAL_MEASUREMENT,
        "ruff": {"findings": dict(sorted(Counter(item.key for item in ruff).items()))},
        "bandit": {
            "findings": dict(sorted(Counter(item.key for item in bandit).items()))
        },
        "radon": {"complexity": complexity},
        "vulture": {"findings": list(vulture), "min_confidence": 80},
        "coverage": {
            "branch_percent": coverage,
            "changed_lines_percent": 90.0,
        },
        "tests": {"passed": passed, "assertions": assertions},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
