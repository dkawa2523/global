"""Read and update the small set of reviewed quality exceptions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.quality_gate.context import (
    BASELINE_PATH,
    SECRETS_BASELINE_PATH,
    Finding,
    _base_ref,
    _die,
    _git,
)
from tools.quality_gate.measurements import _secret_fingerprints


def _load_baseline() -> dict[str, Any]:
    if not BASELINE_PATH.is_file():
        _die(
            "quality baseline is missing; run "
            "`python -m tools.quality baseline` explicitly"
        )
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


def _complexity_exceptions(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("complexity_exceptions")
    if raw is None:
        raw = payload.get("radon", {}).get("complexity", {})
    return {str(key): int(value) for key, value in raw.items() if int(value) > 10}


def _validate_baseline_monotonic(current: dict[str, Any]) -> None:
    previous = _read_base_json("quality/baseline.json")
    if previous is None:
        return
    failures = []
    old_bandit = Counter(previous.get("bandit", {}).get("findings", {}))
    new_bandit = Counter(current.get("bandit", {}).get("findings", {}))
    if new_bandit - old_bandit:
        failures.append("bandit baseline contains new findings")
    old_complexity = _complexity_exceptions(previous)
    for symbol, score in _complexity_exceptions(current).items():
        if score > old_complexity.get(symbol, 10):
            failures.append(f"complexity exception was added or worsened: {symbol}")
    if failures:
        _die("Baseline weakening is forbidden:\n  " + "\n  ".join(failures))


def _check_native_baseline_sizes() -> None:
    base = _base_ref()
    if base is None:
        return
    previous = _read_json_at_ref(".secrets.baseline", base) or {}
    current = json.loads(SECRETS_BASELINE_PATH.read_text(encoding="utf-8"))
    if _secret_fingerprints(current) - _secret_fingerprints(previous):
        _die(".secrets.baseline accepts findings absent from the base branch")


def _baseline_payload(
    *,
    bandit: Sequence[Finding],
    complexity: dict[str, int],
) -> dict[str, Any]:
    return {
        "bandit": {
            "findings": dict(sorted(Counter(item.key for item in bandit).items()))
        },
        "complexity_exceptions": {
            symbol: score for symbol, score in sorted(complexity.items()) if score > 10
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
