"""Adapters that normalize external-tool output into baseline data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from tools.quality_gate.context import (
    ROOT,
    SOURCE_PATHS,
    Finding,
    _capture,
    _finding_key,
    _git,
    _relative_path,
)


def _ruff_findings(paths: Sequence[str]) -> list[Finding]:
    output = _capture(
        ["ruff", "check", "--no-cache", "--output-format=json", *paths],
        allowed={0, 1},
    )
    payload = json.loads(output or "[]")
    return [
        _finding_key(
            "ruff",
            str(item.get("code") or "unknown"),
            item["filename"],
            int(item["location"]["row"]),
            item["message"],
        )
        for item in payload
    ]


def _bandit_findings() -> list[Finding]:
    output = _capture(
        ["bandit", "-q", "-r", *SOURCE_PATHS, "-f", "json"], allowed={0, 1}
    )
    payload = json.loads(output or "{}")
    return [
        _finding_key(
            "bandit",
            f"{item['test_id']}:{item['issue_severity']}",
            item["filename"],
            int(item["line_number"]),
            item["issue_text"],
        )
        for item in payload.get("results", [])
    ]


def _radon_complexity() -> dict[str, int]:
    output = _capture(["radon", "cc", "-j", *SOURCE_PATHS])
    payload = json.loads(output or "{}")
    complexity: dict[str, int] = {}
    for filename, blocks in payload.items():
        path = _relative_path(filename)
        for block in blocks:
            parent = block.get("classname")
            name = str(block["name"])
            qualified = f"{parent}.{name}" if parent else name
            complexity[f"{path}::{qualified}"] = int(block["complexity"])
            for method in block.get("methods", []):
                method_name = f"{name}.{method['name']}"
                complexity[f"{path}::{method_name}"] = int(method["complexity"])
    return dict(sorted(complexity.items()))


def _vulture_findings() -> list[str]:
    output = _capture(
        ["vulture", *SOURCE_PATHS, "--min-confidence", "80"],
        allowed={0, 1, 3},
    )
    return sorted(
        line.replace(str(ROOT), ".").strip()
        for line in output.splitlines()
        if line.strip()
    )


def _tracked_files() -> list[str]:
    excluded = {
        ".secrets.baseline",
        "quality/baseline.json",
        "quality/pyrefly-baseline.json",
    }
    return [
        path
        for path in _git(
            "ls-files", "--cached", "--others", "--exclude-standard"
        ).splitlines()
        if path and path not in excluded
    ]


def _secret_scan() -> dict[str, Any]:
    output = _capture(["detect-secrets", "scan", *_tracked_files()])
    return json.loads(output or "{}")


def _secret_key(filename: str, finding: dict[str, Any]) -> str:
    return "|".join(
        (
            filename.replace("\\", "/"),
            str(finding.get("type", "")),
            str(finding.get("hashed_secret", "")),
        )
    )


def _secret_fingerprints(payload: dict[str, Any]) -> Counter[str]:
    return Counter(
        _secret_key(filename, finding)
        for filename, findings in payload.get("results", {}).items()
        for finding in findings
    )


def _preserve_secret_reviews(
    current: dict[str, Any], previous: dict[str, Any]
) -> dict[str, Any]:
    reviews = {
        _secret_key(filename, finding): bool(finding["is_secret"])
        for filename, findings in previous.get("results", {}).items()
        for finding in findings
        if "is_secret" in finding
    }
    for filename, findings in current.get("results", {}).items():
        for finding in findings:
            key = _secret_key(filename, finding)
            if key in reviews:
                finding["is_secret"] = reviews[key]
    return current
