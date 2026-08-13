"""Reject explicit suppressions and narrow, structural test weakening."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from tools.quality_gate._test_snapshot import (
    AssertionSummary,
    FileSnapshot,
    RaisesSite,
    ToleranceSite,
    _snapshot,
)
from tools.quality_gate.context import ROOT, _base_ref, _die, _git

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


def _assertion_violations(
    previous: Sequence[AssertionSummary], current: Sequence[AssertionSummary]
) -> list[str]:
    old = _assertions_by_scope(previous)
    latest = _assertions_by_scope(current)
    return [
        f"removed or weakened assertion: {scope}"
        for scope, item in old.items()
        if latest.get(scope, AssertionSummary(scope, 0, 0)).count < item.count
    ] + [
        f"constant-true assertion added: {scope}"
        for scope, item in latest.items()
        if item.obviously_true
        > old.get(scope, AssertionSummary(scope, 0, 0)).obviously_true
    ]


def _assertions_by_scope(
    summaries: Sequence[AssertionSummary],
) -> dict[str, AssertionSummary]:
    return {item.scope: item for item in summaries}


def _raises_violations(
    previous: Sequence[RaisesSite], current: Sequence[RaisesSite]
) -> list[str]:
    losses = (
        (
            _raise_counts(previous, "all") - _raise_counts(current, "all"),
            "removed or changed pytest.raises",
        ),
        (
            _raise_counts(previous, "has_match") - _raise_counts(current, "has_match"),
            "removed pytest.raises match",
        ),
        (
            _raise_counts(current, "directly_raises")
            - _raise_counts(previous, "directly_raises"),
            "direct raise added under pytest.raises",
        ),
    )
    return [
        f"{label}: {scope}: {exception}"
        for missing, label in losses
        for scope, exception in missing.elements()
    ]


def _raise_counts(sites: Sequence[RaisesSite], kind: str) -> Counter[tuple[str, str]]:
    return Counter(
        (site.scope, site.exception)
        for site in sites
        if kind == "all" or getattr(site, kind)
    )


def _tolerance_violations(
    previous: Sequence[ToleranceSite], current: Sequence[ToleranceSite]
) -> list[str]:
    old: dict[tuple[str, str], list[ToleranceSite]] = defaultdict(list)
    latest: dict[tuple[str, str], list[ToleranceSite]] = defaultdict(list)
    for site in previous:
        old[site.scope, site.function].append(site)
    for site in current:
        latest[site.scope, site.function].append(site)
    result = _relaxed_tolerances(old, latest)
    old_self = Counter(
        (site.scope, site.function) for site in previous if site.self_fulfilling
    )
    new_self = Counter(
        (site.scope, site.function) for site in current if site.self_fulfilling
    )
    result.extend(
        f"self-fulfilling numerical assertion: {scope}: {function}"
        for (scope, function), count in new_self.items()
        if count > old_self[scope, function]
    )
    return result


def _relaxed_tolerances(
    old: dict[tuple[str, str], list[ToleranceSite]],
    latest: dict[tuple[str, str], list[ToleranceSite]],
) -> list[str]:
    result: list[str] = []
    for key, sites in old.items():
        candidates = latest[key]
        if len(candidates) < len(sites):
            result.append(f"removed numerical assertion: {key[0]}: {key[1]}")
            continue
        for site, candidate in zip(sites, candidates, strict=False):
            if (
                candidate.relative > site.relative
                or candidate.absolute > site.absolute
                or (not site.nan_allowed and candidate.nan_allowed)
            ):
                result.append(
                    f"relaxed numerical assertion: {site.scope}[{site.ordinal}]"
                )
    return result


def _snapshot_violations(previous: FileSnapshot, current: FileSnapshot) -> list[str]:
    result = [
        f"removed test contract: {name}"
        for name in previous.tests
        if name not in current.tests
    ]
    result.extend(_assertion_violations(previous.assertions, current.assertions))
    result.extend(_raises_violations(previous.raises, current.raises))
    result.extend(_tolerance_violations(previous.tolerances, current.tolerances))
    return result


def _policy_diff_violations(diff: str) -> list[str]:
    result: list[str] = []
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].replace("\\", "/")
        elif line.startswith("+") and not line.startswith("+++"):
            addition = line[1:]
            if FORBIDDEN_SECRET_ADDITION.search(addition) or (
                current_file.endswith(".py")
                and FORBIDDEN_PYTHON_ADDITION.search(addition)
            ):
                result.append(addition.strip())
    return result


def _test_sources(base: str | None = None) -> dict[str, str]:
    if base:
        paths = _git("ls-tree", "-r", "--name-only", base, "--", "tests").splitlines()
        return {
            path: _git("show", f"{base}:{path}")
            for path in paths
            if path.endswith(".py")
        }
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "tests").glob("**/*.py"))
    }


def _snapshots(sources: dict[str, str]) -> dict[str, FileSnapshot]:
    return {path: _snapshot(source, namespace=path) for path, source in sources.items()}


def _repository_violations(base: str, current: dict[str, FileSnapshot]) -> list[str]:
    empty = FileSnapshot((), (), (), (), ())
    previous = _snapshots(_test_sources(base))
    return [
        issue
        for path, snapshot in previous.items()
        for issue in _snapshot_violations(snapshot, current.get(path, empty))
    ]


def _skip_violations(snapshots: dict[str, FileSnapshot]) -> list[str]:
    return [issue for snapshot in snapshots.values() for issue in snapshot.skips]


def _check_policy_additions() -> None:
    base = _base_ref()
    if base is None:
        return
    current = _snapshots(_test_sources())
    diff = _git("diff", "--unified=0", base, "--")
    issues = [
        *_policy_diff_violations(diff),
        *_repository_violations(base, current),
        *_skip_violations(current),
    ]
    if issues:
        _die(
            "Forbidden quality-policy weakening:\n"
            + "\n".join(f"  {issue}" for issue in issues)
        )
