"""Reject explicit quality-policy weakening while allowing test evolution."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

from tools.quality_gate._test_snapshot import (
    EMPTY_SNAPSHOT,
    AssertionSummary,
    ComparisonContract,
    FileSnapshot,
    RaisesSite,
    Tolerance,
    ToleranceSite,
    _snapshot,
)
from tools.quality_gate.context import ROOT, _base_ref, _die, _git

FORBIDDEN_PYTHON_ADDITION = re.compile(
    r"(?:#\s*(?:(?:(?:ruff|flake8):\s*)?noqa|nosec|type:\s*ignore"
    r"|pyrefly:\s*ignore(?:-errors|\[[^]]*\])?"
    r"|pragma:\s*no\s+(?:branch|cover|mutate)))",
    re.IGNORECASE,
)
FORBIDDEN_SECRET_ADDITION = re.compile(
    r"#\s*pragma:\s*(?:allowlist|whitelist)\s+(?:nextline\s+)?secret",
    re.IGNORECASE,
)
ESCAPED_REGEX_META = re.compile(r"\\([.^$*+?{}\[\]|()\\])")

SiteT = TypeVar("SiteT")


def _raises_preserved(previous: RaisesSite, current: RaisesSite) -> bool:
    if previous.exception != current.exception:
        return False
    if not previous.directly_raises and current.directly_raises:
        return False
    if previous.match is None or previous.match == current.match:
        return True
    return (
        previous.literal_match is not None
        and current.literal_match is not None
        and ESCAPED_REGEX_META.sub(r"\1", current.literal_match)
        == previous.literal_match
    )


def _augment_match(
    old_index: int,
    candidates: dict[int, list[int]],
    owners: dict[int, int],
    visited: set[int],
) -> bool:
    for new_index in candidates[old_index]:
        if new_index in visited:
            continue
        visited.add(new_index)
        owner = owners.get(new_index)
        if owner is None or _augment_match(owner, candidates, owners, visited):
            owners[new_index] = old_index
            return True
    return False


def _maximum_matches(
    previous: Sequence[SiteT],
    current: Sequence[SiteT],
    old_indices: Sequence[int],
    new_indices: Sequence[int],
    preserved: Callable[[SiteT, SiteT], bool],
) -> dict[int, int]:
    candidates = {
        old_index: [
            new_index
            for new_index in new_indices
            if preserved(previous[old_index], current[new_index])
        ]
        for old_index in old_indices
    }
    owners: dict[int, int] = {}
    ordered = sorted(
        (len(options), old_index) for old_index, options in candidates.items()
    )
    for _, old_index in ordered:
        _augment_match(old_index, candidates, owners, set())
    return {old_index: new_index for new_index, old_index in owners.items()}


def _indices_by_key(
    sites: Sequence[SiteT],
    indices: set[int],
    key: Callable[[SiteT], Hashable],
) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = {}
    for index in sorted(indices):
        groups.setdefault(key(sites[index]), []).append(index)
    return groups


def _match_stage(
    previous: Sequence[SiteT],
    current: Sequence[SiteT],
    old_remaining: set[int],
    new_remaining: set[int],
    missing: list[SiteT],
    key: Callable[[SiteT], Hashable],
    preserved: Callable[[SiteT, SiteT], bool],
) -> None:
    old_groups = _indices_by_key(previous, old_remaining, key)
    new_groups = _indices_by_key(current, new_remaining, key)
    for group_key, old_indices in old_groups.items():
        new_indices = new_groups.get(group_key, [])
        if not new_indices:
            continue
        matched = _maximum_matches(
            previous, current, old_indices, new_indices, preserved
        )
        old_remaining.difference_update(matched)
        new_remaining.difference_update(matched.values())
        unmatched_old = [item for item in old_indices if item in old_remaining]
        unmatched_new = [item for item in new_indices if item in new_remaining]
        failed_count = min(len(unmatched_old), len(unmatched_new))
        failed = unmatched_old[:failed_count]
        missing.extend(previous[item] for item in failed)
        old_remaining.difference_update(failed)
        new_remaining.difference_update(new_indices)


def _missing_sites(
    previous: Sequence[SiteT],
    current: Sequence[SiteT],
    keys: Sequence[Callable[[SiteT], Hashable]],
    preserved: Callable[[SiteT, SiteT], bool],
) -> list[SiteT]:
    old_remaining = set(range(len(previous)))
    new_remaining = set(range(len(current)))
    missing: list[SiteT] = []
    for key in keys:
        _match_stage(
            previous,
            current,
            old_remaining,
            new_remaining,
            missing,
            key,
            preserved,
        )
    missing.extend(previous[item] for item in sorted(old_remaining))
    return missing


def _missing_raises(
    previous: Sequence[RaisesSite], current: Sequence[RaisesSite]
) -> list[RaisesSite]:
    return _missing_sites(
        previous,
        current,
        (
            lambda site: (site.scope, site.guarded_target),
            lambda site: (site.scope, site.guarded_call),
            lambda site: site.scope,
        ),
        _raises_preserved,
    )


def _tolerance_preserved(previous: Tolerance, current: Tolerance) -> bool:
    if previous.numeric is not None and current.numeric is not None:
        return current.numeric <= previous.numeric
    return previous.fingerprint == current.fingerprint


def _tolerance_site_preserved(previous: ToleranceSite, current: ToleranceSite) -> bool:
    if (
        previous.function != current.function
        or previous.dynamic_arguments != current.dynamic_arguments
        or (not previous.self_fulfilling and current.self_fulfilling)
    ):
        return False
    if not _tolerance_preserved(previous.relative, current.relative):
        return False
    if not _tolerance_preserved(previous.absolute, current.absolute):
        return False
    return _nan_policy_preserved(previous, current)


def _nan_policy_preserved(previous: ToleranceSite, current: ToleranceSite) -> bool:
    if previous.nan_allowed is False and current.nan_allowed is True:
        return False
    if previous.nan_allowed is None or current.nan_allowed is None:
        return previous.nan_fingerprint == current.nan_fingerprint
    return True


def _missing_tolerances(
    previous: Sequence[ToleranceSite], current: Sequence[ToleranceSite]
) -> list[ToleranceSite]:
    return _missing_sites(
        previous,
        current,
        (
            lambda site: (
                site.scope,
                site.function,
                site.subject,
                site.reference,
            ),
            lambda site: (site.scope, site.function, site.subject),
        ),
        _tolerance_site_preserved,
    )


def _comparisons_preserved(
    previous: Sequence[ComparisonContract], current: Sequence[ComparisonContract]
) -> bool:
    missing = _missing_sites(
        previous,
        current,
        (lambda item: (item.subject, item.operator),),
        _comparison_preserved,
    )
    return not missing


def _comparison_preserved(
    previous: ComparisonContract, current: ComparisonContract
) -> bool:
    if previous.subject != current.subject or previous.operator != current.operator:
        return False
    if previous.operator in {"In", "NotIn"}:
        return _membership_preserved(previous, current)
    if previous.bound is None:
        return True
    if current.bound is None:
        return False
    if previous.operator in {"Gt", "GtE"}:
        return current.bound >= previous.bound
    if previous.operator in {"Lt", "LtE"}:
        return current.bound <= previous.bound
    return True


def _membership_preserved(
    previous: ComparisonContract, current: ComparisonContract
) -> bool:
    if previous.members is None or current.members is None:
        return (
            previous.members is None
            and current.members is None
            and previous.membership_fingerprint == current.membership_fingerprint
        )
    if previous.operator == "In":
        return current.members <= previous.members
    return current.members >= previous.members


def _direct_raise_counts(sites: Sequence[RaisesSite]) -> Counter[str]:
    return Counter(site.scope for site in sites if site.directly_raises)


def _assertion_violations(
    previous: Sequence[AssertionSummary], current: Sequence[AssertionSummary]
) -> list[str]:
    current_assertions = {item.scope: item for item in current}
    violations: list[str] = []
    empty = AssertionSummary("", 0, 0, ())
    for item in previous:
        latest = current_assertions.get(item.scope, empty)
        weakened = item.count > latest.count or not _comparisons_preserved(
            item.comparisons, latest.comparisons
        )
        if weakened:
            violations.append(f"removed or weakened assertion: {item.scope}")
        if latest.obviously_true > item.obviously_true:
            violations.append(f"constant-true assertion added: {item.scope}")
    return violations


def _raise_violations(
    previous: Sequence[RaisesSite], current: Sequence[RaisesSite]
) -> list[str]:
    missing_raises = _missing_raises(previous, current)
    violations = [
        f"removed or weakened pytest.raises: {item.scope}[{item.ordinal}]"
        for item in missing_raises
    ]
    previous_direct = _direct_raise_counts(previous)
    current_direct = _direct_raise_counts(current)
    violations.extend(
        f"direct raise added under pytest.raises: {scope}"
        for scope, count in current_direct.items()
        if count > previous_direct.get(scope, 0)
        and all(item.scope != scope for item in missing_raises)
    )
    return violations


def _snapshot_violations(previous: FileSnapshot, current: FileSnapshot) -> list[str]:
    violations = [
        f"removed test contract: {name}"
        for name in previous.tests
        if name not in current.tests
    ]
    violations.extend(_assertion_violations(previous.assertions, current.assertions))
    violations.extend(_raise_violations(previous.raises, current.raises))
    violations.extend(
        f"removed or relaxed numerical assertion: {item.scope}[{item.ordinal}]"
        for item in _missing_tolerances(previous.tolerances, current.tolerances)
    )
    return violations


def _policy_diff_violations(diff: str) -> list[str]:
    violations: list[str] = []
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].replace("\\", "/")
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        addition = line[1:]
        forbidden_secret = FORBIDDEN_SECRET_ADDITION.search(addition)
        forbidden_python = current_file.endswith(".py") and (
            FORBIDDEN_PYTHON_ADDITION.search(addition)
        )
        if forbidden_secret or forbidden_python:
            violations.append(addition.strip())
    return violations


def _base_test_sources(base: str) -> dict[str, str]:
    paths = _git("ls-tree", "-r", "--name-only", base, "--", "tests").splitlines()
    return {
        path: _git("show", f"{base}:{path}") for path in paths if path.endswith(".py")
    }


def _current_test_sources() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "tests").glob("**/*.py"))
    }


def _snapshots(sources: dict[str, str]) -> dict[str, FileSnapshot]:
    return {path: _snapshot(source, namespace=path) for path, source in sources.items()}


def _repository_violations(
    previous: dict[str, FileSnapshot], current: dict[str, FileSnapshot]
) -> list[str]:
    return [
        violation
        for path, old_snapshot in previous.items()
        for violation in _snapshot_violations(
            old_snapshot, current.get(path, EMPTY_SNAPSHOT)
        )
    ]


def _removed_contract_violations(base: str) -> list[str]:
    return _repository_violations(
        _snapshots(_base_test_sources(base)), _snapshots(_current_test_sources())
    )


def _check_policy_additions() -> None:
    base = _base_ref()
    if base is None:
        return
    previous = _snapshots(_base_test_sources(base))
    current = _snapshots(_current_test_sources())
    violations = _policy_diff_violations(_git("diff", "--unified=0", base, "--"))
    violations.extend(_repository_violations(previous, current))
    violations.extend(
        violation for snapshot in current.values() for violation in snapshot.skips
    )
    if violations:
        _die(
            "Forbidden quality-policy weakening:\n"
            + "\n".join(f"  {line}" for line in violations)
        )
