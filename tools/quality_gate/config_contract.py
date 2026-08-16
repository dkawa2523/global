"""Prevent quality tools from silently skipping code."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.quality_gate.context import (
    ROOT,
    QualityFailure,
    _base_ref,
    _source_for_path,
)


def _table(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: object) -> set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return set()
    return set(value)


def _named_lists(
    value: Mapping[str, Any], names: set[str], prefix: str = ""
) -> dict[str, set[str]]:
    """Collect list-valued settings without interpreting a tool's schema."""

    found: dict[str, set[str]] = {}
    for name, item in value.items():
        path = f"{prefix}.{name}" if prefix else name
        if name in names:
            found[path] = _string_list(item)
        if isinstance(item, dict):
            found.update(_named_lists(item, names, path))
    return found


def _ruff_rule_violations(
    ruff: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[str]:
    violations: list[str] = []
    lint = _table(ruff.get("lint"))
    reference_lint = _table(reference.get("lint"))
    selected = _string_list(lint.get("select")) | _string_list(
        lint.get("extend-select")
    )
    reference_selected = _string_list(reference_lint.get("select")) | _string_list(
        reference_lint.get("extend-select")
    )
    if missing := reference_selected - selected:
        violations.append("Ruff select cannot shrink: " + ", ".join(sorted(missing)))
    if "ALL" in selected:
        violations.append("Ruff select must list rules explicitly instead of ALL")
    if ruff.get("fix") is True or ruff.get("unsafe-fixes") is True:
        violations.append("Ruff quality checks must not rewrite source")

    global_ignores = _named_lists(ruff, {"ignore", "extend-ignore"})
    if ignored := {path: rules for path, rules in global_ignores.items() if rules}:
        violations.append(
            "Ruff rules cannot be ignored globally: " + ", ".join(sorted(ignored))
        )
    return violations


def _ruff_exclusion_violations(
    ruff: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[str]:
    current_excludes = _named_lists(ruff, {"exclude", "extend-exclude"})
    reference_excludes = _named_lists(reference, {"exclude", "extend-exclude"})
    additions = {
        pattern
        for path, patterns in current_excludes.items()
        for pattern in patterns - reference_excludes.get(path, set())
    }
    return (
        ["Ruff excludes cannot grow silently: " + ", ".join(sorted(additions))]
        if additions
        else []
    )


def _ruff_per_file_ignore_violations(
    ruff: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[str]:
    lint = _table(ruff.get("lint"))
    reference_lint = _table(reference.get("lint"))
    current_ignores = {
        **_table(lint.get("per-file-ignores")),
        **_table(lint.get("extend-per-file-ignores")),
    }
    reference_ignores = {
        **_table(reference_lint.get("per-file-ignores")),
        **_table(reference_lint.get("extend-per-file-ignores")),
    }
    ignore_additions = {
        f"{pattern}:{rule}"
        for pattern, raw_rules in current_ignores.items()
        for rule in _string_list(raw_rules)
        - _string_list(reference_ignores.get(pattern))
    }
    return (
        ["Ruff per-file ignores cannot grow: " + ", ".join(sorted(ignore_additions))]
        if ignore_additions
        else []
    )


def _ruff_violations(
    tool: Mapping[str, Any], reference_tool: Mapping[str, Any]
) -> list[str]:
    ruff = _table(tool.get("ruff"))
    reference = _table(reference_tool.get("ruff"))
    return [
        *_ruff_rule_violations(ruff, reference),
        *_ruff_exclusion_violations(ruff, reference),
        *_ruff_per_file_ignore_violations(ruff, reference),
    ]


def _coverage_violations(
    tool: Mapping[str, Any], reference_tool: Mapping[str, Any]
) -> list[str]:
    coverage = _table(tool.get("coverage"))
    reference = _table(reference_tool.get("coverage"))
    violations: list[str] = []
    run = _table(coverage.get("run"))
    reference_run = _table(reference.get("run"))
    if reference_run.get("branch") is True and run.get("branch") is not True:
        violations.append("coverage branch measurement cannot be disabled")
    if missing := _string_list(reference_run.get("source")) - _string_list(
        run.get("source")
    ):
        violations.append(
            "coverage source cannot shrink: " + ", ".join(sorted(missing))
        )
    current_exclusions = _named_lists(
        coverage,
        {"omit", "exclude_lines", "exclude_also", "partial_branches", "partial_also"},
    )
    reference_exclusions = _named_lists(
        reference,
        {"omit", "exclude_lines", "exclude_also", "partial_branches", "partial_also"},
    )
    additions = {
        pattern
        for path, patterns in current_exclusions.items()
        for pattern in patterns - reference_exclusions.get(path, set())
    }
    if additions:
        violations.append(
            "coverage exclusions cannot grow: " + ", ".join(sorted(additions))
        )
    return violations


def _pyrefly_violations(
    tool: Mapping[str, Any], reference_tool: Mapping[str, Any]
) -> list[str]:
    pyrefly = _table(tool.get("pyrefly"))
    reference = _table(reference_tool.get("pyrefly"))
    violations: list[str] = []
    if additions := set(pyrefly) - set(reference) - {"errors"}:
        violations.append(
            "Pyrefly options are not reviewed: " + ", ".join(sorted(additions))
        )
    for name in ("preset", "min-severity"):
        if reference.get(name) != pyrefly.get(name):
            violations.append(f"Pyrefly {name} cannot be weakened")
    if missing := _string_list(reference.get("project-includes")) - _string_list(
        pyrefly.get("project-includes")
    ):
        violations.append(
            "Pyrefly project-includes cannot shrink: " + ", ".join(sorted(missing))
        )
    allowed_disabled = {"unnecessary-type-conversion"}
    errors = _table(pyrefly.get("errors"))
    reference_errors = _table(reference.get("errors"))
    changed_errors = {
        name
        for name, value in errors.items()
        if value != reference_errors.get(name)
        and not (name in allowed_disabled and value is False)
    }
    if changed_errors:
        violations.append(
            "Pyrefly diagnostics are not reviewed: " + ", ".join(sorted(changed_errors))
        )
    return violations


def _forbidden_edges(tool: Mapping[str, Any]) -> set[tuple[str, str, bool, bool]]:
    import_linter = _table(tool.get("importlinter"))
    contracts = import_linter.get("contracts")
    if not isinstance(contracts, list):
        return set()

    edges: set[tuple[str, str, bool, bool]] = set()
    for value in contracts:
        contract = _table(value)
        if contract.get("type") != "forbidden" or contract.get("ignore_imports"):
            continue
        sources = _string_list(contract.get("source_modules"))
        targets = _string_list(contract.get("forbidden_modules"))
        as_packages = contract.get("as_packages", True) is True
        indirect = contract.get("allow_indirect_imports", False) is not True
        for source in sources:
            for target in targets:
                edges.add((source, target, as_packages, False))
                if indirect:
                    edges.add((source, target, as_packages, True))
    return edges


def _import_linter_violations(
    tool: Mapping[str, Any], reference_tool: Mapping[str, Any]
) -> list[str]:
    current = _table(tool.get("importlinter"))
    reference = _table(reference_tool.get("importlinter"))
    violations: list[str] = []
    if current.get("root_package") != reference.get("root_package"):
        violations.append("import-linter root_package cannot change")
    missing = _forbidden_edges(reference_tool) - _forbidden_edges(tool)
    if not missing:
        return violations
    edges = sorted({f"{source} -> {target}" for source, target, _, _ in missing})
    violations.append("import-linter contracts were weakened: " + ", ".join(edges))
    return violations


def _mutmut_violations(
    tool: Mapping[str, Any], reference_tool: Mapping[str, Any]
) -> list[str]:
    mutmut = _table(tool.get("mutmut"))
    reference = _table(reference_tool.get("mutmut"))
    missing = {
        f"{name}:{value}"
        for name in (
            "source_paths",
            "only_mutate",
            "pytest_add_cli_args_test_selection",
        )
        for value in _string_list(reference.get(name)) - _string_list(mutmut.get(name))
    }
    if not missing:
        return []
    return ["mutation scope cannot shrink: " + ", ".join(sorted(missing))]


def _config_violations(
    config: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[str]:
    tool = _table(config.get("tool"))
    reference_tool = _table(reference.get("tool"))
    return [
        *_ruff_violations(tool, reference_tool),
        *_pyrefly_violations(tool, reference_tool),
        *_coverage_violations(tool, reference_tool),
        *_import_linter_violations(tool, reference_tool),
        *_mutmut_violations(tool, reference_tool),
    ]


def _check_quality_config(path: Path | None = None) -> None:
    """Reject only configuration changes that can hide unchecked code."""

    config_path = ROOT / "pyproject.toml" if path is None else path
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise QualityFailure(f"Cannot read quality configuration: {exc}") from exc

    violations = _config_violations(config, _reference_config(config_path, config))
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise QualityFailure(f"Quality configuration was weakened:\n{details}")


def _reference_config(
    config_path: Path, current: Mapping[str, Any]
) -> Mapping[str, Any]:
    primary = (ROOT / "pyproject.toml").resolve()
    if config_path.resolve() != primary:
        source = primary.read_text(encoding="utf-8")
    else:
        source = _source_for_path("pyproject.toml", source_ref=_base_ref())
        if source is None:
            return current
    try:
        return tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise QualityFailure(
            f"Cannot read reference quality configuration: {exc}"
        ) from exc
