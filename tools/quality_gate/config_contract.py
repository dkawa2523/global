"""Runtime contract preventing silent weakening of quality configuration."""

from __future__ import annotations

import os
import re
import shlex
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

REQUIRED_RUFF_RULES = {
    "E",
    "F",
    "I",
    "UP",
    "B",
    "SIM",
    "C4",
    "A",
    "TID",
    "PT",
    "RUF",
    "S",
    "C90",
}
ALLOWED_RUFF_IGNORES = {
    ("tools/benchmarks/*.py", "E402"),
    ("tools/quality_gate/context.py", "S603"),
    ("tools/quality_gate/context.py", "S607"),
    ("tests/**/*.py", "S101"),
}
ALLOWED_RUFF_EXCLUDES = {
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "**/__pycache__",
    "*.egg-info",
    "build",
    "dist",
    "htmlcov",
    "site",
}
REQUIRED_PYREFLY_INCLUDES = {"plasma_global/**/*.py", "tools/**/*.py"}
REQUIRED_PYTEST_MARKERS = {"external_benchmark", "nightly", "property"}
REQUIRED_PYTEST_EXCLUSIONS = {"external_benchmark", "nightly"}
ALTERNATE_CONFIG_NAMES = {
    ".coveragerc",
    ".importlinter",
    ".ruff.toml",
    "pyrefly.toml",
    "pytest.ini",
    "ruff.toml",
    "setup.cfg",
    "tox.ini",
}
CONFIG_SEARCH_EXCLUDES = {
    ".agents",
    ".codex",
    ".git",
    ".hypothesis",
    ".pytest_cache",
    ".pyrefly_cache",
    ".quality-reports",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "mutants",
    "node_modules",
    "site",
    "site_ja",
    "venv",
    "env",
}


def _table(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _strings(value: object) -> set[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return set(value)


def _missing(required: set[str], configured: object) -> set[str]:
    values = _strings(configured)
    return required if values is None else required - values


def _unexpected_keys(
    table: Mapping[str, Any], allowed: set[str], label: str
) -> list[str]:
    unexpected = set(table) - allowed
    if not unexpected:
        return []
    return [f"{label} contains unapproved settings: {', '.join(sorted(unexpected))}"]


def _ruff_selection_violations(lint: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    missing = _missing(REQUIRED_RUFF_RULES, lint.get("select"))
    if missing:
        violations.append(f"Ruff select is missing: {', '.join(sorted(missing))}")
    selected = (_strings(lint.get("select")) or set()) | (
        _strings(lint.get("extend-select")) or set()
    )
    if "ALL" in selected:
        violations.append("Ruff ALL is forbidden; select rules explicitly")
    return violations


def _ruff_runtime_violations(ruff: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    if ruff.get("target-version") != "py311":
        violations.append("Ruff target-version must remain py311")
    line_length = ruff.get("line-length")
    if (
        isinstance(line_length, bool)
        or not isinstance(line_length, int)
        or line_length > 88
    ):
        violations.append("Ruff line-length must be an integer no greater than 88")
    excluded = _strings(ruff.get("extend-exclude", []))
    if excluded is None or excluded - ALLOWED_RUFF_EXCLUDES:
        violations.append("Ruff extend-exclude contains unapproved paths")
    return violations


def _ruff_lint_limit_violations(lint: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    maximum = _table(lint.get("mccabe")).get("max-complexity")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum > 10:
        violations.append("Ruff max-complexity must be an integer no greater than 10")
    for key in ("ignore", "extend-ignore"):
        ignored = _strings(lint.get(key))
        if ignored:
            violations.append(f"Ruff {key} cannot disable rules globally")
    return violations


def _ruff_limit_violations(
    ruff: Mapping[str, Any], lint: Mapping[str, Any]
) -> list[str]:
    return [*_ruff_runtime_violations(ruff), *_ruff_lint_limit_violations(lint)]


def _ruff_violations(tool: Mapping[str, Any]) -> list[str]:
    ruff = _table(tool.get("ruff"))
    lint = _table(ruff.get("lint"))
    violations = _unexpected_keys(
        ruff, {"target-version", "line-length", "extend-exclude", "lint"}, "Ruff"
    )
    violations.extend(
        _unexpected_keys(
            lint,
            {
                "select",
                "extend-select",
                "mccabe",
                "per-file-ignores",
                "extend-per-file-ignores",
            },
            "Ruff lint",
        )
    )
    violations.extend(_ruff_selection_violations(lint))
    violations.extend(_ruff_limit_violations(ruff, lint))
    violations.extend(_ruff_ignore_violations(lint.get("per-file-ignores")))
    violations.extend(_ruff_ignore_violations(lint.get("extend-per-file-ignores")))
    return violations


def _ruff_ignore_violations(value: object) -> list[str]:
    ignores = _table(value)
    violations: list[str] = []
    for pattern, rules in ignores.items():
        configured = _strings(rules)
        if configured is None:
            violations.append(f"Ruff ignores for {pattern} must be a string array")
            continue
        unexpected = {(pattern, rule) for rule in configured} - ALLOWED_RUFF_IGNORES
        if unexpected:
            extras = ", ".join(rule for _, rule in sorted(unexpected))
            violations.append(f"Ruff adds unapproved ignores for {pattern}: {extras}")
    return violations


def _pyrefly_violations(tool: Mapping[str, Any]) -> list[str]:
    pyrefly = _table(tool.get("pyrefly"))
    violations = _unexpected_keys(
        pyrefly,
        {"preset", "min-severity", "project-includes"},
        "Pyrefly",
    )
    if pyrefly.get("preset") != "strict":
        violations.append("Pyrefly preset must remain strict")
    if pyrefly.get("min-severity") != "warn":
        violations.append("Pyrefly min-severity must remain warn")
    missing = _missing(REQUIRED_PYREFLY_INCLUDES, pyrefly.get("project-includes"))
    if missing:
        violations.append(f"Pyrefly includes are missing: {', '.join(sorted(missing))}")
    return violations


def _marker_expression(arguments: list[str]) -> str | None:
    positions = [index for index, value in enumerate(arguments) if value == "-m"]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        return None
    return arguments[positions[0] + 1]


def _negated_markers(expression: str | None) -> set[str] | None:
    if expression is None:
        return None
    parts = re.split(r"\s+and\s+", expression.strip())
    matches = [re.fullmatch(r"not\s+([A-Za-z_]\w*)", part.strip()) for part in parts]
    if not matches or any(match is None for match in matches):
        return None
    return {match.group(1) for match in matches if match is not None}


def _marker_exclusions(addopts: object) -> set[str] | None:
    if not isinstance(addopts, str):
        return None
    try:
        arguments = shlex.split(addopts)
    except ValueError:
        return None
    return _negated_markers(_marker_expression(arguments))


def _marker_names(value: object) -> set[str] | None:
    markers = _strings(value)
    if markers is None:
        return None
    return {marker.partition(":")[0].strip() for marker in markers}


def _pytest_violations(tool: Mapping[str, Any]) -> list[str]:
    pytest = _table(_table(tool.get("pytest")).get("ini_options"))
    violations = _unexpected_keys(pytest, {"testpaths", "addopts", "markers"}, "pytest")
    if "tests" in _missing({"tests"}, pytest.get("testpaths")):
        violations.append("pytest testpaths must include tests")
    names = _marker_names(pytest.get("markers"))
    missing = (
        REQUIRED_PYTEST_MARKERS if names is None else REQUIRED_PYTEST_MARKERS - names
    )
    if missing:
        violations.append(
            f"pytest marker declarations are missing: {', '.join(sorted(missing))}"
        )
    exclusions = _marker_exclusions(pytest.get("addopts"))
    if exclusions != REQUIRED_PYTEST_EXCLUSIONS:
        violations.append(
            "pytest must exclude exactly external_benchmark and nightly by default"
        )
    arguments: list[str]
    try:
        arguments = shlex.split(str(pytest.get("addopts", "")))
    except ValueError:
        arguments = []
    expected_arguments = ["-q", "-m", "not external_benchmark and not nightly"]
    if arguments != expected_arguments:
        violations.append("pytest addopts contains unapproved options")
    return violations


def _coverage_violations(tool: Mapping[str, Any]) -> list[str]:
    coverage = _table(tool.get("coverage"))
    run = _table(coverage.get("run"))
    report = _table(coverage.get("report"))
    violations = _unexpected_keys(coverage, {"run", "report"}, "coverage")
    violations.extend(_unexpected_keys(run, {"branch", "source"}, "coverage run"))
    violations.extend(
        _unexpected_keys(
            report,
            {"show_missing", "skip_covered", "precision"},
            "coverage report",
        )
    )
    if run.get("branch") is not True:
        violations.append("coverage branch measurement must remain enabled")
    if "plasma_global" in _missing({"plasma_global"}, run.get("source")):
        violations.append("coverage source must include plasma_global")
    return violations


def _import_contract_option_violations(contracts: object) -> list[str]:
    if not isinstance(contracts, list):
        return ["import-linter contracts must be an array"]
    allowed = {
        "name",
        "type",
        "source_modules",
        "forbidden_modules",
        "as_packages",
        "allow_indirect_imports",
    }
    return [
        violation
        for index, value in enumerate(contracts)
        for violation in _unexpected_keys(
            _table(value), allowed, f"import-linter contract {index}"
        )
    ]


def _indirect_requirements(allows_indirect: bool) -> tuple[bool, ...]:
    return (False,) if allows_indirect else (False, True)


def _forbidden_edges(contracts: object) -> set[tuple[str, str, bool, bool]]:
    """Return direct and indirect requirements for each forbidden edge."""

    if not isinstance(contracts, list):
        return set()
    edges: set[tuple[str, str, bool, bool]] = set()
    for value in contracts:
        contract = _table(value)
        if contract.get("type") != "forbidden" or contract.get("ignore_imports"):
            continue
        sources = _strings(contract.get("source_modules")) or set()
        targets = _strings(contract.get("forbidden_modules")) or set()
        as_packages = contract.get("as_packages", True) is True
        allows_indirect = contract.get("allow_indirect_imports", False) is True
        edges.update(
            (source, target, as_packages, indirect)
            for source in sources
            for target in targets
            for indirect in _indirect_requirements(allows_indirect)
        )
    return edges


def _missing_edge_identities(
    current: set[tuple[str, str, bool, bool]],
    reference: set[tuple[str, str, bool, bool]],
) -> set[tuple[str, str, bool]]:
    return {
        (source, target, as_packages)
        for source, target, as_packages, _ in reference - current
    }


def _import_violations(
    tool: Mapping[str, Any], reference_tool: Mapping[str, Any]
) -> list[str]:
    import_linter = _table(tool.get("importlinter"))
    reference = _table(reference_tool.get("importlinter"))
    violations = _unexpected_keys(
        import_linter, {"root_package", "contracts"}, "import-linter"
    )
    violations.extend(
        _import_contract_option_violations(import_linter.get("contracts"))
    )
    if import_linter.get("root_package") != "plasma_global":
        violations.append("import-linter root_package must remain plasma_global")
    current_edges = _forbidden_edges(import_linter.get("contracts"))
    reference_edges = _forbidden_edges(reference.get("contracts"))
    missing = _missing_edge_identities(current_edges, reference_edges)
    if missing:
        description = ", ".join(
            f"{source} -> {target}" for source, target, _ in sorted(missing)
        )
        violations.append(f"import-linter contracts were weakened: {description}")
    return violations


def _mutmut_violations(tool: Mapping[str, Any]) -> list[str]:
    mutmut = _table(tool.get("mutmut"))
    violations = _unexpected_keys(
        mutmut,
        {"source_paths", "only_mutate", "pytest_add_cli_args_test_selection"},
        "mutmut",
    )
    if mutmut.get("source_paths") != ["plasma_global"]:
        violations.append("mutmut source_paths must remain plasma_global")
    if mutmut.get("only_mutate") != ["plasma_global/models/rates.py"]:
        violations.append(
            "mutmut only_mutate must remain plasma_global/models/rates.py"
        )
    expected_tests = ["tests/test_rate_contracts.py"]
    if mutmut.get("pytest_add_cli_args_test_selection") != expected_tests:
        violations.append(
            "mutmut test selection must remain tests/test_rate_contracts.py"
        )
    return violations


def _config_violations(
    config: Mapping[str, Any], reference: Mapping[str, Any]
) -> list[str]:
    tool = _table(config.get("tool"))
    reference_tool = _table(reference.get("tool"))
    return [
        *_ruff_violations(tool),
        *_pyrefly_violations(tool),
        *_pytest_violations(tool),
        *_coverage_violations(tool),
        *_import_violations(tool, reference_tool),
        *_mutmut_violations(tool),
    ]


def _alternate_config_paths(config_path: Path) -> list[str]:
    root = config_path.parent.resolve()
    primary = config_path.resolve()
    alternates: list[str] = []
    for directory, children, files in os.walk(root):
        children[:] = [
            child for child in children if child not in CONFIG_SEARCH_EXCLUDES
        ]
        parent = Path(directory)
        for name in files:
            candidate = parent / name
            is_alternate = name in ALTERNATE_CONFIG_NAMES
            is_nested_project = (
                name == "pyproject.toml" and candidate.resolve() != primary
            )
            if is_alternate or is_nested_project:
                alternates.append(candidate.relative_to(root).as_posix())
    return sorted(alternates)


def _check_quality_config(path: Path | None = None) -> None:
    """Reject configuration changes that weaken the repository quality gates."""

    config_path = ROOT / "pyproject.toml" if path is None else path
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise QualityFailure(f"Cannot read quality configuration: {exc}") from exc
    reference = _reference_config(config_path, config)
    alternates = _alternate_config_paths(config_path)
    violations = _config_violations(config, reference)
    if alternates:
        violations.append(
            "alternate quality configuration files are forbidden: "
            + ", ".join(alternates)
        )
    if violations:
        details = "\n".join(f"- {violation}" for violation in violations)
        raise QualityFailure(f"Quality configuration was weakened:\n{details}")


def _reference_config(
    config_path: Path, current: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Load the comparison config used for monotonic import contracts."""

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
