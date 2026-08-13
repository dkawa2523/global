from __future__ import annotations

import ast
import fnmatch
import importlib
import tomllib
from pathlib import Path

import plasma_global
import plasma_global.api
from plasma_global import errors
from plasma_global.core import exceptions as legacy_core_exceptions

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "plasma_global" / "core"
ALLOWED_CORE_PREFIXES = (
    "plasma_global.core",
    "plasma_global.errors",
    "plasma_global.models",
)
LEGACY_PACKAGE_NAMES = {
    "config",
    "coupling",
    "eedf",
    "electrical",
    "io",
    "numerics",
    "observables",
    "physics",
    "plotters",
    "reactor",
    "workflows",
}
LEGACY_ROOT_MODULES = {"backends.py", "validation.py"}
INTERNAL_PACKAGE_INITIALIZERS = (
    "chemistry",
    "core",
    "experimental",
    "input",
    "models",
)
FORBIDDEN_PUBLIC_ALIASES = (
    "CaseConfig",
    "build_case",
    "compile_case",
    "load_config",
    "run_case",
    "save_result",
    "simulate_case",
    "write_output",
)


def _plasma_global_references(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    references: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            references.add(node.module)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("plasma_global.")
        ):
            references.add(node.value)
    return {name for name in references if name.startswith("plasma_global.")}


def test_core_imports_only_core_models_and_shared_errors() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted(CORE.rglob("*.py")):
        rejected = sorted(
            reference
            for reference in _plasma_global_references(path)
            if not any(
                reference == prefix or reference.startswith(f"{prefix}.")
                for prefix in ALLOWED_CORE_PREFIXES
            )
        )
        if rejected:
            offenders[path.relative_to(ROOT).as_posix()] = rejected

    assert offenders == {}


def test_core_exception_imports_are_compatibility_aliases() -> None:
    assert (
        legacy_core_exceptions.ModelConfigurationError is errors.ModelConfigurationError
    )
    assert legacy_core_exceptions.StateDomainError is errors.StateDomainError
    assert legacy_core_exceptions.QuasineutralityError is errors.QuasineutralityError


def test_compiled_power_path_has_no_dynamic_configuration_contract() -> None:
    for relative_path in (
        "plasma_global/core/domain.py",
        "plasma_global/input/_compile_power.py",
        "plasma_global/input/compile_reactor.py",
        "plasma_global/input/compile_recipe.py",
        "plasma_global/models/power.py",
    ):
        path = ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        dynamic_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
        ]
        assert dynamic_calls == [], relative_path
        assert "Mapping[str, object]" not in source, relative_path
        assert "command: object" not in source, relative_path


def test_top_level_api_is_exactly_three_operations() -> None:
    expected = ["load_case", "simulate", "write_result"]

    assert plasma_global.__all__ == expected
    assert plasma_global.api.__all__ == expected
    assert all(callable(getattr(plasma_global, name)) for name in expected)
    assert all(
        getattr(plasma_global, name) is getattr(plasma_global.api, name)
        for name in expected
    )
    assert not any(hasattr(plasma_global, name) for name in FORBIDDEN_PUBLIC_ALIASES)
    assert not any(
        hasattr(plasma_global.api, name) for name in FORBIDDEN_PUBLIC_ALIASES
    )


def test_legacy_runtime_packages_and_modules_have_no_source_code() -> None:
    legacy_sources = {
        path.relative_to(ROOT).as_posix()
        for package_name in LEGACY_PACKAGE_NAMES
        for path in (ROOT / "plasma_global" / package_name).rglob("*.py")
    }
    legacy_sources.update(
        f"plasma_global/{module_name}"
        for module_name in LEGACY_ROOT_MODULES
        if (ROOT / "plasma_global" / module_name).exists()
    )

    assert legacy_sources == set()


def test_internal_package_initializers_do_not_reexport_implementations() -> None:
    for package_name in INTERNAL_PACKAGE_INITIALIZERS:
        module = importlib.import_module(f"plasma_global.{package_name}")
        assert module.__all__ == []

        initializer = ROOT / "plasma_global" / package_name / "__init__.py"
        tree = ast.parse(initializer.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
        )


def test_unreferenced_presentation_assets_are_absent() -> None:
    presentation_suffixes = {".key", ".odp", ".ppt", ".pptx"}
    presentation_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if (
            (path.is_dir() and path.name.casefold() == "slides")
            or (path.is_file() and path.suffix.casefold() in presentation_suffixes)
        )
    }

    assert presentation_paths == set()


def test_distribution_excludes_caches_assets_and_environment_directories() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools_config = config["tool"]["setuptools"]
    discovery = setuptools_config["packages"]["find"]
    includes = discovery["include"]
    excludes = discovery["exclude"]
    ruff_excludes = set(config["tool"]["ruff"]["extend-exclude"])

    assert setuptools_config["include-package-data"] is False
    assert set(includes) == {"plasma_global*", "tools*"}
    assert {
        "*.assets",
        "*.assets.*",
        "*.__pycache__",
        "*.__pycache__.*",
    }.issubset(excludes)
    assert {
        ".pytest_cache",
        ".ruff_cache",
        ".uv-cache",
        ".venv",
        "**/__pycache__",
        "*.egg-info",
        "build",
        "dist",
    }.issubset(ruff_excludes)
    assert all(
        not any(fnmatch.fnmatch(name, pattern) for pattern in includes)
        for name in (
            ".pytest_cache",
            ".ruff_cache",
            ".uv-cache",
            ".venv",
            "benchmarks",
            "docs",
            "examples",
            "scripts",
            "slides",
            "tests",
        )
    )

    candidates = {
        path.parent.relative_to(ROOT).as_posix().replace("/", ".")
        for path in ROOT.rglob("*.py")
    }
    packages = {
        package
        for package in candidates
        if any(fnmatch.fnmatch(package, pattern) for pattern in includes)
        and not any(fnmatch.fnmatch(package, pattern) for pattern in excludes)
    }
    forbidden_parts = {"__pycache__", "assets", ".venv", ".uv-cache"}
    assert packages
    assert all(forbidden_parts.isdisjoint(package.split(".")) for package in packages)

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    ignored = {
        line.strip() for line in gitignore if line.strip() and not line.startswith("#")
    }
    assert {
        "__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".coverage",
        ".venv/",
        ".uv-cache/",
        "*.egg-info/",
    }.issubset(ignored)
