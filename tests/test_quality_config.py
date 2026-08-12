from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import settings

from tools.quality_gate.config_contract import _check_quality_config
from tools.quality_gate.context import ROOT, QualityFailure

PYPROJECT = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    assert old in PYPROJECT
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT.replace(old, new, 1), encoding="utf-8")
    return path


def test_current_quality_configuration_satisfies_contract() -> None:
    _check_quality_config()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('"C90"', '"D"', "Ruff select is missing"),
        (
            'select = ["E", "F", "I", "UP", "B", "SIM", "C4", "A", "TID", '
            '"PT", "RUF", "S", "C90"]',
            'select = ["E", "F", "I", "UP", "B", "SIM", "C4", "A", "TID", '
            '"PT", "RUF", "S", "C90"]\nignore = ["F821"]',
            "cannot disable rules globally",
        ),
        ("max-complexity = 10", "max-complexity = 11", "max-complexity"),
        ('target-version = "py311"', 'target-version = "py310"', "target-version"),
        ("line-length = 88", "line-length = 89", "line-length"),
        ('"S", "C90"]', '"S", "C90", "ALL"]', "Ruff ALL is forbidden"),
        ('  "site",', '  "site",\n  "tests",', "unapproved paths"),
        ("line-length = 88", "line-length = 88\nfix = true", "unapproved settings"),
        (
            '"tests/**/*.py" = ["S101"]',
            '"tests/**/*.py" = ["S101", "F821"]',
            "unapproved ignores",
        ),
        ('preset = "strict"', 'preset = "basic"', "preset must remain strict"),
        ('min-severity = "warn"', 'min-severity = "error"', "must remain warn"),
        (
            'min-severity = "warn"',
            'min-severity = "warn"\nignore = ["bad-argument-type"]',
            "unapproved settings",
        ),
        (
            'project-includes = ["plasma_global/**/*.py", "tools/**/*.py"]',
            'project-includes = ["plasma_global/**/*.py"]',
            "Pyrefly includes are missing",
        ),
        ('testpaths = ["tests"]', 'testpaths = ["checks"]', "testpaths"),
        (
            "addopts = \"-q -m 'not external_benchmark and not nightly'\"",
            'addopts = "-q --ignore=tests/test_v3_power.py '
            "-m 'not external_benchmark and not nightly'\"",
            "unapproved options",
        ),
        (
            "not external_benchmark and not nightly",
            "not external_benchmark and not nightly and not property",
            "must exclude exactly",
        ),
        ("branch = true", "branch = false", "branch measurement"),
        ("branch = true", 'branch = true\nomit = ["*/models/*"]', "unapproved"),
        ('source = ["plasma_global"]', 'source = ["tools"]', "coverage source"),
        (
            '  "plasma_global.postprocess",',
            '  "plasma_global.unknown",',
            "import-linter",
        ),
        (
            "allow_indirect_imports = true",
            "as_packages = false\nallow_indirect_imports = true",
            "plasma_global.core contract is missing",
        ),
        (
            'source_paths = ["plasma_global"]',
            'source_paths = ["tools"]',
            "source_paths",
        ),
        (
            'only_mutate = ["plasma_global/models/rates.py"]',
            'only_mutate = ["plasma_global/models/power.py"]',
            "mutmut only_mutate",
        ),
        (
            'pytest_add_cli_args_test_selection = ["tests/test_rate_contracts.py"]',
            'pytest_add_cli_args_test_selection = ["tests/test_v3_power.py"]',
            "mutmut test selection",
        ),
    ],
)
def test_quality_contract_rejects_weakened_settings(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = _variant(tmp_path, old, new)

    with pytest.raises(QualityFailure, match=message):
        _check_quality_config(path)


def test_quality_contract_allows_stronger_rules_and_fewer_ignores(
    tmp_path: Path,
) -> None:
    source = (
        PYPROJECT.replace(
            '"S", "C90"]',
            '"S", "C90", "D"]',
            1,
        )
        .replace(
            '"tests/**/*.py" = ["S101"]',
            '"tests/**/*.py" = []',
            1,
        )
        .replace(
            '  "plasma_global.postprocess",',
            '  "plasma_global.postprocess",\n  "plasma_global.unknown",',
            1,
        )
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(source, encoding="utf-8")

    _check_quality_config(path)


def test_hypothesis_profiles_keep_required_example_counts() -> None:
    assert settings.get_profile("ci").max_examples >= 100
    assert settings.get_profile("nightly").max_examples >= 1_000


@pytest.mark.parametrize(
    "relative_path", ["ruff.toml", "plasma_global/ruff.toml", "tools/pyproject.toml"]
)
def test_quality_contract_rejects_alternate_configuration_files(
    tmp_path: Path,
    relative_path: str,
) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(PYPROJECT, encoding="utf-8")
    alternate = tmp_path / relative_path
    alternate.parent.mkdir(parents=True, exist_ok=True)
    alternate.write_text('lint.ignore = ["F821"]', encoding="utf-8")

    with pytest.raises(QualityFailure, match="alternate quality configuration"):
        _check_quality_config(path)
