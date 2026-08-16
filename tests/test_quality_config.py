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


def test_property_profiles_keep_scientific_sample_floors() -> None:
    assert settings.get_profile("ci").max_examples >= 100
    assert settings.get_profile("nightly").max_examples >= 1_000


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "[tool.ruff.lint]\nselect =",
            '[tool.ruff.lint]\nignore = ["F821"]\nselect =',
            "ignored globally",
        ),
        (
            "line-length = 88",
            'line-length = 88\nexclude = ["plasma_global"]',
            "excludes cannot grow",
        ),
        (
            'select = ["E", "F", "I", "UP", "B", "SIM", "C4", "A", '
            '"TID", "PT", "RUF", "S", "C90"]',
            'select = ["F"]',
            "Ruff select cannot shrink",
        ),
        (
            '"tools/benchmarks/*.py" = ["E402"]',
            '"tools/benchmarks/*.py" = ["E402", "F821"]',
            "per-file ignores cannot grow",
        ),
        (
            '"tests/**/*.py" = ["S101"]',
            '"tests/**/*.py" = ["S101"]\n\n'
            "[tool.ruff.lint.extend-per-file-ignores]\n"
            '"plasma_global/*.py" = ["F821"]',
            "per-file ignores cannot grow",
        ),
        (
            'project-includes = ["plasma_global/**/*.py", "tools/**/*.py"]',
            'project-includes = ["plasma_global/**/*.py"]',
            "project-includes cannot shrink",
        ),
        (
            'min-severity = "warn"',
            'min-severity = "warn"\npermissive-ignores = true',
            "Pyrefly options are not reviewed",
        ),
        (
            "branch = true",
            'branch = true\nomit = ["plasma_global/models/*"]',
            "coverage exclusions cannot grow",
        ),
        (
            "show_missing = true",
            'show_missing = true\nexclude_also = ["raise AssertionError"]',
            "coverage exclusions cannot grow",
        ),
        (
            'root_package = "plasma_global"',
            'root_package = "tools"',
            "root_package cannot change",
        ),
        (
            '  "plasma_global.postprocess",',
            '  "plasma_global.unchecked",',
            "import-linter contracts were weakened",
        ),
        (
            'only_mutate = ["plasma_global/models/rates.py"]',
            "only_mutate = []",
            "mutation scope cannot shrink",
        ),
    ],
)
def test_quality_contract_rejects_silent_scope_reduction(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = _variant(tmp_path, old, new)

    with pytest.raises(QualityFailure, match=message):
        _check_quality_config(path)


def test_quality_contract_leaves_tool_semantics_to_each_tool(tmp_path: Path) -> None:
    path = _variant(
        tmp_path,
        'target-version = "py311"\nline-length = 88',
        'target-version = "py310"\nline-length = 100',
    )

    _check_quality_config(path)
