from __future__ import annotations

import shutil
import zipfile
from contextlib import chdir
from pathlib import Path

from setuptools.build_meta import build_wheel

import plasma_global
import plasma_global.api

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_exports_supported_operations() -> None:
    expected = ["load_case", "simulate", "write_result"]

    assert plasma_global.__all__ == expected
    assert plasma_global.api.__all__ == expected
    assert all(callable(getattr(plasma_global, name)) for name in expected)
    assert all(
        getattr(plasma_global, name) is getattr(plasma_global.api, name)
        for name in expected
    )


def test_wheel_contains_the_runtime_and_cli(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(ROOT / name, project / name)
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "plasma_global", project / "plasma_global", ignore=ignored)

    wheel_directory = tmp_path / "dist"
    wheel_directory.mkdir()
    with chdir(project):
        wheel_path = wheel_directory / build_wheel(str(wheel_directory))

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        entry_points_path = next(
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = wheel.read(entry_points_path).decode("utf-8")

    assert "plasma_global/__init__.py" in names
    assert "plasma_global/cli.py" in names
    assert "plasma-global = plasma_global.cli:main" in entry_points
    assert "quality-" not in entry_points
    assert not any(
        "__pycache__" in name or name.startswith("tests/") or name.startswith("tools/")
        for name in names
    )
