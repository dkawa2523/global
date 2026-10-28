"""Copy a working example and just its dependencies from installed resources."""

from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from plasma_global._result_common import new_output_directory
from plasma_global._yaml import load_unique_yaml


def _catalog() -> dict[str, Path]:
    source = Path(str(files("plasma_global.examples"))).resolve()
    cases = source / "cases"
    return {
        path.relative_to(cases).with_suffix("").as_posix(): path
        for path in sorted(cases.rglob("*.yaml"))
    }


def list_examples() -> dict[str, str]:
    """List installed case IDs and their descriptions without compiling physics."""
    result = {}
    for name, path in _catalog().items():
        with path.open(encoding="utf-8") as stream:
            result[name] = str(load_unique_yaml(stream)["case"].get("description", ""))
    return result


def _relocate_files(
    value: Any, base: Path, source: Path, dependencies: set[Path]
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "file" and isinstance(item, str):
                dependency = (base / item).resolve()
                if dependency in dependencies:
                    value[key] = dependency.relative_to(source).as_posix()
            else:
                _relocate_files(item, base, source, dependencies)
    elif isinstance(value, list):
        for item in value:
            _relocate_files(item, base, source, dependencies)


def initialize_example(name: str, output: str | Path) -> Path:
    from plasma_global.build import compile_case
    from plasma_global.input.load import load_case

    source = Path(str(files("plasma_global.examples"))).resolve()
    available = _catalog()
    if name not in available:
        raise ValueError(f"unknown example {name!r}; available: {sorted(available)}")
    case_file = available[name]
    compiled = compile_case(load_case(case_file))
    dependencies = {
        Path(path) for path in compiled.metadata["provenance"]["input_files"]["files"]
    }
    directory = Path(output).resolve()
    with new_output_directory(directory) as stage:
        for path in sorted(dependencies):
            if path == case_file:
                continue
            relative = path.relative_to(source)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        with case_file.open(encoding="utf-8") as stream:
            document = load_unique_yaml(stream)

        _relocate_files(document, case_file.parent, source, dependencies)
        (stage / "case.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        compile_case(load_case(stage / "case.yaml"))
    return directory / "case.yaml"
