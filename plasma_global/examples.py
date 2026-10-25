"""Copy a working example and just its dependencies from installed resources."""

from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from plasma_global._result_common import new_output_directory
from plasma_global._yaml import load_unique_yaml


def initialize_example(name: str, output: str | Path) -> Path:
    from plasma_global.build import compile_case
    from plasma_global.input.load import load_case

    source = Path(str(files("plasma_global").joinpath("assets/examples"))).resolve()
    available = {path.stem for path in (source / "cases").glob("*.yaml")}
    if name not in available:
        raise ValueError(f"unknown example {name!r}; available: {sorted(available)}")
    case_file = source / "cases" / f"{name}.yaml"
    compiled = compile_case(load_case(case_file))
    dependencies = compiled.metadata["provenance"]["input_files"]["files"]
    directory = Path(output).resolve()
    with new_output_directory(directory) as stage:
        for raw in dependencies:
            path = Path(raw)
            if path == case_file:
                continue
            relative = path.relative_to(source)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        with case_file.open(encoding="utf-8") as stream:
            document = load_unique_yaml(stream)

        def relocate(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if (
                        key == "file"
                        and isinstance(item, str)
                        and item.startswith("../")
                    ):
                        value[key] = item[3:]
                    else:
                        relocate(item)
            elif isinstance(value, list):
                for item in value:
                    relocate(item)

        relocate(document)
        (stage / "case.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        compile_case(load_case(stage / "case.yaml"))
    return directory / "case.yaml"
