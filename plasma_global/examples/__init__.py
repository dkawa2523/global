"""Copy a working example and just its dependencies from installed resources."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from plasma_global._result_common import new_output_directory
from plasma_global._yaml import load_unique_yaml
from plasma_global.input._paths import map_input_paths


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


def initialize_example(name: str, output: str | Path) -> Path:
    from plasma_global.build import compile_case
    from plasma_global.input.load import load_case

    source = Path(str(files("plasma_global.examples"))).resolve()
    available = _catalog()
    if name not in available:
        raise ValueError(f"unknown example {name!r}; available: {sorted(available)}")
    case_file = available[name]
    compiled = compile_case(load_case(case_file))
    dependencies = compiled.input_sources
    directory = Path(output).resolve()
    with new_output_directory(directory) as stage:
        for path in sorted(dependencies):
            if path == case_file:
                continue
            relative = path.relative_to(source)
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(dependencies[path])
        document = compiled.case.to_dict()
        map_input_paths(
            document, lambda value: Path(value).relative_to(source).as_posix()
        )
        from plasma_global.input.load import case_from_mapping

        copied = case_from_mapping(document, base_directory=stage)
        (stage / "case.yaml").write_text(copied.to_yaml(), encoding="utf-8")
        compile_case(load_case(stage / "case.yaml"))
    return directory / "case.yaml"
