"""Load one schema-v3 case document, with one optional include per file."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from plasma_global.errors import CaseValidationError
from plasma_global.input.schema import CaseSpec


class CaseLoadError(CaseValidationError):
    """A case document could not be read, merged, or validated."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=_UniqueKeyLoader)
    except OSError as exc:
        raise CaseLoadError(f"cannot read case input {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CaseLoadError(f"invalid YAML in {path}: {exc}") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise CaseLoadError(f"case input {path} must contain a YAML mapping")
    return document


def _merge(base: Any, override: Any) -> Any:
    """Merge mappings recursively; every other value replaces the base."""

    if isinstance(base, dict) and isinstance(override, dict):
        merged = deepcopy(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = _merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override)


def _absolute_path(value: Any, declaring_file: Path, field: str) -> Path:
    if not isinstance(value, str):
        raise CaseLoadError(
            f"{declaring_file}: {field} must be a path string, got "
            f"{type(value).__name__}"
        )
    if not value.strip():
        raise CaseLoadError(f"{declaring_file}: {field} must not be empty")
    path = Path(value)
    if not path.is_absolute():
        path = declaring_file.parent / path
    # Deliberately do not call expandvars() or expanduser().
    return path.resolve(strict=False)


def _mapping_at(document: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = document.get(key)
    return value if isinstance(value, dict) else None


def _resolve_declared_paths(
    document: dict[str, Any], declaring_file: Path
) -> dict[str, Any]:
    """Resolve only fields whose v3 schema declares path semantics."""

    resolved = deepcopy(document)

    chemistry = _mapping_at(resolved, "chemistry")
    if chemistry is not None and "manifest" in chemistry:
        chemistry["manifest"] = _absolute_path(
            chemistry["manifest"], declaring_file, "chemistry.manifest"
        )

    models = _mapping_at(resolved, "models")
    if models is not None:
        for section_name in ("electrons", "electron_density"):
            section = _mapping_at(models, section_name)
            if section is not None and "file" in section:
                section["file"] = _absolute_path(
                    section["file"],
                    declaring_file,
                    f"models.{section_name}.file",
                )

    reactor = _mapping_at(resolved, "reactor")
    ports = reactor.get("power_ports") if reactor is not None else None
    if isinstance(ports, list):
        for index, port in enumerate(ports):
            if not isinstance(port, dict):
                continue
            model = port.get("model")
            if isinstance(model, dict) and "file" in model:
                model["file"] = _absolute_path(
                    model["file"],
                    declaring_file,
                    f"reactor.power_ports[{index}].model.file",
                )

    recipe = _mapping_at(resolved, "recipe")
    steps = recipe.get("steps") if recipe is not None else None
    if isinstance(steps, list):
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            commands = step.get("commands")
            if not isinstance(commands, dict):
                continue
            power_commands = commands.get("power_ports")
            if not isinstance(power_commands, dict):
                continue
            for port_id, command in power_commands.items():
                if isinstance(command, dict) and "file" in command:
                    command["file"] = _absolute_path(
                        command["file"],
                        declaring_file,
                        (
                            f"recipe.steps[{step_index}].commands.power_ports"
                            f"[{port_id!r}].file"
                        ),
                    )

    return resolved


def _load_document(
    path: Path, stack: tuple[Path, ...]
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    source = path.resolve(strict=False)
    if source in stack:
        chain = " -> ".join(str(item) for item in (*stack, source))
        raise CaseLoadError(f"include cycle detected: {chain}")

    raw = _read_yaml(source)
    include = raw.pop("include", None)
    if include is not None and not isinstance(include, str):
        raise CaseLoadError(
            f"{source}: include must be one path string; lists and mappings are not allowed"
        )

    local = _resolve_declared_paths(raw, source)
    if include is None:
        return local, ()
    if not include.strip():
        raise CaseLoadError(f"{source}: include must not be empty")

    include_path = Path(include)
    if not include_path.is_absolute():
        include_path = source.parent / include_path
    include_path = include_path.resolve(strict=False)
    base, inherited = _load_document(include_path, (*stack, source))
    return _merge(base, local), (*inherited, include_path)


def load_case(path: str | Path) -> CaseSpec:
    """Load and validate a schema-v3 case.

    Relative paths are converted to absolute paths before include merging, so
    an inherited path always retains the directory of the file that declared
    it.  Environment variables and ``~`` are intentionally left literal.
    """

    if not isinstance(path, (str, Path)):
        raise TypeError("load_case path must be a string or pathlib.Path")
    source = Path(path).resolve(strict=False)
    merged, included_files = _load_document(source, ())
    try:
        case = CaseSpec.model_validate(merged)
    except ValidationError as exc:
        raise CaseLoadError(f"invalid schema-v3 case {source}:\n{exc}") from exc
    return case._with_source(source, included_files)


__all__ = ["CaseLoadError", "load_case"]
