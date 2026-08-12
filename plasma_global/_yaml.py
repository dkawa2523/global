"""Safe YAML loading with duplicate-key rejection."""

from __future__ import annotations

from typing import IO, Any

import yaml


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

YamlLoadError = yaml.YAMLError


def load_unique_yaml(stream: IO[str]) -> Any:
    """Load one document through ``SafeLoader`` and reject duplicate keys."""

    loader = _UniqueKeyLoader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


__all__ = ["YamlLoadError", "load_unique_yaml"]
