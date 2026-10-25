"""Safe YAML loading with duplicate-key rejection."""

from __future__ import annotations

import re
from typing import IO, Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


# Use YAML 1.2 scalar meanings without changing PyYAML's global loader.
# In particular, 012 is decimal and 1:20 is a string, never octal/base-60.
_UniqueKeyLoader.yaml_implicit_resolvers = {
    key: [
        (tag, pattern)
        for tag, pattern in resolvers
        if tag
        not in {
            "tag:yaml.org,2002:bool",
            "tag:yaml.org,2002:int",
            "tag:yaml.org,2002:float",
            "tag:yaml.org,2002:timestamp",
        }
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[+-]?(?:[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
    list("+-0123456789"),
)


def _construct_integer(loader: _UniqueKeyLoader, node: yaml.ScalarNode) -> int:
    value = loader.construct_scalar(node)
    digits = value.lstrip("+-")
    base = 8 if digits.startswith("0o") else (16 if digits.startswith("0x") else 10)
    return int(value, base)


_UniqueKeyLoader.add_constructor("tag:yaml.org,2002:int", _construct_integer)
_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
        r"|[0-9]+[eE][+-]?[0-9]+|\.(?:inf|Inf|INF|nan|NaN|NAN))$"
    ),
    list("+-0123456789."),
)


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
