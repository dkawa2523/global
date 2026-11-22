"""Visit declared input references without interpreting provenance annotations."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


def _record_file(record: Any, transform: Callable[[str], str]) -> None:
    if isinstance(record, dict) and record.get("file") is not None:
        value = record["file"]
        if not isinstance(value, (str, Path)):
            raise ValueError("file references must be strings or paths")
        record["file"] = transform(str(value))


def _reaction_paths(data: dict[str, Any], transform: Callable[[str], str]) -> None:
    for family in ("gas_reactions", "surface_reactions"):
        reactions = data.get(family, [])
        if isinstance(reactions, list):
            for reaction in reactions:
                if isinstance(reaction, dict):
                    _record_file(reaction.get("rate"), transform)


def _chemistry_paths(data: dict[str, Any], transform: Callable[[str], str]) -> None:
    if "file" in data:
        _record_file(data, transform)
        return
    for name in (
        "species",
        "gas_reactions",
        "boundary_reactions",
        "surface_reactions",
        "rate_models",
        "cross_sections",
    ):
        if isinstance(data.get(name), str):
            data[name] = transform(data[name])
    models = data.get("rate_models", {})
    if isinstance(models, dict):
        for model in models.values():
            _record_file(model, transform)
    _reaction_paths(data, transform)
    curves = data.get("cross_sections", [])
    if isinstance(curves, list):
        for curve in curves:
            _record_file(curve, transform)


def map_input_paths(document: dict[str, Any], transform: Callable[[str], str]) -> None:
    """Transform only file fields belonging to the public case input."""

    _chemistry_paths(document["chemistry"], transform)
    models = document["models"]
    _record_file(models["electrons"]["coefficients"], transform)
    _record_file(models.get("electron_density"), transform)
    for port in document["reactor"].get("power_ports", {}).values():
        _record_file(port["model"], transform)
    for step in document["run"].get("steps") or ():
        for command in step.get("commands", {}).get("power_ports", {}).values():
            _record_file(command, transform)


def absolute_input_paths(document: dict[str, Any], base: Path) -> None:
    map_input_paths(document, lambda value: str((base / value).resolve()))
