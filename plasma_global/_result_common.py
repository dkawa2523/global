"""Shared immutable names and scalar serialization for result codecs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

RESULT_FORMAT = "plasma_global_result"
RESULT_FORMAT_VERSION = 1
RESULT_H5_NAME = "result.h5"
SUMMARY_YAML_NAME = "summary.yaml"
RESULT_CSV_NAME = "result.csv"

_UNIT_MARKERS = (
    ("_J_m3_s", "J m^-3 s^-1"),
    ("_m3_s", "m^-3 s^-1"),
    ("_J_m3", "J m^-3"),
    ("_m3", "m^-3"),
    ("_m2_s", "m^2 s^-1"),
    ("_m2", "m^2"),
    ("_s_inv", "s^-1"),
    ("_eV", "eV"),
    ("_Td", "Td"),
    ("_K", "K"),
    ("_W", "W"),
    ("_V", "V"),
    ("_A", "A"),
)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [plain_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def yaml_text(value: Any) -> str:
    return yaml.safe_dump(plain_value(value), sort_keys=True, allow_unicode=True)


def series_unit(label: str) -> str:
    """Return the canonical unit encoded by a public series label."""

    if label.startswith("n["):
        return "m^-3"
    if label.startswith(("electron_energy[", "gas_internal_energy[")):
        return "J m^-3"
    if label.startswith("coverage["):
        return "1"
    head = label.partition("[")[0]
    for marker, unit in _UNIT_MARKERS:
        if head.endswith(marker):
            return unit
    return "unspecified"


__all__ = [
    "RESULT_CSV_NAME",
    "RESULT_FORMAT",
    "RESULT_FORMAT_VERSION",
    "RESULT_H5_NAME",
    "SUMMARY_YAML_NAME",
    "ensure_parent",
    "plain_value",
    "series_unit",
    "yaml_text",
]
