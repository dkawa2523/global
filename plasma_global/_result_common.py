"""Shared immutable names and scalar serialization for result codecs."""

from __future__ import annotations

import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plasma_global.core.result import series_unit as series_unit

RESULT_FORMAT = "plasma_global_result"
RESULT_FORMAT_VERSION = 3
RESULT_H5_NAME = "result.h5"
SUMMARY_YAML_NAME = "summary.yaml"
RESULT_CSV_NAME = "result.csv"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def new_output_directory(directory: Path) -> Generator[Path, None, None]:
    """Stage a complete artifact set and publish it as a new directory."""

    if directory.exists():
        raise FileExistsError(
            f"Output directory already exists: {directory}; "
            "choose a new output directory"
        )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as raw:
        stage = Path(raw)
        yield stage
        if directory.exists():
            raise FileExistsError(
                f"Output directory already exists: {directory}; "
                "choose a new output directory"
            )
        stage.rename(directory)


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
