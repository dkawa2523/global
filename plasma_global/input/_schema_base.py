"""Shared primitives for the schema-v4 input models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _path_value(value: Any) -> Path:
    """Accept YAML's string representation without enabling other coercions."""

    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        raise TypeError("path values must be strings")
    if not value.strip():
        raise ValueError("path values must not be empty")
    return Path(value)


InputPath = Annotated[Path, BeforeValidator(_path_value)]
Identifier = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    """Strict values shared by public input and resolved configuration."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )


def _require_unique(items: list[Any], attribute: str, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        value = str(getattr(item, attribute))
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        values = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {label}: {values}")
