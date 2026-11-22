"""Shared primitives for the schema-v5 input models."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from plasma_global._identifiers import IDENTIFIER_PATTERN


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
Identifier = Annotated[str, Field(min_length=1, pattern=IDENTIFIER_PATTERN)]
NonemptyText = Annotated[str, Field(min_length=1, pattern=r".*\S.*")]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    """Strict values shared by public input."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
    )
