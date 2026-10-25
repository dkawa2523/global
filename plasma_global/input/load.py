"""Read the single public YAML input format and resolve file locations."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from plasma_global._yaml import YamlLoadError, load_unique_yaml
from plasma_global.errors import CaseValidationError
from plasma_global.input._resolved_case import ResolvedCase
from plasma_global.input.schema import CaseSpec


class CaseLoadError(CaseValidationError):
    """An input document cannot be read or resolved."""


def load_case(path: str | Path) -> ResolvedCase:
    source = Path(path).resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            document = load_unique_yaml(stream)
        case = CaseSpec.model_validate(document).resolve(source.parent)
    except (OSError, ValidationError, ValueError, TypeError, YamlLoadError) as exc:
        raise CaseLoadError(f"invalid case {source}: {exc}") from exc
    return case._with_source(source, ())
