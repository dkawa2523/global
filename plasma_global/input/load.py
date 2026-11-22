"""Read the single public YAML input format and resolve file locations."""

from __future__ import annotations

from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from plasma_global._yaml import YamlLoadError, load_unique_yaml
from plasma_global.errors import CaseValidationError
from plasma_global.input.schema import CaseSpec


class CaseLoadError(CaseValidationError):
    """An input document cannot be read or resolved."""


def case_from_mapping(
    data: Mapping[str, Any], *, base_directory: str | Path
) -> CaseSpec:
    """Validate a public declaration with an explicit input-path origin."""
    try:
        return CaseSpec.model_validate(dict(data))._with_origin(Path(base_directory))
    except (ValidationError, ValueError, TypeError) as exc:
        raise CaseLoadError(f"invalid case: {exc}") from exc


def load_case(path: str | Path) -> CaseSpec:
    source = Path(path).resolve()
    try:
        content = source.read_bytes()
        with StringIO(content.decode("utf-8-sig")) as stream:
            document = load_unique_yaml(stream)
        case = CaseSpec.model_validate(document)
    except (OSError, ValidationError, ValueError, TypeError, YamlLoadError) as exc:
        raise CaseLoadError(f"invalid case {source}: {exc}") from exc
    return case._with_origin(source.parent, source, content)
