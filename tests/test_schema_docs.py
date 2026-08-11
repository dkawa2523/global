from __future__ import annotations

from pathlib import Path

import yaml

from plasma_global.input.schema import CaseSpec
from tools.generate_schema_docs import render_schema_documents

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "docs" / "generated"


def test_generated_schema_documents_are_reproducible() -> None:
    rendered = render_schema_documents()

    assert set(rendered) == {"case.schema.json", "minimal_case.yaml"}
    for name, expected in rendered.items():
        assert (GENERATED / name).read_text(encoding="utf-8") == expected


def test_generated_minimal_case_matches_the_public_schema() -> None:
    document = yaml.safe_load(
        (GENERATED / "minimal_case.yaml").read_text(encoding="utf-8")
    )

    case = CaseSpec.model_validate(document)

    assert case.schema_version == 3
    assert case.models.electron_closure.kind == "electron_energy"
