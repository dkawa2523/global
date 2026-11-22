from __future__ import annotations

import json
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

    assert case.schema_version == 5
    assert case.models.electrons.kind == "energy_balance"


def test_manual_dc_readout_matches_the_regenerated_benchmark() -> None:
    results = json.loads(
        (ROOT / "benchmarks/theory/results.json").read_text(encoding="utf-8")
    )
    comparison = results["cross_code"]["zdplaskin"]["comparisons"]
    voltage = comparison["plasma_voltage_V"]["final"]
    current = comparison["current_A"]["final"]
    field = comparison["reduced_field_Td"]["final"]
    manual = (ROOT / "docs/index.md").read_text(encoding="utf-8")

    for readout in (
        f"約{voltage:.2f} V",
        f"約{1000.0 - voltage:.2f} V",
        f"約{current:.8f} A",
        f"約{field:.5f} Td",
    ):
        assert readout in manual
