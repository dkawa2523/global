from __future__ import annotations

from pathlib import Path

import pytest

from plasma_global.chemistry.compile import compile_chemistry
from plasma_global.chemistry.data import load_chemistry
from plasma_global.errors import MigrationError
from tools.importers.chemistry_v2 import convert_v2_chemistry

ROOT = Path(__file__).resolve().parents[1]
V2_FIXTURES = ROOT / "tests" / "fixtures" / "v2"


def test_crane_chemistry_migrates_to_canonical_runtime(tmp_path: Path) -> None:
    source = (
        V2_FIXTURES / "chemistry_crane_two_reaction_argon" / "chemistry_manifest.yaml"
    )
    output = convert_v2_chemistry(source, tmp_path)
    compiled = compile_chemistry(load_chemistry(output))

    assert compiled.species_ids == ("Ar", "Ar_plus")
    assert compiled.reaction_ids == (
        "CRANE_AR_ION",
        "CRANE_ARPLUS_THREE_BODY_RECOMB",
    )
    assert compiled.boundary_reactions[0].incident_species == "Ar_plus"


def test_multiphysics_chemistry_migrates_all_model_families(tmp_path: Path) -> None:
    source = V2_FIXTURES / "chemistry" / "chemistry_manifest.yaml"
    output = convert_v2_chemistry(source, tmp_path)
    compiled = compile_chemistry(load_chemistry(output))

    assert "Ar_plus" in compiled.species_ids
    assert compiled.surface_reactions
    assert compiled.cross_sections


@pytest.mark.parametrize(
    ("fixture_name", "message"),
    [
        pytest.param(
            "chemistry_argon_lxcat",
            "contains 2 concatenated axes",
            id="cross-section-segment",
        ),
        pytest.param(
            "chemistry_zdplaskin_example2",
            "cannot infer one neutral wall product",
            id="boundary-products",
        ),
    ],
)
def test_ambiguous_v2_scientific_data_requires_an_explicit_mapping(
    tmp_path: Path,
    fixture_name: str,
    message: str,
) -> None:
    source = V2_FIXTURES / fixture_name / "chemistry_manifest.yaml"

    with pytest.raises(MigrationError, match=message):
        convert_v2_chemistry(source, tmp_path)
