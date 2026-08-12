from __future__ import annotations

from pathlib import Path

import pytest

import tools.importers.chemistry_v2 as chemistry_v2
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
    warnings: list[str] = []
    output = convert_v2_chemistry(source, tmp_path, migration_warnings=warnings)
    compiled = compile_chemistry(load_chemistry(output))

    assert compiled.species_ids == ("Ar", "Ar_plus")
    assert compiled.reaction_ids == (
        "CRANE_AR_ION",
        "CRANE_ARPLUS_THREE_BODY_RECOMB",
    )
    assert compiled.boundary_reactions[0].incident_species == "Ar_plus"
    assert warnings == [
        "legacy chemistry had no cv_over_kb; the migration boundary made the "
        "former ideal-gas heat-capacity assumption explicit (Ar=1.5, Ar_plus=1.5)"
    ]
    expected_files = {
        "boundary_reactions.csv": (
            "id,equation,zones,surfaces\nwall_Ar_plus_neutralization,Ar_plus -> Ar,,\n"
        ),
        "chemistry.yaml": (
            "schema_version: 3\n"
            "species: species.csv\n"
            "gas_reactions: gas_reactions.csv\n"
            "boundary_reactions: boundary_reactions.csv\n"
            "surface_reactions: surface_reactions.csv\n"
            "rate_models: rate_models.yaml\n"
            "provenance:\n"
            f"  migrated_from: {source.resolve()}\n"
        ),
        "gas_reactions.csv": (
            "id,equation,rate_model,energy_loss_eV,zones,surfaces,notes\n"
            "CRANE_AR_ION,e + Ar -> e + e + Ar_plus,RM_CRANE_AR_ION,,plasma,,"
            "CRANE TwoReactionArgon ionization rate at E/N=30 Td; converted from "
            "2.1736169000623e-12 cm3/s.\n"
            "CRANE_ARPLUS_THREE_BODY_RECOMB,e + Ar_plus + Ar -> Ar + Ar,"
            "RM_CRANE_ARPLUS_THREE_BODY_RECOMB,,plasma,,CRANE TwoReactionArgon "
            "three-body recombination; converted from 1.0e-25 cm6/s.\n"
        ),
        "rate_models.yaml": (
            "rate_models:\n"
            "  RM_CRANE_AR_ION:\n"
            "    kind: constant\n"
            "    value: 2.1736169000623e-18\n"
            "  RM_CRANE_ARPLUS_THREE_BODY_RECOMB:\n"
            "    kind: constant\n"
            "    value: 1.0e-37\n"
        ),
        "species.csv": (
            "id,phase,charge,mass_amu,elements,state_tags,surfaces,display_name,"
            "cv_over_kb\n"
            "e,gas,-1,0.00054858,,electron,,e,\n"
            "Ar,gas,0,39.948,Ar:1,stable|parent,,Ar,1.5\n"
            "Ar_plus,gas,1,39.948,Ar:1,ion,,Ar+,1.5\n"
        ),
        "surface_reactions.csv": (
            "id,equation,rate_model,energy_loss_eV,zones,surfaces,notes\n"
        ),
    }
    generated_files = {
        path.relative_to(tmp_path).as_posix(): path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert generated_files == expected_files


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


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        pytest.param(
            {
                "backend": "electron_impact_xsec",
                "cross_section_id": "xs",
                "branching_yield": "0.25",
            },
            {"kind": "electron_impact", "cross_section": "xs", "branching_yield": 0.25},
            id="electron-impact",
        ),
        pytest.param(
            {"backend": "arrhenius", "A": "2", "beta": "-1", "Ea_eV": "3"},
            {"kind": "arrhenius", "A": 2.0, "beta": -1.0, "activation_eV": 3.0},
            id="arrhenius",
        ),
        pytest.param(
            {"backend": "constant", "value": "4"},
            {"kind": "constant", "value": 4.0},
            id="constant",
        ),
        pytest.param(
            {"backend": "first_order_loss", "rate_s_inv": "5"},
            {"kind": "first_order", "rate_s_inv": 5.0},
            id="first-order",
        ),
        pytest.param(
            {
                "backend": "te_power_law",
                "A": "6",
                "Tref_K": "300",
                "alpha": "-0.5",
                "electron_temperature_factor": 2.0 / 3.0,
            },
            {
                "kind": "experimental.electron_temperature_power_law",
                "A": 6.0,
                "reference_temperature_K": 300.0,
                "exponent": -0.5,
            },
            id="electron-temperature",
        ),
        pytest.param(
            {"backend": "sticking", "sticking_value": "0.1"},
            {"kind": "sticking", "value": 0.1, "coverage": {"kind": "constant"}},
            id="sticking",
        ),
        pytest.param(
            {
                "backend": "ion_assisted",
                "yield_value": "0.2",
                "threshold_eV": "1",
                "reference_energy_eV": "2",
                "energy_exponent": "3",
            },
            {
                "kind": "ion_assisted",
                "yield": 0.2,
                "threshold_eV": 1.0,
                "reference_energy_eV": 2.0,
                "exponent": 3.0,
                "coverage": {"kind": "constant"},
            },
            id="ion-assisted",
        ),
        pytest.param(
            {"backend": "desorption", "nu0_s_inv": "7", "Ea_eV": "8"},
            {"kind": "desorption", "frequency_s_inv": 7.0, "activation_eV": 8.0},
            id="desorption",
        ),
        pytest.param(
            {"backend": "langmuir_hinshelwood", "A_m2_s_inv": "9", "Ea_eV": "10"},
            {"kind": "langmuir_hinshelwood", "A_m2_s_inv": 9.0, "activation_eV": 10.0},
            id="langmuir-hinshelwood",
        ),
    ],
)
def test_converts_each_legacy_rate_family(
    tmp_path: Path, legacy: dict[str, object], expected: dict[str, object]
) -> None:
    assert (
        chemistry_v2._convert_rate_model("rate", legacy, tmp_path, tmp_path) == expected
    )


def test_tabulated_rate_conversion_publishes_canonical_table(tmp_path: Path) -> None:
    (tmp_path / "legacy.csv").write_text(
        "temperature,coefficient,ignored\n300,1e-15,a\n400,2e-15,b\n",
        encoding="utf-8",
    )

    converted = chemistry_v2._convert_rate_model(
        "table",
        {
            "backend": "tabulated_1d",
            "file": "legacy.csv",
            "x_column": "temperature",
            "value_column": "coefficient",
            "x": "gas_temperature_K",
            "bounds_policy": "clip",
        },
        tmp_path,
        tmp_path,
    )

    assert converted == {
        "kind": "tabulated_1d",
        "axis": "gas_temperature_K",
        "file": "tables/table.csv",
        "bounds": "clip",
    }
    assert (tmp_path / "tables" / "table.csv").read_text(encoding="utf-8") == (
        "x,value\n300,1e-15\n400,2e-15\n"
    )


def test_cross_section_conversion_selects_segment_and_removes_exact_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.csv"
    source.write_text(
        "energy_eV,sigma_m2\n0,0\n1,1\n0,2\n0,2\n2,3\n",
        encoding="utf-8",
    )
    destination = tmp_path / "output" / "canonical.csv"

    chemistry_v2._write_canonical_cross_section(source, destination, segment_index=1)

    assert destination.read_text(encoding="utf-8") == (
        "energy_eV,sigma_m2\n0.0,2.0\n2.0,3.0\n"
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        pytest.param(
            "energy_eV,sigma_m2\nnot-a-number,1\n",
            "must contain numeric SI values",
            id="non-numeric",
        ),
        pytest.param(
            "energy_eV,sigma_m2\nnan,1\n",
            "is non-finite",
            id="non-finite",
        ),
        pytest.param(
            "energy_eV,sigma_m2\n0,-1\n",
            "is negative",
            id="negative",
        ),
    ],
)
def test_cross_section_reader_preserves_data_validation_errors(
    tmp_path: Path, contents: str, message: str
) -> None:
    source = tmp_path / "legacy.csv"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(MigrationError) as error:
        chemistry_v2._read_cross_section_segments(source)

    assert str(error.value) == f"cross section {source}:2 {message}"


@pytest.mark.parametrize(
    ("segments", "segment_index", "message"),
    [
        pytest.param(
            [[(0.0, 1.0), (1.0, 2.0)], [(0.0, 3.0), (1.0, 4.0)]],
            None,
            "contains 2 concatenated axes; select one explicitly with "
            "--cross-section-segments",
            id="ambiguous-segment",
        ),
        pytest.param(
            [[(0.0, 1.0), (1.0, 2.0)]],
            1,
            "segment 1 is outside 0..0",
            id="segment-out-of-range",
        ),
        pytest.param(
            [[(0.0, 1.0), (0.0, 2.0)]],
            None,
            "has conflicting values at 0 eV",
            id="conflicting-duplicate",
        ),
        pytest.param(
            [[(0.0, 1.0)]],
            None,
            "needs at least two unique nodes",
            id="insufficient-nodes",
        ),
    ],
)
def test_cross_section_canonicalization_preserves_selection_errors(
    tmp_path: Path,
    segments: list[list[tuple[float, float]]],
    segment_index: int | None,
    message: str,
) -> None:
    source = tmp_path / "legacy.csv"

    with pytest.raises(MigrationError) as error:
        chemistry_v2._canonical_cross_section_rows(source, segments, segment_index)

    assert str(error.value) == f"cross section {source} {message}"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        pytest.param(
            {"cv_over_kb": " 2.75 "},
            ("2.75", None),
            id="explicit",
        ),
        pytest.param(
            {"phase": "surface", "state_tags": "", "elements": "site:1"},
            ("", None),
            id="non-gas",
        ),
        pytest.param(
            {"phase": "gas", "state_tags": "electron", "elements": ""},
            ("", None),
            id="electron",
        ),
        pytest.param(
            {"phase": "gas", "state_tags": "", "elements": "Ar:1"},
            (1.5, 1.5),
            id="monatomic",
        ),
        pytest.param(
            {"phase": "gas", "state_tags": "", "elements": "O:2"},
            (2.5, 2.5),
            id="diatomic",
        ),
        pytest.param(
            {"phase": "gas", "state_tags": "", "elements": "H:2;O:1"},
            (3.0, 3.0),
            id="polyatomic",
        ),
    ],
)
def test_legacy_cv_inference_preserves_species_classification(
    row: dict[str, str], expected: tuple[str | float, float | None]
) -> None:
    row.setdefault("canonical_id", "species")

    assert chemistry_v2._legacy_cv_over_kb(row) == expected


@pytest.mark.parametrize(
    ("elements", "message"),
    [
        pytest.param(
            "site:1",
            "has an unusable legacy element formula",
            id="site-only",
        ),
        pytest.param(
            "",
            "needs an explicit cv_over_kb for v3",
            id="missing-elements",
        ),
    ],
)
def test_legacy_cv_inference_preserves_invalid_formula_errors(
    elements: str, message: str
) -> None:
    row = {
        "canonical_id": "species",
        "phase": "gas",
        "state_tags": "",
        "elements": elements,
    }

    with pytest.raises(MigrationError) as error:
        chemistry_v2._legacy_cv_over_kb(row)

    assert str(error.value) == f"species species {message}"
