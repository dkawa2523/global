from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import plasma_global.chemistry.compile as compile_module
from plasma_global._input_sources import InputSourceStore
from plasma_global.chemistry._contracts import normalized_cross_section_curve
from plasma_global.chemistry._data_csv import load_reactions, load_species
from plasma_global.chemistry._data_yaml import load_definition
from plasma_global.chemistry._maxwell import compile_maxwell_rate
from plasma_global.chemistry.compile import compile_chemistry
from plasma_global.chemistry.data import (
    ChemistryData,
    CrossSectionData,
    RateModelData,
    ReactionData,
    SpeciesData,
    load_chemistry,
)
from plasma_global.errors import ChemistryError, ModelDomainError


def _mechanism(
    tmp_path: Path,
    *,
    equation: str = "e + Ar -> Ar_plus + e + e",
    bad_curve: bool = False,
) -> Path:
    (tmp_path / "species.csv").write_text(
        "id,phase,charge,mass_amu,elements,state_tags\n"
        "e,gas,-1,0.00054858,,electron\n"
        "Ar,gas,0,39.948,Ar:1,\n"
        "Ar_plus,gas,1,39.948,Ar:1,ion\n",
        encoding="utf-8",
    )
    (tmp_path / "gas.csv").write_text(
        f"id,equation,rate_model\nionize,{equation},electron_impact\n",
        encoding="utf-8",
    )
    (tmp_path / "boundary.csv").write_text(
        "id,equation,zones\nneutralize,Ar_plus -> Ar,plasma\n",
        encoding="utf-8",
    )
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n"
        "  electron_impact:\n"
        "    kind: electron_impact\n"
        "    cross_section: ionization\n"
        "    branching_yield: 1.0\n",
        encoding="utf-8",
    )
    curve = (
        "energy_eV,sigma_m2\n1,0\n10,2e-20\n5,1e-20\n"
        if bad_curve
        else "energy_eV,sigma_m2\n1,0\n5,1e-20\n10,2e-20\n"
    )
    (tmp_path / "xs.csv").write_text(curve, encoding="utf-8")
    (tmp_path / "cross_sections.yaml").write_text(
        "cross_sections:\n"
        "  - id: ionization\n"
        "    kind: ionization\n"
        "    target: Ar\n"
        "    threshold_eV: 5\n"
        "    electron_energy_transfer_eV: -5\n"
        "    file: xs.csv\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "chemistry.yaml"
    manifest.write_text(
        "schema_version: 4\n"
        "species: species.csv\n"
        "gas_reactions: gas.csv\n"
        "boundary_reactions: boundary.csv\n"
        "rate_models: rates.yaml\n"
        "cross_sections: cross_sections.yaml\n",
        encoding="utf-8",
    )
    return manifest


def test_compiles_balanced_chemistry_to_readonly_arrays(tmp_path: Path) -> None:
    chemistry = compile_chemistry(load_chemistry(_mechanism(tmp_path)))

    assert chemistry.species_ids == ("Ar", "Ar_plus")
    np.testing.assert_array_equal(chemistry.stoichiometry, [[-1.0, 1.0]])
    np.testing.assert_array_equal(chemistry.reactant_orders, [[1.0, 0.0]])
    np.testing.assert_array_equal(chemistry.electron_orders, [1.0])
    assert chemistry.electron_energy_transfer_eV.tolist() == [-5.0]
    assert chemistry.boundary_reactions[0].products == {"Ar": 1.0}
    assert chemistry.boundary_reactions[0].wall_charge_per_event == 1.0
    assert chemistry.jacobian_species_pattern.tolist() == [
        [True, True],
        [True, True],
    ]
    with pytest.raises(ValueError, match="read-only"):
        chemistry.stoichiometry[0, 0] = 0.0


def test_signed_electron_energy_transfer_is_loaded_and_compiled(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "cross_sections.yaml").write_text(
        "cross_sections:\n"
        "  - id: ionization\n"
        "    kind: ionization\n"
        "    target: Ar\n"
        "    threshold_eV: 5\n"
        "    electron_energy_transfer_eV: -5\n"
        "    file: xs.csv\n",
        encoding="utf-8",
    )

    loaded = load_chemistry(manifest)
    chemistry = compile_chemistry(loaded)

    assert loaded.cross_sections["ionization"].electron_energy_transfer_eV == -5.0
    np.testing.assert_array_equal(chemistry.electron_energy_transfer_eV, [-5.0])


def _species_validation_chemistry(
    *, electron: SpeciesData, cross_section_target: str
) -> ChemistryData:
    cross_section = CrossSectionData(
        id="momentum",
        kind="momentum_transfer",
        target=cross_section_target,
        threshold_eV=0.0,
        electron_energy_transfer_eV=0.0,
        energy_eV=np.array([0.0, 1.0]),
        sigma_m2=np.array([1.0e-20, 1.0e-20]),
    )
    return ChemistryData(
        source=Path("species-validation.yaml"),
        species=(
            electron,
            SpeciesData("Ar", "gas", 0, 39.948, {"Ar": 1.0}),
            SpeciesData("site", "surface", 0, 0.0, {"site": 1.0}),
        ),
        gas_reactions=(),
        boundary_reactions=(),
        surface_reactions=(),
        rate_models={},
        cross_sections={cross_section.id: cross_section},
    )


def test_species_validation_reports_invalid_electron_before_cross_section() -> None:
    chemistry = _species_validation_chemistry(
        electron=SpeciesData("e", "surface", -1, 0.00054858, {}),
        cross_section_target="missing",
    )

    with pytest.raises(
        ChemistryError,
        match="exactly one gas electron species 'e' with charge -1",
    ):
        compile_chemistry(chemistry)


@pytest.mark.parametrize("target", ["missing", "e", "site"])
def test_species_validation_rejects_nonheavy_cross_section_target(target: str) -> None:
    chemistry = _species_validation_chemistry(
        electron=SpeciesData("e", "gas", -1, 0.00054858, {}),
        cross_section_target=target,
    )

    with pytest.raises(
        ChemistryError,
        match=(
            rf"cross section momentum target {target!r} must be a declared heavy "
            "gas species"
        ),
    ):
        compile_chemistry(chemistry)


def _programmatic_cross_section(**changes: object) -> CrossSectionData:
    baseline = CrossSectionData(
        id="programmatic",
        kind="momentum_transfer",
        target="Ar",
        threshold_eV=0.0,
        electron_energy_transfer_eV=0.0,
        energy_eV=np.array([0.0, 1.0, 10.0]),
        sigma_m2=np.array([0.0, 1.0e-20, 2.0e-20]),
    )
    return replace(baseline, **changes)


def _chemistry_with_cross_section(cross_section: CrossSectionData) -> ChemistryData:
    baseline = _species_validation_chemistry(
        electron=SpeciesData("e", "gas", -1, 0.00054858, {}),
        cross_section_target="Ar",
    )
    return replace(baseline, cross_sections={"candidate": cross_section})


@pytest.mark.parametrize(
    ("energy_eV", "sigma_m2", "message"),
    [
        (
            np.array([0.0, 1.0, 10.0]),
            np.array([0.0, -1.0e-20, 1.0e-20]),
            "negative cross sections",
        ),
        (
            np.array([0.0, 10.0, 1.0]),
            np.array([0.0, 1.0e-20, 2.0e-20]),
            "strictly increasing",
        ),
        (
            np.array([0.0, 1.0, 1.0]),
            np.array([0.0, 1.0e-20, 2.0e-20]),
            "strictly increasing",
        ),
        (
            np.array([0.0, 1.0, 10.0]),
            np.array([0.0, np.nan, 1.0e-20]),
            "non-finite",
        ),
        (
            np.array([0.0, 1.0, 10.0]),
            np.array([0.0, 1.0e-20]),
            "equal one-dimensional arrays",
        ),
        (
            np.array([0, 10**10000], dtype=object),
            np.array([0.0, 1.0e-20]),
            "numeric arrays",
        ),
        (
            np.array(["0", "1"]),
            np.array(["0", "1"]),
            "numeric arrays",
        ),
        (
            np.array([False, True]),
            np.array([False, True]),
            "numeric arrays",
        ),
    ],
    ids=(
        "negative-sigma",
        "unordered-energy",
        "duplicate-energy",
        "nonfinite-sigma",
        "shape-mismatch",
        "overflowing-energy",
        "numeric-strings",
        "booleans",
    ),
)
def test_programmatic_cross_section_curve_is_rejected_at_every_public_entry(
    energy_eV: np.ndarray, sigma_m2: np.ndarray, message: str
) -> None:
    cross_section = _programmatic_cross_section(
        energy_eV=energy_eV,
        sigma_m2=sigma_m2,
    )

    with pytest.raises(ChemistryError, match=message):
        compile_maxwell_rate(cross_section)
    with pytest.raises(ChemistryError, match=message):
        compile_chemistry(_chemistry_with_cross_section(cross_section))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "unsupported"}, "unsupported kind"),
        ({"kind": []}, "unsupported kind"),
        ({"threshold_eV": -1.0}, "invalid threshold"),
        ({"threshold_eV": np.nan}, "invalid threshold"),
        ({"threshold_eV": "1.0"}, "invalid threshold"),
        ({"threshold_eV": 10**10000}, "invalid threshold"),
    ],
    ids=(
        "kind",
        "unhashable-kind",
        "negative-threshold",
        "nan-threshold",
        "string-threshold",
        "overflowing-threshold",
    ),
)
def test_programmatic_cross_section_metadata_is_rejected_at_every_public_entry(
    changes: dict[str, object], message: str
) -> None:
    cross_section = _programmatic_cross_section(**changes)

    with pytest.raises(ChemistryError, match=message):
        compile_maxwell_rate(cross_section)
    with pytest.raises(ChemistryError, match=message):
        compile_chemistry(_chemistry_with_cross_section(cross_section))


@pytest.mark.parametrize(
    ("field", "yaml_value", "message"),
    [
        ("id", "[ionization]", "id must be a non-empty string"),
        ("kind", "[ionization]", "kind must be a non-empty string"),
        ("target", "{species: Ar}", "target must be a non-empty string"),
        ("threshold_eV", "[5]", "invalid threshold"),
        ("file", "[xs.csv]", "file must be a non-empty path"),
    ],
)
def test_manifest_cross_section_metadata_rejects_non_scalar_values(
    tmp_path: Path,
    field: str,
    yaml_value: str,
    message: str,
) -> None:
    manifest = _mechanism(tmp_path)
    path = tmp_path / "cross_sections.yaml"
    text = path.read_text(encoding="utf-8")
    indentation = "  - " if field == "id" else "    "
    current_line = next(
        line for line in text.splitlines() if line.startswith(f"{indentation}{field}:")
    )
    path.write_text(
        text.replace(current_line, f"{indentation}{field}: {yaml_value}"),
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match=message):
        load_chemistry(manifest)


def test_maxwellian_cross_section_rate_is_positive(tmp_path: Path) -> None:
    chemistry = compile_chemistry(load_chemistry(_mechanism(tmp_path)))
    context = SimpleNamespace(
        mean_energy_eV=0.5,
        electron_temperature_eV=1.0 / 3.0,
        reduced_field_Td=10.0,
        gas_temperature_K=300.0,
    )

    assert chemistry.rate_evaluators[0](context) > 0.0


def test_maxwellian_rate_rejects_unresolved_cross_section_tail(
    tmp_path: Path,
) -> None:
    chemistry = compile_chemistry(load_chemistry(_mechanism(tmp_path)))
    context = SimpleNamespace(
        mean_energy_eV=7.0,
        electron_temperature_eV=14.0 / 3.0,
        reduced_field_Td=10.0,
        gas_temperature_K=300.0,
    )

    with pytest.raises(ModelDomainError, match="distribution-tail range"):
        chemistry.rate_evaluators[0](context)


def test_maxwellian_rejects_missing_low_energy_momentum_support() -> None:
    sigma_m2 = 1.0e-20
    cross_section = CrossSectionData(
        id="constant-truncated-at-low-energy",
        kind="momentum_transfer",
        target="Ar",
        threshold_eV=0.0,
        electron_energy_transfer_eV=0.0,
        energy_eV=np.geomspace(1.0e-3, 10.0, 4001),
        sigma_m2=np.full(4001, sigma_m2),
    )

    with pytest.raises(ChemistryError, match="must explicitly cover 0 eV"):
        compile_maxwell_rate(cross_section)


def test_onset_curve_normalizer_is_readonly_and_zero_below_threshold() -> None:
    source_energy = np.array([0.0, 20.0])
    source_sigma = np.array([0.0, 1.0e-20])
    cross_section = CrossSectionData(
        id="gapped-excitation",
        kind="excitation",
        target="Ar",
        threshold_eV=10.0,
        electron_energy_transfer_eV=-(10.0),
        energy_eV=source_energy,
        sigma_m2=source_sigma,
    )

    energy, sigma = normalized_cross_section_curve(cross_section)

    np.testing.assert_array_equal(source_energy, [0.0, 20.0])
    np.testing.assert_array_equal(source_sigma, [0.0, 1.0e-20])
    np.testing.assert_array_equal(
        np.interp([5.0, 10.0, 15.0], energy, sigma),
        [0.0, 0.0, 0.5e-20],
    )
    assert not energy.flags.writeable
    assert not sigma.flags.writeable


def test_rejects_unbalanced_reaction(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path, equation="e + Ar -> Ar_plus + Ar_plus + e")
    with pytest.raises(ChemistryError, match="conserve"):
        compile_chemistry(load_chemistry(manifest))


@pytest.mark.parametrize(
    "equation",
    [
        "2 e + Ar -> Ar_plus + e + e + e",
        "e + 2 Ar -> Ar_plus + Ar + e + e",
        "e + Ar + Ar_plus -> Ar_plus + Ar_plus + e + e",
    ],
)
def test_electron_impact_requires_one_electron_target_collision(
    tmp_path: Path, equation: str
) -> None:
    manifest = _mechanism(tmp_path, equation=equation)

    with pytest.raises(
        ChemistryError,
        match="electron-impact reaction ionize must have exactly one unit electron",
    ):
        compile_chemistry(load_chemistry(manifest))


@pytest.mark.parametrize(
    "equation",
    [
        "2 Ar -> Ar_plus + Ar + e",
        "Ar + Ar_plus -> 2 Ar_plus + e",
    ],
)
def test_first_order_rate_requires_one_unit_reactant(
    tmp_path: Path, equation: str
) -> None:
    manifest = _mechanism(tmp_path, equation=equation)
    (tmp_path / "gas.csv").write_text(
        f"id,equation,rate_model\nionize,{equation},decay\n",
        encoding="utf-8",
    )
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n  decay:\n    kind: first_order\n    rate_s_inv: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ChemistryError,
        match="first-order reaction ionize must have exactly one unit reactant",
    ):
        compile_chemistry(load_chemistry(manifest))


def test_first_order_rate_accepts_one_unit_electron_reactant(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path, equation="e -> e")
    (tmp_path / "gas.csv").write_text(
        "id,equation,rate_model\nrelax,e -> e,relaxation\n",
        encoding="utf-8",
    )
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n  relaxation:\n    kind: first_order\n    rate_s_inv: 1.0\n",
        encoding="utf-8",
    )

    chemistry = compile_chemistry(load_chemistry(manifest))

    np.testing.assert_array_equal(chemistry.electron_orders, [1.0])
    np.testing.assert_array_equal(chemistry.reactant_orders, [[0.0, 0.0]])


def test_rejects_electron_energy_transfer_without_electron_topology(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "gas.csv").write_text(
        "id,equation,rate_model,electron_energy_transfer_eV\n"
        "neutral,Ar -> Ar,decay,5\n",
        encoding="utf-8",
    )
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n  decay:\n    kind: first_order\n    rate_s_inv: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="does not contain an electron"):
        compile_chemistry(load_chemistry(manifest))


@pytest.mark.parametrize(
    ("file_name", "manifest_entry", "contents"),
    [
        (
            "boundary.csv",
            "",
            "id,equation,electron_energy_transfer_eV\nneutralize,Ar_plus -> Ar,-5\n",
        ),
        (
            "surface.csv",
            "surface_reactions: surface.csv\n",
            "id,equation,rate_model,electron_energy_transfer_eV\n"
            "surface_loss,Ar -> Ar,unused,-5\n",
        ),
    ],
)
def test_rejects_unused_non_gas_electron_energy_fields(
    tmp_path: Path,
    file_name: str,
    manifest_entry: str,
    contents: str,
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / file_name).write_text(contents, encoding="utf-8")
    if manifest_entry:
        manifest.write_text(
            manifest.read_text(encoding="utf-8") + manifest_entry,
            encoding="utf-8",
        )

    with pytest.raises(ChemistryError, match="do not support electron-energy"):
        load_chemistry(manifest)


def test_programmatic_boundary_allows_zero_electron_energy_transfer(
    tmp_path: Path,
) -> None:
    loaded = load_chemistry(_mechanism(tmp_path))
    boundary = replace(loaded.boundary_reactions[0], electron_energy_transfer_eV=0.0)

    compiled = compile_chemistry(replace(loaded, boundary_reactions=(boundary,)))

    assert compiled.boundary_reactions[0].id == boundary.id


@pytest.mark.parametrize(
    "field_name",
    ["electron_energy_transfer_eV", "gas_heating_eV"],
)
def test_boundary_loader_allows_effectively_zero_energy_fields(
    tmp_path: Path, field_name: str
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "boundary.csv").write_text(
        f"id,equation,{field_name}\nneutralize,Ar_plus -> Ar,0\n",
        encoding="utf-8",
    )

    compiled = compile_chemistry(load_chemistry(manifest))

    assert compiled.boundary_reactions[0].id == "neutralize"


def test_boundary_loader_rejects_nonzero_gas_heating(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "boundary.csv").write_text(
        "id,equation,gas_heating_eV\nneutralize,Ar_plus -> Ar,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="do not support gas_heating_eV"):
        load_chemistry(manifest)


def test_wall_absorption_is_explicit_in_a_boundary_csv(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path)
    path = tmp_path / "boundary.csv"
    path.write_text("id,equation\nabsorb,Ar_plus -> 0\n", encoding="utf-8")
    reaction = compile_chemistry(load_chemistry(manifest)).boundary_reactions[0]
    assert not reaction.products
    assert reaction.wall_charge_per_event == 1.0

    path.write_text("id,equation\nabsorb,Ar_plus ->\n", encoding="utf-8")
    with pytest.raises(ChemistryError, match="write '-> 0'"):
        load_chemistry(manifest)


def test_surface_gas_heat_uses_the_same_sign_in_yaml_and_csv(tmp_path: Path) -> None:
    table = tmp_path / "surface.csv"
    table.write_text(
        "id,equation,rate_model,gas_heating_eV\n"
        "desorb,adsorbed -> Ar + site,desorption,-0.25\n",
        encoding="utf-8",
    )
    source = tmp_path / "chemistry.yaml"
    definitions = {
        "species": [{"id": "Ar", "mass_amu": 39.948}],
        "rate_models": {
            "desorption": {
                "kind": "desorption",
                "frequency_s_inv": 1.0,
                "activation_eV": 0.0,
            }
        },
    }
    inline = load_definition(
        {
            **definitions,
            "surface_reactions": [
                {
                    "id": "desorb",
                    "equation": "adsorbed -> Ar + site",
                    "rate": "desorption",
                    "gas_heating_eV": -0.25,
                }
            ],
        },
        source,
        tmp_path,
    )
    csv = load_definition(
        {**definitions, "surface_reactions": table.name}, source, tmp_path
    )
    assert inline.surface_reactions == csv.surface_reactions
    assert csv.surface_reactions[0].gas_heating_eV == -0.25


def _generic_rate_dimension_chemistry(
    reactions: tuple[ReactionData, ...], rate_model: RateModelData
) -> ChemistryData:
    return ChemistryData(
        source=Path("rate-dimension.yaml"),
        species=(
            SpeciesData("e", "gas", -1, 0.00054858, {}),
            SpeciesData("A", "gas", 0, 10.0, {"X": 1.0}),
            SpeciesData("B", "gas", 0, 10.0, {"X": 1.0}),
            SpeciesData("A_plus", "gas", 1, 10.0, {"X": 1.0}),
            SpeciesData("AB", "gas", 0, 20.0, {"X": 2.0}),
        ),
        gas_reactions=reactions,
        boundary_reactions=(),
        surface_reactions=(),
        rate_models={rate_model.id: rate_model},
        cross_sections={},
    )


def test_rejects_element_balanced_reaction_with_inconsistent_heavy_mass() -> None:
    rate_model = RateModelData("convert", "first_order", {"rate_s_inv": 1.0})
    chemistry = _generic_rate_dimension_chemistry(
        (ReactionData("convert", {"A": 1.0}, {"B": 1.0}, "convert", None),),
        rate_model,
    )
    chemistry = replace(
        chemistry,
        species=tuple(
            replace(species, mass_amu=100.0) if species.id == "B" else species
            for species in chemistry.species
        ),
    )

    with pytest.raises(ChemistryError, match="does not conserve heavy-species mass"):
        compile_chemistry(chemistry)


def test_heavy_mass_balance_allows_normal_atomic_weight_rounding() -> None:
    rate_model = RateModelData("split", "first_order", {"rate_s_inv": 1.0})
    chemistry = ChemistryData(
        source=Path("rounded-masses.yaml"),
        species=(
            SpeciesData("e", "gas", -1, 0.00054858, {}),
            SpeciesData("CF4", "gas", 0, 88.0043, {"C": 1.0, "F": 4.0}),
            SpeciesData("CF3", "gas", 0, 69.0037, {"C": 1.0, "F": 3.0}),
            SpeciesData("F", "gas", 0, 18.9984, {"F": 1.0}),
        ),
        gas_reactions=(
            ReactionData(
                "split",
                {"CF4": 1.0},
                {"CF3": 1.0, "F": 1.0},
                rate_model.id,
                None,
            ),
        ),
        boundary_reactions=(),
        surface_reactions=(),
        rate_models={rate_model.id: rate_model},
        cross_sections={},
    )

    compiled = compile_chemistry(chemistry)

    assert compiled.reaction_ids == ("split",)


def test_mass_balance_accepts_equal_neutral_ion_mass_convention() -> None:
    rate_model = RateModelData("ionize", "first_order", {"rate_s_inv": 1.0})
    chemistry = _generic_rate_dimension_chemistry(
        (
            ReactionData(
                "ionize",
                {"A": 1.0},
                {"A_plus": 1.0, "e": 1.0},
                rate_model.id,
                None,
            ),
        ),
        rate_model,
    )
    chemistry = replace(
        chemistry,
        species=tuple(
            replace(species, mass_amu=1.0) if species.id in {"A", "A_plus"} else species
            for species in chemistry.species
        ),
    )

    compiled = compile_chemistry(chemistry)

    assert compiled.reaction_ids == ("ionize",)


def test_mass_balance_accepts_charge_corrected_gas_ionization_mass() -> None:
    rate_model = RateModelData(
        "ionize_h", "constant", {"value": 1.0, "overall_order": 2.0}
    )
    chemistry = ChemistryData(
        source=Path("hydrogen-ionization.yaml"),
        species=(
            SpeciesData("e", "gas", -1, 0.000548579909, {}),
            SpeciesData("H", "gas", 0, 1.00782503223, {"H": 1.0}),
            SpeciesData("H_plus", "gas", 1, 1.007276466621, {"H": 1.0}),
        ),
        gas_reactions=(
            ReactionData(
                "ionize_h",
                {"e": 1.0, "H": 1.0},
                {"H_plus": 1.0, "e": 2.0},
                rate_model.id,
                None,
            ),
        ),
        boundary_reactions=(),
        surface_reactions=(),
        rate_models={rate_model.id: rate_model},
        cross_sections={},
    )

    compiled = compile_chemistry(chemistry)

    assert compiled.reaction_ids == ("ionize_h",)


def test_mass_balance_accepts_charge_corrected_boundary_mass() -> None:
    chemistry = ChemistryData(
        source=Path("hydrogen-neutralization.yaml"),
        species=(
            SpeciesData("e", "gas", -1, 0.000548579909, {}),
            SpeciesData("H", "gas", 0, 1.00782503223, {"H": 1.0}),
            SpeciesData("H_plus", "gas", 1, 1.007276466621, {"H": 1.0}),
        ),
        gas_reactions=(),
        boundary_reactions=(
            ReactionData(
                "neutralize_h",
                {"H_plus": 1.0},
                {"H": 1.0},
                None,
                None,
            ),
        ),
        surface_reactions=(),
        rate_models={},
        cross_sections={},
    )

    compiled = compile_chemistry(chemistry)

    assert compiled.boundary_reactions[0].id == "neutralize_h"


@pytest.mark.parametrize("reaction_count", [128, 256])
def test_jacobian_pattern_does_not_overflow_with_many_shared_dependencies(
    reaction_count: int,
) -> None:
    rate_model = RateModelData("shared", "constant", {"value": 1.0})
    chemistry = _generic_rate_dimension_chemistry(
        tuple(
            ReactionData(
                f"convert_{index}",
                {"A": 1.0},
                {"B": 1.0},
                rate_model.id,
                None,
            )
            for index in range(reaction_count)
        ),
        rate_model,
    )

    pattern = compile_chemistry(chemistry).jacobian_species_pattern

    np.testing.assert_array_equal(
        pattern,
        [
            [True, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
    )


def test_rejects_rate_model_shared_across_incompatible_orders() -> None:
    rate_model = RateModelData("shared", "constant", {"value": 1.0})
    chemistry = _generic_rate_dimension_chemistry(
        (
            ReactionData("unimolecular", {"A": 1.0}, {"B": 1.0}, "shared", None),
            ReactionData(
                "bimolecular",
                {"A": 1.0, "B": 1.0},
                {"AB": 1.0},
                "shared",
                None,
            ),
        ),
        rate_model,
    )

    with pytest.raises(
        ChemistryError,
        match=r"incompatible overall orders.*s\^-1.*m3/s",
    ):
        compile_chemistry(chemistry)


def test_rejects_declared_rate_order_that_disagrees_with_reaction() -> None:
    rate_model = RateModelData(
        "declared", "constant", {"value": 1.0, "overall_order": 2.0}
    )
    chemistry = _generic_rate_dimension_chemistry(
        (ReactionData("convert", {"A": 1.0}, {"B": 1.0}, "declared", None),),
        rate_model,
    )

    with pytest.raises(
        ChemistryError,
        match=r"declares overall_order 2.*reaction 'convert'.*order 1",
    ):
        compile_chemistry(chemistry)


def test_loader_accepts_matching_explicit_rate_order_in_canonical_si(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "gas.csv").write_text(
        "id,equation,rate_model\nneutral,Ar -> Ar,decay\n",
        encoding="utf-8",
    )
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n"
        "  decay:\n"
        "    kind: constant\n"
        "    value: 1.0\n"
        "    overall_order: 1\n",
        encoding="utf-8",
    )

    chemistry = compile_chemistry(load_chemistry(manifest))

    assert chemistry.rate_evaluators[0](SimpleNamespace()) == 1.0


def test_product_electron_can_receive_explicit_chemi_ionization_energy() -> None:
    rate_model = RateModelData("decay", "first_order", {"rate_s_inv": 1.0})
    chemistry = _generic_rate_dimension_chemistry(
        (
            ReactionData(
                "chemi_ionize",
                {"A": 1.0},
                {"A_plus": 1.0, "e": 1.0},
                "decay",
                electron_energy_transfer_eV=2.0,
            ),
        ),
        rate_model,
    )

    compiled = compile_chemistry(chemistry)

    np.testing.assert_array_equal(compiled.electron_energy_transfer_eV, [2.0])


def test_chemistry_compile_rejects_unsupported_surface_flux_shape() -> None:
    rate_model = RateModelData("stick", "sticking", {"value": 0.2})
    chemistry = ChemistryData(
        source=Path("surface-shape.yaml"),
        species=(
            SpeciesData("e", "gas", -1, 0.00054858, {}),
            SpeciesData("A", "gas", 0, 40.0, {"X": 1.0}),
            SpeciesData(
                "site",
                "surface",
                0,
                0.0,
                {"site": 1.0},
                state_tags=frozenset({"site"}),
            ),
            SpeciesData("Ads", "surface", 0, 40.0, {"X": 1.0, "site": 1.0}),
        ),
        gas_reactions=(),
        boundary_reactions=(),
        surface_reactions=(
            ReactionData(
                "stick_twice",
                {"A": 2.0, "site": 2.0},
                {"Ads": 2.0},
                rate_model.id,
                None,
            ),
        ),
        rate_models={rate_model.id: rate_model},
        cross_sections={},
    )

    with pytest.raises(
        ChemistryError,
        match="sticking surface reaction stick_twice must have exactly one unit",
    ):
        compile_chemistry(chemistry)


def test_rejects_nonmonotone_cross_section_without_repair(tmp_path: Path) -> None:
    with pytest.raises(ChemistryError, match="strictly increasing"):
        load_chemistry(_mechanism(tmp_path, bad_curve=True))


def test_rejects_nonzero_cross_section_below_declared_threshold(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "xs.csv").write_text(
        "energy_eV,sigma_m2\n1,1e-20\n5,0\n10,2e-20\n",
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="nonzero below threshold"):
        load_chemistry(manifest)


def test_rejects_gap_between_threshold_and_first_cross_section_node(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "xs.csv").write_text(
        "energy_eV,sigma_m2\n1,0\n4,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="explicit node at threshold"):
        load_chemistry(manifest)


def test_threshold_contract_is_ulp_aware_and_not_applied_to_momentum(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    near_threshold = float(np.nextafter(5.0, np.inf))
    (tmp_path / "xs.csv").write_text(
        f"energy_eV,sigma_m2\n1,0\n{near_threshold!r},0\n10,2e-20\n",
        encoding="utf-8",
    )
    compile_chemistry(load_chemistry(manifest))

    (tmp_path / "cross_sections.yaml").write_text(
        "cross_sections:\n"
        "  - id: ionization\n"
        "    kind: momentum_transfer\n"
        "    target: Ar\n"
        "    threshold_eV: 5\n"
        "    electron_energy_transfer_eV: 0.0\n"
        "    file: xs.csv\n",
        encoding="utf-8",
    )
    (tmp_path / "xs.csv").write_text(
        "energy_eV,sigma_m2\n0,1e-20\n1,1e-20\n10,2e-20\n",
        encoding="utf-8",
    )

    assert load_chemistry(manifest).cross_sections["ionization"].kind == (
        "momentum_transfer"
    )


def test_rejects_duplicate_keys_in_canonical_chemistry_yaml(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n  duplicate:\n    kind: constant\n    value: 1.0\n"
        "rate_models: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="duplicate key"):
        load_chemistry(manifest)


def test_rejects_duplicate_columns_in_canonical_chemistry_csv(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "species.csv").write_text(
        "id,phase,charge,charge,mass_amu,elements,state_tags\n"
        "e,gas,-1,-1,0.00054858,,electron\n",
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="duplicate CSV columns"):
        load_chemistry(manifest)


def test_shared_tabulated_rate_model_is_compiled_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _mechanism(tmp_path)
    (tmp_path / "gas.csv").write_text(
        "id,equation,rate_model\n"
        "ionize_a,e + Ar -> Ar_plus + e + e,shared\n"
        "ionize_b,e + Ar -> Ar_plus + e + e,shared\n",
        encoding="utf-8",
    )
    (tmp_path / "rate.csv").write_text("x,value\n1,1e-15\n10,2e-15\n", encoding="utf-8")
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n"
        "  shared:\n"
        "    kind: tabulated_1d\n"
        "    axis: mean_energy_eV\n"
        "    file: rate.csv\n"
        "    bounds: error\n",
        encoding="utf-8",
    )
    calls = 0
    original = compile_module.load_rate_table

    def counted(path: Path, *, sources=None) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return original(path, sources=sources)

    monkeypatch.setattr(compile_module, "load_rate_table", counted)
    sources = InputSourceStore()
    chemistry = compile_chemistry(
        load_chemistry(manifest, sources=sources), sources=sources
    )
    assert (tmp_path / "rate.csv").resolve() in sources.contents()

    assert calls == 1
    assert chemistry.rate_evaluators[0] is chemistry.rate_evaluators[1]


def test_compile_preserves_matrix_and_metadata_order(tmp_path: Path) -> None:
    electron = SpeciesData("e", "gas", -1, 0.00054858, {})
    oxygen = SpeciesData("O", "gas", 0, 16.0, {"O": 1.0})
    oxygen_ion = SpeciesData("O_plus", "gas", 1, 16.0, {"O": 1.0})
    oxygen_molecule = SpeciesData("O2", "gas", 0, 32.0, {"O": 2.0})
    oxygen_anion = SpeciesData("O_minus", "gas", -1, 16.0, {"O": 1.0})
    site = SpeciesData("site", "surface", 0, 0.0, {"site": 1.0})
    adsorbed_oxygen = SpeciesData(
        "O_site",
        "surface",
        0,
        16.0,
        {"O": 1.0, "site": 1.0},
    )
    shared_rate = RateModelData("shared", "constant", {"value": 2.0e-15})
    bimolecular_rate = RateModelData("bimolecular", "constant", {"value": 2.0e-15})
    surface_rate = RateModelData("stick", "sticking", {"value": 1.0})
    gas_reactions = (
        ReactionData(
            "dissociate",
            {"O2": 1.0},
            {"O": 2.0},
            "shared",
            0.0,
            gas_heating_eV=0.25,
            zones=("bulk",),
        ),
        ReactionData(
            "attach",
            {"e": 1.0, "O": 1.0},
            {"O_minus": 1.0},
            "bimolecular",
            None,
        ),
        ReactionData(
            "detach",
            {"O_minus": 1.0},
            {"e": 1.0, "O": 1.0},
            "shared",
            -(0.5),
        ),
    )
    boundary_reactions = (
        ReactionData(
            "neutralize",
            {"O_plus": 1.0},
            {"O": 1.0},
            None,
            None,
            zones=("plasma",),
            surfaces=("wall",),
        ),
    )
    surface_reactions = (
        ReactionData(
            "adsorb",
            {"O": 1.0, "site": 1.0},
            {"O_site": 1.0},
            "stick",
            None,
            surfaces=("wall",),
        ),
    )
    data = ChemistryData(
        source=tmp_path / "chemistry.yaml",
        species=(
            electron,
            oxygen,
            oxygen_ion,
            oxygen_molecule,
            oxygen_anion,
            site,
            adsorbed_oxygen,
        ),
        gas_reactions=gas_reactions,
        boundary_reactions=boundary_reactions,
        surface_reactions=surface_reactions,
        rate_models=MappingProxyType(
            {
                shared_rate.id: shared_rate,
                bimolecular_rate.id: bimolecular_rate,
                surface_rate.id: surface_rate,
            }
        ),
        cross_sections=MappingProxyType({}),
        provenance=MappingProxyType({"source": "contract-test"}),
    )

    chemistry = compile_chemistry(data)

    assert chemistry.species_ids == ("O", "O_plus", "O2", "O_minus")
    assert chemistry.reaction_ids == ("dissociate", "attach", "detach")
    assert chemistry.reaction_zones == (("bulk",), (), ())
    np.testing.assert_array_equal(chemistry.charges, [0.0, 1.0, 0.0, -1.0])
    np.testing.assert_array_equal(
        chemistry.masses_kg,
        [
            oxygen.mass_kg,
            oxygen_ion.mass_kg,
            oxygen_molecule.mass_kg,
            oxygen_anion.mass_kg,
        ],
    )
    np.testing.assert_array_equal(
        chemistry.stoichiometry,
        [[2.0, 0.0, -1.0, 0.0], [-1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]],
    )
    np.testing.assert_array_equal(
        chemistry.reactant_orders,
        [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    )
    np.testing.assert_array_equal(chemistry.electron_orders, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(
        chemistry.electron_energy_transfer_eV, [0.0, 0.0, -0.5]
    )
    np.testing.assert_array_equal(chemistry.gas_heating_eV, [0.25, 0.0, 0.0])
    assert chemistry.element_names == ("O",)
    np.testing.assert_array_equal(chemistry.element_matrix, [[1.0, 1.0, 2.0, 1.0]])
    np.testing.assert_array_equal(
        chemistry.jacobian_species_pattern,
        [
            [True, True, True, True],
            [False, False, False, False],
            [False, False, True, False],
            [True, True, False, True],
        ],
    )
    assert chemistry.boundary_reactions[0].incident_species == "O_plus"
    assert chemistry.boundary_reactions[0].products == {"O": 1.0}
    assert chemistry.boundary_reactions[0].zones == ("plasma",)
    assert chemistry.boundary_reactions[0].surfaces == ("wall",)
    assert chemistry.boundary_reactions[0].probability == 1.0
    assert chemistry.boundary_reactions[0].wall_charge_per_event == 1.0
    assert chemistry.surface_reactions == surface_reactions
    assert chemistry.provenance == {"source": "contract-test"}
    for values in (
        chemistry.charges,
        chemistry.masses_kg,
        chemistry.stoichiometry,
        chemistry.reactant_orders,
        chemistry.electron_orders,
        chemistry.electron_energy_transfer_eV,
        chemistry.gas_heating_eV,
        chemistry.element_matrix,
        chemistry.jacobian_species_pattern,
    ):
        if values is chemistry.jacobian_species_pattern:
            assert values.dtype == bool
        else:
            assert values.dtype == float
        assert not values.flags.writeable


@pytest.mark.parametrize("source_kind", ["yaml", "csv", "python"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("elements", {"Ar": -1.0}), ("mass_amu", 0.0), ("cv_over_kb", 0.0)],
)
def test_species_physical_contract_is_shared_by_all_inputs(
    tmp_path, source_kind, field, value
):
    record = {
        "id": "Ar",
        "phase": "gas",
        "charge": 0,
        "mass_amu": 39.948,
        "elements": {"Ar": 1.0},
    }
    record[field] = value
    with pytest.raises(ChemistryError, match=field):
        _load_species_input(tmp_path, source_kind, record)


def _load_species_input(
    tmp_path: Path, source_kind: str, record: dict[str, Any]
) -> None:
    if source_kind == "python":
        SpeciesData(**record)
    elif source_kind == "yaml":
        load_definition({"species": [record]}, tmp_path / "inline.yaml", tmp_path)
    else:
        path = tmp_path / "species.csv"
        elements = ";".join(
            f"{name}:{count}" for name, count in record["elements"].items()
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("id", "phase", "charge", "mass_amu", "elements", "cv_over_kb")
            )
            writer.writerow(
                (
                    record["id"],
                    "gas",
                    0,
                    record["mass_amu"],
                    elements,
                    record.get("cv_over_kb", ""),
                )
            )
        load_species(path)


def test_chemistry_sources_capture_bytes_used_by_nested_readers(tmp_path):
    manifest = _mechanism(tmp_path)
    store = InputSourceStore()
    data = load_chemistry(manifest, sources=store)
    captured = dict(store.contents())
    assert set(captured) == {path.resolve() for path in data.source_files}
    (tmp_path / "xs.csv").write_text("changed after parsing", encoding="utf-8")
    compiled = compile_chemistry(data, sources=store)
    assert compiled.cross_sections["ionization"].sigma_m2[-1] == 2e-20
    assert dict(store.contents()) == captured


def test_rate_contract_is_also_checked_for_python_construction(tmp_path):
    data = load_chemistry(_mechanism(tmp_path))
    invalid = RateModelData(
        "electron_impact",
        "electron_impact",
        {"cross_section": "ionization", "branching_yield": -1.0},
    )
    with pytest.raises(ChemistryError, match="branching_yield"):
        compile_chemistry(replace(data, rate_models={invalid.id: invalid}))


@pytest.mark.parametrize("field", ["id", "target"])
def test_cross_section_identifiers_are_rejected_at_construction(field: str) -> None:
    with pytest.raises(ChemistryError, match="identifier"):
        _programmatic_cross_section(**{field: ""})


@pytest.mark.parametrize("source_kind", ["yaml", "csv", "python"])
@pytest.mark.parametrize("bad_id", ["Ar,part", " Ar", "Ar ", "Ar[part]", "Ar\npart"])
def test_species_series_components_share_the_same_identifier_boundary(
    tmp_path: Path,
    source_kind: str,
    bad_id: str,
) -> None:
    record = {
        "id": bad_id,
        "phase": "gas",
        "charge": 0,
        "mass_amu": 39.948,
        "elements": {"Ar": 1.0},
    }
    with pytest.raises(ChemistryError, match="identifier"):
        _load_species_input(tmp_path, source_kind, record)


@pytest.mark.parametrize(
    "field", ["id", "rate_model", "zones", "surfaces", "reactants", "products"]
)
def test_direct_reaction_identifiers_and_selectors_reject_series_delimiters(
    field: str,
) -> None:
    record = {
        "id": "convert",
        "reactants": {"A": 1.0},
        "products": {"B": 1.0},
        "rate_model": "conversion",
        "electron_energy_transfer_eV": None,
    }
    if field in {"zones", "surfaces"}:
        record[field] = ("z,part",)
    elif field in {"reactants", "products"}:
        record[field] = {"z,part": 1.0}
    else:
        record[field] = "z,part"
    with pytest.raises(ChemistryError, match="identifier"):
        ReactionData(**record)


@pytest.mark.parametrize(
    "parameters",
    [
        {"cross_section": "bad,curve"},
        {"rate_id": " rate"},
        {"coverage": {"kind": "site_blocking", "site_species": "site,bad"}},
        {"coverage": {"kind": "species_power", "species": "species[bad]"}},
    ],
)
def test_direct_rate_identifier_references_use_the_same_contract(parameters) -> None:
    with pytest.raises(ChemistryError, match="identifier"):
        RateModelData("valid", "constant", parameters)


@pytest.mark.parametrize("field", ["id", "target"])
def test_cross_section_components_reject_reserved_series_delimiters(field: str) -> None:
    with pytest.raises(ChemistryError, match="identifier"):
        _programmatic_cross_section(**{field: "z,part"})


def test_identifier_validation_preserves_scientific_punctuation_and_annotations() -> (
    None
):
    species = SpeciesData(
        "Ar+:励起*",
        "surface",
        1,
        39.948,
        {"element,key[]": 1.0},
        surfaces=("wafer:*",),
        metadata={"notes": " ,[]\n "},
    )
    reaction = ReactionData(
        "reaction:励起*",
        {species.id: 1.0},
        {species.id: 1.0},
        "rate:1*",
        None,
        zones=("領域",),
        surfaces=("wafer:*",),
    )
    rate = RateModelData("rate:1*", "electron_table", {"rate_id": "Ar+:励起*"})
    assert reaction.reactants[species.id] == 1.0
    assert species.elements["element,key[]"] == 1.0
    assert species.metadata["notes"] == " ,[]\n "
    assert rate.parameters["rate_id"] == species.id


@pytest.mark.parametrize("bad_id", ["Ar,part", " Ar", "Ar "])
def test_public_case_rejects_ambiguous_chemistry_series_components(
    tmp_path: Path, bad_id: str
) -> None:
    from plasma_global import case_from_mapping, compile_case
    from tools.generate_schema_docs import minimal_case_document

    document = minimal_case_document()
    document["chemistry"]["species"][1]["id"] = bad_id
    case = case_from_mapping(document, base_directory=tmp_path)
    with pytest.raises(ChemistryError, match="identifier"):
        compile_case(case)


def test_public_case_preserves_chemical_punctuation_in_saved_species_series(
    tmp_path: Path,
) -> None:
    from plasma_global import case_from_mapping, simulate
    from tools.generate_schema_docs import minimal_case_document

    document = minimal_case_document()
    species = "Ar+:meta*"
    document["chemistry"]["species"][2]["id"] = species
    document["reactor"]["zones"]["plasma"]["initial"]["seed_densities_m3"] = {
        species: 1e12
    }
    result = simulate(case_from_mapping(document, base_directory=tmp_path))
    np.testing.assert_array_equal(result.series(f"n[plasma,{species}]"), 1e12)


@pytest.mark.parametrize("used", [False, True])
def test_declared_rate_table_is_captured_and_copied_even_when_unused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    used: bool,
) -> None:
    import yaml

    import plasma_global.examples as examples
    from plasma_global import compile_case, load_case
    from tools.generate_schema_docs import minimal_case_document

    source = tmp_path / "package"
    (source / "cases").mkdir(parents=True)
    (source / "data").mkdir()
    table = source / "data" / "shared.csv"
    table.write_text("x,value\n200,1\n1000,2\n", encoding="utf-8")
    document = minimal_case_document()
    document["chemistry"]["rate_models"] = {
        "shared": {
            "kind": "tabulated_1d",
            "axis": "gas_temperature_K",
            "file": "../data/shared.csv",
            "bounds": "error",
            "overall_order": 1,
        }
    }
    document["chemistry"]["provenance"] = {"file": "notes/source.csv"}
    if used:
        document["chemistry"]["gas_reactions"] = [
            {
                "id": "identity",
                "equation": "Ar -> Ar",
                "rate": "shared",
            }
        ]
    source_case = source / "cases" / "shared.yaml"
    source_case.write_text(yaml.safe_dump(document), encoding="utf-8")
    original = compile_case(load_case(source_case))
    record = original.metadata["provenance"]["input_files"]["files"][str(table)]
    assert "declared_rate_table" in record["roles"]
    assert ("chemistry.rate_table" in record["roles"]) is used
    assert original.input_sources[table] == table.read_bytes()

    monkeypatch.setattr(examples, "files", lambda package: source)
    copied_path = examples.initialize_example("shared", tmp_path / "copied")
    copied = load_case(copied_path)
    prepared = compile_case(copied)
    assert (
        copied_path.parent / "data" / "shared.csv"
    ).read_bytes() == table.read_bytes()
    assert copied.chemistry["provenance"]["file"] == "notes/source.csv"
    assert not any(path.name == "source.csv" for path in prepared.input_sources)


@pytest.mark.parametrize("source_kind", ["yaml", "csv", "python"])
@pytest.mark.parametrize("selector", ["zones", "surfaces"])
def test_reaction_selectors_are_validated_without_trimming(
    tmp_path: Path,
    source_kind: str,
    selector: str,
) -> None:
    with pytest.raises(ChemistryError, match="identifier"):
        _load_reaction_selector(tmp_path, source_kind, selector)


def _load_reaction_selector(tmp_path: Path, source_kind: str, selector: str) -> None:
    if source_kind == "python":
        ReactionData(
            "identity", {"Ar": 1.0}, {"Ar": 1.0}, "rate", None, **{selector: (" edge",)}
        )
    elif source_kind == "yaml":
        load_definition(
            {
                "gas_reactions": [
                    {
                        "id": "identity",
                        "equation": "Ar -> Ar",
                        "rate": "rate",
                        selector: [" edge"],
                    }
                ]
            },
            tmp_path / "inline.yaml",
            tmp_path,
        )
    else:
        path = tmp_path / "reaction.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("id", "equation", "rate_model", selector))
            writer.writerow(("identity", "Ar -> Ar", "rate", " edge"))
        load_reactions(path)


def test_direct_rate_model_id_and_chemistry_mapping_keys_are_components(
    tmp_path: Path,
) -> None:
    with pytest.raises(ChemistryError, match="identifier"):
        RateModelData("rate,part", "constant", {"value": 1.0})
    data = load_chemistry(_mechanism(tmp_path))
    with pytest.raises(ChemistryError, match="identifier"):
        replace(data, rate_models={"rate,part": next(iter(data.rate_models.values()))})
