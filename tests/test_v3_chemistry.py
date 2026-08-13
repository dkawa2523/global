from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import plasma_global.chemistry.compile as compile_module
from plasma_global.chemistry._contracts import normalized_cross_section_curve
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
        "    energy_loss_eV: 5\n"
        "    file: xs.csv\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "chemistry.yaml"
    manifest.write_text(
        "schema_version: 3\n"
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
    assert chemistry.energy_loss_eV.tolist() == [5.0]
    assert chemistry.boundary_reactions[0].products == {"Ar": 1.0}
    assert chemistry.boundary_reactions[0].wall_charge_per_event == 1.0
    assert chemistry.jacobian_species_pattern.tolist() == [
        [True, True],
        [True, True],
    ]
    with pytest.raises(ValueError, match="read-only"):
        chemistry.stoichiometry[0, 0] = 0.0


def test_signed_electron_energy_transfer_preserves_legacy_loss_view(
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
    np.testing.assert_array_equal(chemistry.energy_loss_eV, [5.0])


def test_rejects_ambiguous_electron_energy_fields(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path)
    path = tmp_path / "cross_sections.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    energy_loss_eV: 5\n",
            "    energy_loss_eV: 5\n    electron_energy_transfer_eV: -5\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="exactly one of"):
        load_chemistry(manifest)


def test_rejects_ambiguous_electron_energy_fields_before_parsing_values(
    tmp_path: Path,
) -> None:
    manifest = _mechanism(tmp_path)
    path = tmp_path / "cross_sections.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "    energy_loss_eV: 5\n",
            "    energy_loss_eV: invalid\n"
            "    electron_energy_transfer_eV: also-invalid\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ChemistryError, match="exactly one of"):
        load_chemistry(manifest)


def _species_validation_chemistry(
    *, electron: SpeciesData, cross_section_target: str
) -> ChemistryData:
    cross_section = CrossSectionData(
        id="momentum",
        kind="momentum_transfer",
        target=cross_section_target,
        threshold_eV=0.0,
        energy_loss_eV=0.0,
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

    with pytest.raises(ModelDomainError, match="Maxwellian tail-safe range"):
        chemistry.rate_evaluators[0](context)


def test_maxwellian_constant_cross_section_matches_analytic_rate() -> None:
    sigma_m2 = 2.0e-20
    cross_section = CrossSectionData(
        id="constant",
        kind="ionization",
        target="Ar",
        threshold_eV=0.0,
        energy_loss_eV=0.0,
        energy_eV=np.linspace(0.0, 200.0, 4001),
        sigma_m2=np.full(4001, sigma_m2),
    )
    axis, rate = compile_module.maxwell_rate_table(cross_section)
    cutoff = 1.5 * cross_section.energy_eV[-1] / axis[-1]
    unresolved_tail = (1.0 + cutoff) * np.exp(-cutoff)
    assert unresolved_tail == pytest.approx(1.0e-6, rel=1.0e-12)
    mean_energy_eV = 2.0
    electron_temperature_eV = (2.0 / 3.0) * mean_energy_eV
    expected = sigma_m2 * np.sqrt(
        8.0
        * compile_module.E_CHARGE
        * electron_temperature_eV
        / (np.pi * compile_module.ELECTRON_MASS_KG)
    )

    assert np.interp(mean_energy_eV, axis, rate) == pytest.approx(expected, rel=3.0e-4)


def test_maxwellian_rejects_missing_low_energy_momentum_support() -> None:
    sigma_m2 = 1.0e-20
    cross_section = CrossSectionData(
        id="constant-truncated-at-low-energy",
        kind="momentum_transfer",
        target="Ar",
        threshold_eV=0.0,
        energy_loss_eV=0.0,
        energy_eV=np.geomspace(1.0e-3, 10.0, 4001),
        sigma_m2=np.full(4001, sigma_m2),
    )

    with pytest.raises(ChemistryError, match="must explicitly cover 0 eV"):
        compile_module.maxwell_rate_table(cross_section)


def test_maxwellian_quadrature_resolves_zero_threshold_lower_tail() -> None:
    sigma_m2 = 1.0e-20
    energy_eV = np.concatenate(([0.0], np.geomspace(1.0e-3, 10.0, 4001)))
    cross_section = CrossSectionData(
        id="constant-with-zero-support",
        kind="momentum_transfer",
        target="Ar",
        threshold_eV=0.0,
        energy_loss_eV=0.0,
        energy_eV=energy_eV,
        sigma_m2=np.full(energy_eV.size, sigma_m2),
    )

    axis, rate = compile_module.maxwell_rate_table(cross_section)
    mean_energy_eV = axis[0]
    electron_temperature_eV = (2.0 / 3.0) * mean_energy_eV
    expected = sigma_m2 * np.sqrt(
        8.0
        * compile_module.E_CHARGE
        * electron_temperature_eV
        / (np.pi * compile_module.ELECTRON_MASS_KG)
    )

    assert rate[0] == pytest.approx(expected, rel=3.0e-4)


def test_maxwellian_quadrature_keeps_positive_threshold_lower_tail_zero() -> None:
    sigma_m2 = 1.0e-20
    threshold_eV = 2.0
    cross_section = CrossSectionData(
        id="constant-above-threshold",
        kind="ionization",
        target="Ar",
        threshold_eV=threshold_eV,
        energy_loss_eV=threshold_eV,
        energy_eV=np.linspace(threshold_eV, 200.0, 4001),
        sigma_m2=np.full(4001, sigma_m2),
    )
    axis, rate = compile_module.maxwell_rate_table(cross_section)
    mean_energy_eV = 2.0
    electron_temperature_eV = (2.0 / 3.0) * mean_energy_eV
    reduced_threshold = threshold_eV / electron_temperature_eV
    full_constant_rate = sigma_m2 * np.sqrt(
        8.0
        * compile_module.E_CHARGE
        * electron_temperature_eV
        / (np.pi * compile_module.ELECTRON_MASS_KG)
    )
    expected = (
        full_constant_rate * (1.0 + reduced_threshold) * np.exp(-reduced_threshold)
    )

    assert np.interp(mean_energy_eV, axis, rate) == pytest.approx(expected, rel=3.0e-4)


def test_onset_curve_normalizer_is_readonly_and_zero_below_threshold() -> None:
    source_energy = np.array([0.0, 20.0])
    source_sigma = np.array([0.0, 1.0e-20])
    cross_section = CrossSectionData(
        id="gapped-excitation",
        kind="excitation",
        target="Ar",
        threshold_eV=10.0,
        energy_loss_eV=10.0,
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
            "id,equation,rate_model,energy_loss_eV\nsurface_loss,Ar -> Ar,unused,5\n",
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


def test_programmatic_boundary_allows_zero_legacy_energy_field(tmp_path: Path) -> None:
    loaded = load_chemistry(_mechanism(tmp_path))
    boundary = replace(loaded.boundary_reactions[0], energy_loss_eV=0.0)

    compiled = compile_chemistry(replace(loaded, boundary_reactions=(boundary,)))

    assert compiled.boundary_reactions[0].id == boundary.id


@pytest.mark.parametrize(
    "field_name",
    ["energy_loss_eV", "electron_energy_transfer_eV", "gas_heating_eV"],
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


def test_rejects_legacy_rate_model_shared_across_incompatible_orders() -> None:
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
                None,
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
        "    energy_loss_eV: 0\n"
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
    original = compile_module._table

    def counted(path: Path) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(compile_module, "_table", counted)
    chemistry = compile_chemistry(load_chemistry(manifest))

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
    surface_rate = RateModelData("stick", "surface_sticking", {})
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
            0.5,
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
    np.testing.assert_array_equal(chemistry.energy_loss_eV, [0.0, 0.0, 0.5])
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
        chemistry.energy_loss_eV,
        chemistry.gas_heating_eV,
        chemistry.element_matrix,
        chemistry.jacobian_species_pattern,
    ):
        if values is chemistry.jacobian_species_pattern:
            assert values.dtype == bool
        else:
            assert values.dtype == float
        assert not values.flags.writeable
