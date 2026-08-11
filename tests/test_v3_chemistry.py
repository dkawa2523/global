from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import plasma_global.chemistry.compile as compile_module
from plasma_global.chemistry.compile import compile_chemistry
from plasma_global.chemistry.data import CrossSectionData, load_chemistry
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
        "rate_models:\n  electron_impact:\n    kind: electron_impact\n    cross_section: ionization\n    branching_yield: 1.0\n",
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
    with pytest.raises(ValueError):
        chemistry.stoichiometry[0, 0] = 0.0


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


def test_rejects_unbalanced_reaction(tmp_path: Path) -> None:
    manifest = _mechanism(tmp_path, equation="e + Ar -> Ar_plus + Ar_plus + e")
    with pytest.raises(ChemistryError, match="conserve"):
        compile_chemistry(load_chemistry(manifest))


def test_rejects_nonmonotone_cross_section_without_repair(tmp_path: Path) -> None:
    with pytest.raises(ChemistryError, match="strictly increasing"):
        load_chemistry(_mechanism(tmp_path, bad_curve=True))


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
