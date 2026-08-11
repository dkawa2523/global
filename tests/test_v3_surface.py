from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from plasma_global.chemistry.data import RateModelData, load_chemistry
from plasma_global.errors import CaseValidationError
from plasma_global.models.surface import CompiledSurfaceModel, SurfaceGeometry


def _chemistry(tmp_path: Path) -> Path:
    (tmp_path / "species.csv").write_text(
        "id,phase,charge,mass_amu,elements,state_tags,surfaces\n"
        "e,gas,-1,0.00054858,,electron,\n"
        "A,gas,0,40,A:1,,\n"
        "wall:*,surface,0,0,site:1,site,wall\n"
        "wall:A*,surface,0,40,A:1;site:1,adsorbate,wall\n",
        encoding="utf-8",
    )
    (tmp_path / "gas.csv").write_text("id,equation,rate_model\n", encoding="utf-8")
    (tmp_path / "surface.csv").write_text(
        "id,equation,rate_model,zones,surfaces\n"
        "stick,A + wall:* -> wall:A*,stick,plasma,wall\n",
        encoding="utf-8",
    )
    (tmp_path / "rates.yaml").write_text(
        "rate_models:\n  stick:\n    kind: sticking\n    value: 0.1\n    coverage:\n      kind: site_blocking\n      site_species: wall:*\n      exponent: 1.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "chemistry.yaml"
    manifest.write_text(
        "schema_version: 3\n"
        "species: species.csv\n"
        "gas_reactions: gas.csv\n"
        "surface_reactions: surface.csv\n"
        "rate_models: rates.yaml\n",
        encoding="utf-8",
    )
    return manifest


def test_free_site_is_algebraic_and_adsorption_conserves_element(
    tmp_path: Path,
) -> None:
    chemistry = load_chemistry(_chemistry(tmp_path))
    surface = SurfaceGeometry("wall", "plasma", 2.0, 1.0e19, 300.0, {"wall:A*": 0.0})
    model = CompiledSurfaceModel(
        chemistry,
        gas_species_ids=("A",),
        gas_masses_kg=np.array([40.0 * 1.66053906660e-27]),
        zone_ids=("plasma",),
        zone_volumes_m3=np.array([1.0]),
        surfaces=(surface,),
    )
    state = model.initial_state()
    assert model.coverage(state, "wall", "wall:*") == 1.0

    state[0] = np.nextafter(0.37, 1.0)
    free_site = model.coverage(state, "wall", "wall:*")
    assert abs((free_site + state[0]) - 1.0) <= np.finfo(float).eps

    evaluated = model.evaluate(state, np.array([[1.0e20]]), np.array([300.0]))
    gas_particles_per_s = evaluated.gas_derivative_m3_s[0, 0]
    surface_particles_per_s = (
        evaluated.coverage_derivative_s_inv[0]
        * surface.site_density_m2
        * surface.area_m2
    )
    assert gas_particles_per_s * 1.0 + surface_particles_per_s == pytest.approx(0.0)


def test_free_site_cannot_be_an_independent_initial_condition(tmp_path: Path) -> None:
    chemistry = load_chemistry(_chemistry(tmp_path))
    with pytest.raises(CaseValidationError, match="algebraic free site"):
        CompiledSurfaceModel(
            chemistry,
            gas_species_ids=("A",),
            gas_masses_kg=np.array([40.0 * 1.66053906660e-27]),
            zone_ids=("plasma",),
            zone_volumes_m3=np.array([1.0]),
            surfaces=(
                SurfaceGeometry("wall", "plasma", 1.0, 1.0e19, 300.0, {"wall:*": 1.0}),
            ),
        )


def test_surface_rate_configuration_is_compiled_and_rates_are_optional(
    tmp_path: Path,
) -> None:
    chemistry = load_chemistry(_chemistry(tmp_path))
    parameters: dict[str, object] = {
        "value": 0.1,
        "coverage": {
            "kind": "site_blocking",
            "site_species": "wall:*",
            "exponent": 1.0,
        },
    }
    chemistry = replace(
        chemistry,
        rate_models={
            "stick": RateModelData("stick", "sticking", parameters),
        },
    )
    model = CompiledSurfaceModel(
        chemistry,
        gas_species_ids=("A",),
        gas_masses_kg=np.array([40.0 * 1.66053906660e-27]),
        zone_ids=("plasma",),
        zone_volumes_m3=np.array([1.0]),
        surfaces=(
            SurfaceGeometry("wall", "plasma", 2.0, 1.0e19, 300.0, {"wall:A*": 0.2}),
        ),
    )
    state = model.initial_state()
    gas = np.array([[1.0e20]])
    temperature = np.array([300.0])
    expected = model.evaluate(state, gas, temperature)

    # The compiled numeric kernel must be independent of its source mapping.
    parameters.clear()
    parameters["coverage"] = {"kind": "unsupported"}
    without_diagnostics = model.evaluate(state, gas, temperature, collect_rates=False)

    assert without_diagnostics.rates_m2_s == {}
    assert without_diagnostics.gas_derivative_m3_s == pytest.approx(
        expected.gas_derivative_m3_s
    )
    assert without_diagnostics.coverage_derivative_s_inv == pytest.approx(
        expected.coverage_derivative_s_inv
    )


def test_compiled_surface_model_and_numeric_arrays_are_immutable(
    tmp_path: Path,
) -> None:
    chemistry = load_chemistry(_chemistry(tmp_path))
    model = CompiledSurfaceModel(
        chemistry,
        gas_species_ids=("A",),
        gas_masses_kg=np.array([40.0 * 1.66053906660e-27]),
        zone_ids=("plasma",),
        zone_volumes_m3=np.array([1.0]),
        surfaces=(
            SurfaceGeometry("wall", "plasma", 2.0, 1.0e19, 300.0, {"wall:A*": 0.0}),
        ),
    )

    with pytest.raises(AttributeError, match="immutable"):
        model.domain_atol = 1.0
    with pytest.raises(ValueError, match="read-only"):
        model.gas_masses_kg[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        model.zone_volumes_m3[0] = 2.0
