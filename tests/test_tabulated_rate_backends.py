from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pytest

from plasma_global.chemistry.io import load_mechanism_bundle
from plasma_global.physics.gas_rates import gas_rate_coefficient
from plasma_global.physics.surface_rates import evaluate_surface_rate
from plasma_global.physics.types import CompiledSurfaceReaction, SurfaceRateContext


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding='utf-8')
    return path


def _write_mechanism(tmp_path: Path, *, model_category: str, model_yaml: str) -> Path:
    _write(
        tmp_path / 'species.csv',
        'canonical_id,display_name,phase,charge,mass_amu,elements,aliases,state_tags,zones,surfaces\n',
    )
    _write(
        tmp_path / 'gas_reactions.csv',
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
    )
    _write(
        tmp_path / 'surface_reactions.csv',
        'reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes\n',
    )
    model_file = tmp_path / f'{model_category}_models.yaml'
    _write(model_file, model_yaml)
    return _write(
        tmp_path / 'chemistry_manifest.yaml',
        f"""
        species_file: species.csv
        gas_reactions_file: gas_reactions.csv
        surface_reactions_file: surface_reactions.csv
        model_files:
          {model_category}: {model_file.name}
        """,
    )


def _eedf(mean_energy_eV: float = 2.0, effective_field_Td: float = 50.0):
    return SimpleNamespace(
        rate_coefficients={},
        transport=SimpleNamespace(mean_energy_eV=mean_energy_eV, effective_field_Td=effective_field_Td),
    )


def test_tabulated_1d_gas_rate_can_lookup_supported_axes(tmp_path: Path) -> None:
    _write(tmp_path / 'rates' / 'temperature.csv', 'Tg_K,k_m3_s\n300,1\n500,3\n')
    _write(tmp_path / 'rates' / 'energy.csv', 'eps_eV,k_m3_s\n1,2\n3,4\n')
    _write(tmp_path / 'rates' / 'field.csv', 'Td,k_m3_s\n40,4\n60,8\n')
    _write(tmp_path / 'rates' / 'pressure.csv', 'Pa,k_m3_s\n5,10\n15,20\n')
    manifest = _write_mechanism(
        tmp_path,
        model_category='gas_rate',
        model_yaml="""
        rate_models:
          RM_T:
            backend: tabulated_1d
            file: rates/temperature.csv
            x: gas_temperature_K
            x_column: Tg_K
            value_column: k_m3_s
          RM_E:
            backend: tabulated_1d
            file: rates/energy.csv
            x: mean_energy_eV
            x_column: eps_eV
            value_column: k_m3_s
          RM_F:
            backend: tabulated_1d
            file: rates/field.csv
            x: reduced_field_Td
            x_column: Td
            value_column: k_m3_s
          RM_P:
            backend: tabulated_1d
            file: rates/pressure.csv
            x: pressure_Pa
            x_column: Pa
            value_column: k_m3_s
        """,
    )
    models = load_mechanism_bundle(manifest).rate_models

    assert gas_rate_coefficient(models['RM_T'], 400.0, _eedf(), 10.0) == pytest.approx(2.0)
    assert gas_rate_coefficient(models['RM_E'], 400.0, _eedf(), 10.0) == pytest.approx(3.0)
    assert gas_rate_coefficient(models['RM_F'], 400.0, _eedf(), 10.0) == pytest.approx(6.0)
    assert gas_rate_coefficient(models['RM_P'], 400.0, _eedf(), 10.0) == pytest.approx(15.0)


def test_tabulated_1d_bounds_error_rejects_out_of_range_lookup(tmp_path: Path) -> None:
    _write(tmp_path / 'rates.csv', 'eps_eV,k_m3_s\n0,1\n1,2\n')
    manifest = _write_mechanism(
        tmp_path,
        model_category='gas_rate',
        model_yaml="""
        rate_models:
          RM_TABLE:
            backend: tabulated_1d
            file: rates.csv
            x: mean_energy_eV
            x_column: eps_eV
            value_column: k_m3_s
            bounds_policy: error
        """,
    )
    model = load_mechanism_bundle(manifest).rate_models['RM_TABLE']

    with pytest.raises(ValueError, match='outside bounds'):
        gas_rate_coefficient(model, 300.0, _eedf(mean_energy_eV=2.0), 10.0)


@pytest.mark.parametrize(
    ('csv_text', 'model_extra', 'message'),
    [
        ('x,k\n0,1\n1,2\n', 'x_column: missing\nvalue_column: k', 'missing column'),
        ('x,k\n0,1\n0.5,-2\n', 'x_column: x\nvalue_column: k', 'negative rate'),
        ('x,k\n1,1\n0,2\n', 'x_column: x\nvalue_column: k', 'strictly increasing'),
        ('x,k\n', 'x_column: x\nvalue_column: k', 'no data rows'),
    ],
)
def test_tabulated_1d_loader_fails_fast_for_invalid_tables(tmp_path: Path, csv_text: str, model_extra: str, message: str) -> None:
    _write(tmp_path / 'rates.csv', csv_text)
    extra = model_extra.replace('\n', '\n            ')
    manifest = _write_mechanism(
        tmp_path,
        model_category='gas_rate',
        model_yaml=f"""
        rate_models:
          RM_TABLE:
            backend: tabulated_1d
            file: rates.csv
            x: mean_energy_eV
            {extra}
        """,
    )

    with pytest.raises(ValueError, match=message):
        load_mechanism_bundle(manifest)


def test_ion_yield_table_uses_ion_energy_flux_and_coverage(tmp_path: Path) -> None:
    _write(tmp_path / 'yield.csv', 'ion_energy_eV,yield\n0,0\n100,1\n')
    manifest = _write_mechanism(
        tmp_path,
        model_category='surface_rate',
        model_yaml="""
        rate_models:
          RM_YIELD:
            backend: ion_yield_table
            file: yield.csv
            energy_column: ion_energy_eV
            yield_column: yield
            coverage_factor:
              kind: species_power
              species: wafer:F*
              exponent: 1.0
        """,
    )
    model = load_mechanism_bundle(manifest).rate_models['RM_YIELD']
    system = SimpleNamespace(
        floor_density=1.0,
        gas_species=[SimpleNamespace(charge=1)],
        state_layout=SimpleNamespace(surface_index={'wafer': {'wafer:F*': 0}}),
    )
    rxn = CompiledSurfaceReaction(
        reaction_id='S_YIELD',
        zone_id='plasma',
        surface_id='wafer',
        gas_reactants=[(0, 1.0, 'Ar_plus')],
        surface_reactants=[],
        delta_gas=[],
        delta_surface=[],
        area_over_volume=1.0,
        area_m2=1.0,
        site_density_m2=1.0,
        rate_model=model,
        inventory_idx=None,
        film_factor=0.0,
    )
    context = SurfaceRateContext(
        reaction=rxn,
        gas_row=np.array([1.0e16]),
        gas_temperature_K=300.0,
        state=np.array([0.25]),
        surface_temperature_K=300.0,
        ion_energy_eV=50.0,
        positive_ion_density_m3=1.0e16,
        ion_flux_m2_s=2.0,
    )

    rate = evaluate_surface_rate(SimpleNamespace(system=system), context)

    assert rate == pytest.approx(0.25)


def test_ion_yield_table_rejects_negative_yield(tmp_path: Path) -> None:
    _write(tmp_path / 'yield.csv', 'ion_energy_eV,yield\n0,0\n100,-1\n')
    manifest = _write_mechanism(
        tmp_path,
        model_category='surface_rate',
        model_yaml="""
        rate_models:
          RM_YIELD:
            backend: ion_yield_table
            file: yield.csv
            energy_column: ion_energy_eV
            yield_column: yield
        """,
    )

    with pytest.raises(ValueError, match='negative yield'):
        load_mechanism_bundle(manifest)
