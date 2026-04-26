from __future__ import annotations

import math

import pytest

from scripts.compare_argon_with_pygmol import _equivalent_cylinder, _pygmol_argon_chemistry
from tools.external_benchmarks.pygmol_argon import (
    DEFAULT_CASE,
    build_local_rate_fit_model,
    build_local_rate_table_model,
    fit_pygmol_arrhenius,
    load_benchmark_model,
)
from tools.external_benchmarks.pygmol_sensitivity import apply_variant_to_model, variant_specs
from tools.external_benchmarks.pygmol_same_footing import stage_specs
from plasma_global.workflows.context import load_case_from_yaml


def test_equivalent_cylinder_preserves_volume_and_area() -> None:
    volume = 0.04
    area = 0.6614
    radius, length = _equivalent_cylinder(volume, area, preferred_radius_m=0.10)

    assert math.pi * radius * radius * length == pytest.approx(volume, rel=1.0e-10, abs=1.0e-12)
    assert 2.0 * math.pi * radius * (radius + length) == pytest.approx(area, rel=1.0e-10, abs=1.0e-12)


def test_pygmol_argon_chemistry_keeps_wall_neutralization_explicit() -> None:
    chemistry = _pygmol_argon_chemistry()

    assert chemistry['species_ids'] == ['Ar', 'Ar+']
    assert chemistry['species_surface_sticking_coefficients'] == [0.0, 1.0]
    assert chemistry['species_surface_return_matrix'][0][1] == 1.0
    assert 'Ar_ionization' in chemistry['reactions_ids']
    assert len(chemistry['reactions_ids']) == 3


def test_pygmol_benchmark_model_declares_rate_isolation() -> None:
    model = load_benchmark_model()

    assert model['model']['scope'] == 'external_benchmark_only'
    assert model['model']['rate_alignment_with_local_case'] == 'intentionally_not_aligned'
    assert 'Do not import this file from plasma_global core modules.' in model['model']['guardrails']


def test_pygmol_sensitivity_variants_stay_external_only() -> None:
    model = load_benchmark_model()
    specs = variant_specs()
    ionization_half = next(spec for spec in specs if spec['id'] == 'ionization_rate_x0p5')

    modified = apply_variant_to_model(model, ionization_half)

    assert modified['model']['scope'] == 'external_benchmark_only'
    assert modified['model']['variant_id'] == 'ionization_rate_x0p5'
    assert modified['reactions'][0]['arrhenius']['a'] == pytest.approx(model['reactions'][0]['arrhenius']['a'] * 0.5)
    assert model['reactions'][0]['arrhenius']['a'] != modified['reactions'][0]['arrhenius']['a']


def test_pygmol_arrhenius_fit_recovers_synthetic_rate() -> None:
    import numpy as np

    te = np.geomspace(0.2, 6.0, 40)
    truth = {'a': 2.5e-14, 'b': 0.41, 'c_eV': 12.3}
    rates = truth['a'] * te ** truth['b'] * np.exp(-truth['c_eV'] / te)

    fitted, metrics = fit_pygmol_arrhenius(te, rates)

    assert fitted['a'] == pytest.approx(truth['a'], rel=1.0e-10)
    assert fitted['b'] == pytest.approx(truth['b'], rel=1.0e-10)
    assert fitted['c_eV'] == pytest.approx(truth['c_eV'], rel=1.0e-10)
    assert metrics['max_abs_log10_error'] < 1.0e-10


def test_pygmol_local_rate_fit_model_keeps_generated_rates_external() -> None:
    loaded = load_case_from_yaml(DEFAULT_CASE)

    model = build_local_rate_fit_model(loaded)

    assert model['model']['scope'] == 'external_benchmark_only'
    assert model['model']['rate_alignment_with_local_case'] == 'local_lxcat_arrhenius_fit'
    assert len(model['reactions']) > 3
    assert any(item['local_channel_kind'] == 'ionization' for item in model['reactions'])
    assert any(item['local_channel_kind'] == 'excitation' for item in model['reactions'])
    assert any(item['local_channel_kind'] == 'elastic_momentum' for item in model['reactions'])
    assert max(item['fit_metrics']['rms_log10_error'] for item in model['reactions']) < 2.0


def test_pygmol_local_rate_table_model_carries_runtime_tables() -> None:
    loaded = load_case_from_yaml(DEFAULT_CASE)

    model = build_local_rate_table_model(loaded)
    rate_table = model['model']['rate_table']

    assert model['model']['scope'] == 'external_benchmark_only'
    assert model['model']['rate_alignment_with_local_case'] == 'local_lxcat_rate_table'
    assert len(rate_table['te_eV']) >= 10
    assert set(rate_table['rate_by_reaction_id']) == {item['id'] for item in model['reactions']}
    assert all(len(values) == len(rate_table['te_eV']) for values in rate_table['rate_by_reaction_id'].values())


def test_pygmol_same_footing_stages_are_cumulative() -> None:
    specs = stage_specs()

    assert [item['stage_id'] for item in specs] == [
        'same_rate_recipe_power_pygmol_wall',
        'same_rate_local_power_pygmol_wall',
        'same_rate_local_power_local_wall',
    ]
    assert specs[0]['power_mode'] == 'recipe'
    assert specs[1]['power_mode'] == 'local-absorbed'
    assert specs[2]['wall_loss_mode'] == 'local-coefficient'
