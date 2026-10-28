from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from plasma_global.chemistry._contracts import normalized_cross_section_curve
from plasma_global.chemistry._maxwell import (
    E_CHARGE,
    ELECTRON_MASS_KG,
    compile_maxwell_rate,
)
from plasma_global.chemistry.data import CrossSectionData, load_chemistry
from plasma_global.errors import ModelDomainError


@pytest.mark.parametrize("threshold", [0.0, 2.0])
@pytest.mark.parametrize("nodes", [2, 1001])
def test_constant_cross_section_matches_analytic_truncated_maxwell_average(
    threshold: float,
    nodes: int,
) -> None:
    sigma = 2.0e-20
    maximum = 200.0
    cross_section = CrossSectionData(
        "constant",
        "ionization",
        "Ar",
        threshold,
        threshold,
        np.linspace(threshold, maximum, nodes),
        np.full(nodes, sigma),
    )
    rate = compile_maxwell_rate(cross_section)
    for mean in (1.0e-12, 5.0e-4, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0):
        temperature = (2.0 / 3.0) * mean
        lower, upper = threshold / temperature, maximum / temperature
        probability = (1.0 + lower) * math.exp(-lower) - (1.0 + upper) * math.exp(
            -upper
        )
        expected = (
            sigma
            * math.sqrt(8.0 * E_CHARGE * temperature / (math.pi * ELECTRON_MASS_KG))
            * probability
        )
        assert rate(mean) == pytest.approx(expected, rel=1.0e-11, abs=0.0)
    assert rate(0.0) == 0.0
    with pytest.raises(ModelDomainError, match="distribution-tail range"):
        rate(2.0 * rate.distribution_mean_energy_limit_eV)


def test_argon_ionization_matches_independent_quadrature() -> None:
    root = Path(__file__).parents[1]
    data = load_chemistry(
        root / "plasma_global/examples/chemistry/argon_lxcat/chemistry.yaml"
    )
    cross_section = data.cross_sections["xs_ar_ionization_zenodo"]
    rate = compile_maxwell_rate(cross_section)
    energy, sigma = normalized_cross_section_curve(cross_section)
    sigma_scale = float(np.max(sigma))
    for mean in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        temperature = (2.0 / 3.0) * mean
        integral, _ = quad(
            lambda value, temperature=temperature: (
                value
                * float(np.interp(value, energy, sigma))
                / sigma_scale
                * math.exp(-value / temperature)
            ),
            float(energy[0]),
            float(energy[-1]),
            points=energy[1:-1],
            epsabs=1.0e-100,
            epsrel=1.0e-10,
            limit=2 * len(energy),
        )
        expected = (
            2.0
            / math.sqrt(math.pi)
            * math.sqrt(2.0 * E_CHARGE / ELECTRON_MASS_KG)
            * sigma_scale
            * integral
            / temperature**1.5
        )
        assert rate(mean) == pytest.approx(expected, rel=1.0e-10, abs=0.0)


def test_small_distribution_tail_cannot_hide_a_large_missing_rate_tail() -> None:
    sigma = 2.0e-20
    narrow = CrossSectionData(
        "narrow",
        "ionization",
        "Ar",
        99.0,
        99.0,
        np.array([99.0, 100.0]),
        np.array([sigma, sigma]),
    )
    with pytest.raises(ModelDomainError, match="insufficient energy support"):
        compile_maxwell_rate(narrow)(8.0)
    extended = CrossSectionData(
        "extended",
        "ionization",
        "Ar",
        99.0,
        99.0,
        np.array([99.0, 200.0]),
        np.array([sigma, sigma]),
    )
    temperature = (2.0 / 3.0) * 8.0
    expected = (
        sigma
        * math.sqrt(8.0 * E_CHARGE * temperature / (math.pi * ELECTRON_MASS_KG))
        * (1.0 + 99.0 / temperature)
        * math.exp(-99.0 / temperature)
    )
    assert compile_maxwell_rate(extended)(8.0) == pytest.approx(
        expected, rel=1e-7, abs=0.0
    )
