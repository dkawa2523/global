from __future__ import annotations

import numpy as np

from plasma_global.models.gas_energy import (
    HeavyEnergyClosure,
    elastic_electron_heating_J_m3_s,
)


def test_heavy_energy_temperature_roundtrip_and_wall_relaxation() -> None:
    closure = HeavyEnergyClosure(
        cv_over_kb=np.array([1.5, 2.5]),
        wall_temperature_K=np.array([300.0]),
        wall_relaxation_s_inv=np.array([10.0]),
    )
    density = np.array([[2.0e20, 1.0e20]])
    energy = closure.energy_J_m3(density, np.array([600.0]))
    np.testing.assert_allclose(closure.temperature_K(density, energy), [600.0])
    np.testing.assert_allclose(
        closure.wall_exchange_J_m3_s(density, energy),
        -10.0 * (energy - closure.energy_J_m3(density, np.array([300.0]))),
    )


def test_elastic_transfer_has_equal_sign_of_temperature_difference() -> None:
    common = {
        "electron_density_m3": 1.0e16,
        "neutral_densities_m3": np.array([2.0e20]),
        "neutral_masses_kg": np.array([6.63e-26]),
        "momentum_rate_coefficients_m3_s": np.array([1.0e-13]),
    }
    heating = elastic_electron_heating_J_m3_s(
        electron_temperature_eV=3.0, gas_temperature_K=300.0, **common
    )
    cooling = elastic_electron_heating_J_m3_s(
        electron_temperature_eV=0.01, gas_temperature_K=1000.0, **common
    )
    assert heating > 0.0
    assert cooling < 0.0
