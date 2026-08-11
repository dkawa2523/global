from __future__ import annotations

import numpy as np

from plasma_global.experimental.eedf import ApproximateTwoTermEEDF, ElectronCollision


def _argon_model() -> ApproximateTwoTermEEDF:
    energy_eV = np.array([0.0, 0.5, 2.0, 10.0, 30.0, 100.0])
    return ApproximateTwoTermEEDF(
        collisions=(
            ElectronCollision(
                collision_id="Ar momentum",
                target_species="Ar",
                kind="momentum",
                energy_eV=energy_eV,
                cross_section_m2=np.array(
                    [3.0e-19, 2.8e-19, 2.0e-19, 1.2e-19, 1.0e-19, 1.0e-19]
                ),
            ),
            ElectronCollision(
                collision_id="Ar excitation",
                target_species="Ar",
                kind="inelastic",
                energy_eV=energy_eV,
                cross_section_m2=np.array([0.0, 0.0, 0.0, 0.0, 1.8e-20, 2.0e-20]),
                energy_loss_eV=11.55,
            ),
        ),
        energy_points=128,
    )


def test_approximate_two_term_returns_normalized_finite_quantities() -> None:
    result = _argon_model().evaluate(80.0, {"Ar": 2.4e20})

    assert np.isclose(np.trapezoid(result.distribution_eV_inv, result.energy_eV), 1.0)
    assert np.all(result.distribution_eV_inv >= 0.0)
    assert np.all(np.isfinite(result.distribution_eV_inv))
    assert np.all(
        np.isfinite(
            [
                result.mean_energy_eV,
                result.mobility_m2_V_s,
                result.diffusion_m2_s,
                result.collisional_loss_W_per_electron,
                *result.rate_coefficients_m3_s.values(),
            ]
        )
    )
    assert min(result.rate_coefficients_m3_s.values()) >= 0.0


def test_approximate_two_term_heats_and_activates_excitation_with_field() -> None:
    model = _argon_model()
    low = model.evaluate(10.0, {"Ar": 2.4e20})
    high = model.evaluate(300.0, {"Ar": 2.4e20})

    assert high.mean_energy_eV > low.mean_energy_eV
    assert (
        high.rate_coefficients_m3_s["Ar excitation"]
        > low.rate_coefficients_m3_s["Ar excitation"]
    )
    assert high.collisional_loss_W_per_electron > low.collisional_loss_W_per_electron
