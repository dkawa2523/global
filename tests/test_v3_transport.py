from __future__ import annotations

import numpy as np

from plasma_global.core.transport import CompiledTransport, SegmentTransport


def test_interzone_transport_conserves_volume_integrals() -> None:
    network = CompiledTransport(
        volumes_m3=np.array([1.0, 3.0]),
        pump_frequency_s_inv=np.zeros(2),
        edge_from=np.array([0]),
        edge_to=np.array([1]),
        edge_conductance_m3_s=np.array([0.2]),
        n_species=2,
    )
    forcing = SegmentTransport(np.zeros((2, 2)), np.zeros(2))
    density = np.array([[10.0, 20.0], [1.0, 2.0]])
    electron = np.array([5.0, 1.0])
    heavy = np.array([7.0, 2.0])

    density_rhs, electron_rhs, heavy_rhs = network.evaluate(
        density, forcing, electron_energy_J_m3=electron, heavy_energy_J_m3=heavy
    )

    np.testing.assert_allclose(network.volumes_m3 @ density_rhs, 0.0, atol=1.0e-15)
    assert electron_rhs is not None and abs(network.volumes_m3 @ electron_rhs) < 1.0e-15
    assert heavy_rhs is not None and abs(network.volumes_m3 @ heavy_rhs) < 1.0e-15


def test_inlet_and_pump_are_compiled_sources() -> None:
    network = CompiledTransport(
        volumes_m3=np.array([2.0]),
        pump_frequency_s_inv=np.array([0.5]),
        edge_from=np.array([], dtype=int),
        edge_to=np.array([], dtype=int),
        edge_conductance_m3_s=np.array([]),
        n_species=1,
    )
    forcing = SegmentTransport(np.array([[3.0]]), np.array([4.0]))
    density_rhs, electron_rhs, heavy_rhs = network.evaluate(
        np.array([[2.0]]),
        forcing,
        electron_energy_J_m3=np.array([6.0]),
        heavy_energy_J_m3=np.array([8.0]),
    )
    np.testing.assert_allclose(density_rhs, [[2.0]])
    np.testing.assert_allclose(electron_rhs, [-3.0])
    np.testing.assert_allclose(heavy_rhs, [0.0])
