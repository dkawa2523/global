from __future__ import annotations

import numpy as np

from plasma_global.experimental.power import (
    CCPPowerPort,
    ICPPowerPort,
    RFEnvelopePort,
    SquarePulse,
)
from plasma_global.models.power import (
    CompiledPowerCommand,
    PowerState,
)


def _state(electron_density_m3: float = 2.0e17) -> PowerState:
    return PowerState(
        electron_density_m3=electron_density_m3,
        neutral_density_m3=2.4e20,
        electron_temperature_eV=3.0,
    )


def test_rf_envelope_is_finite_conservative_and_obeys_square_pulse() -> None:
    port = RFEnvelopePort(
        port_id="lf",
        zone_id="plasma",
        frequency_Hz=2.0e6,
        role="bias",
        coupling_efficiency=0.65,
        reduced_field_per_sqrt_W_Td=3.0,
        pulse=SquarePulse(repetition_Hz=1.0e3, duty_cycle=0.5),
    )
    command = CompiledPowerCommand(kind="power", power_W=100.0)
    on = port.evaluate(0.0, _state(), command)
    off = port.evaluate(0.75e-3, _state(), command)

    assert 0.0 <= on.electron_power_W <= on.observables["delivered_power_W"]
    assert on.observables["self_bias_V"] < 0.0
    assert np.isfinite([on.electron_power_W, *on.observables.values()]).all()
    assert off.electron_power_W == 0.0
    assert off.reduced_field_Td == 0.0


def test_ccp_power_and_ion_energy_increase_with_voltage() -> None:
    port = CCPPowerPort(
        port_id="ccp",
        zone_id="plasma",
        frequency_Hz=13.56e6,
        zone_volume_m3=0.02,
        powered_area_m2=0.02,
        grounded_area_m2=0.10,
        electrode_gap_m=0.04,
        dominant_ion_mass_kg=6.63e-26,
    )
    low = port.evaluate(
        0.0, _state(), CompiledPowerCommand(kind="voltage", voltage_V=100.0)
    )
    high = port.evaluate(
        0.0, _state(), CompiledPowerCommand(kind="voltage", voltage_V=300.0)
    )

    assert high.electron_power_W > low.electron_power_W >= 0.0
    assert (
        high.observables["mean_ion_energy_eV"] > low.observables["mean_ion_energy_eV"]
    )
    assert high.observables["self_bias_V"] < 0.0
    assert high.electron_power_W <= high.observables["delivered_power_W"]
    assert np.isfinite([high.electron_power_W, *high.observables.values()]).all()


def test_icp_coupling_trend_and_zone_split_conserve_absorbed_power() -> None:
    port = ICPPowerPort(
        port_id="icp",
        zone_id="source",
        downstream_zone_id="process",
        downstream_fraction=0.2,
    )
    command = CompiledPowerCommand(kind="power", power_W=300.0)
    low_density = port.evaluate(0.0, _state(1.0e15), command)
    high_density = port.evaluate(0.0, _state(3.0e17), command)
    distribution = port.power_by_zone(high_density)

    assert (
        high_density.observables["coupling_efficiency"]
        > low_density.observables["coupling_efficiency"]
    )
    assert np.isclose(
        sum(distribution.values()), high_density.observables["absorbed_power_W"]
    )
    assert (
        high_density.observables["absorbed_power_W"]
        + high_density.observables["reflected_power_W"]
        == high_density.observables["delivered_power_W"]
    )
    assert np.isfinite(
        [*distribution.values(), *high_density.observables.values()]
    ).all()
