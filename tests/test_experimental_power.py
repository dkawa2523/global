from __future__ import annotations

import numpy as np
import pytest

from plasma_global.errors import ModelDomainError
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


def test_rf_power_command_is_absorbed_power_not_delivered_power() -> None:
    port = RFEnvelopePort(
        port_id="source",
        zone_id="plasma",
        frequency_Hz=13.56e6,
        coupling_efficiency=0.65,
        effective_impedance_ohm=50.0,
    )

    result = port.evaluate(
        0.0,
        _state(),
        CompiledPowerCommand(kind="power", power_W=100.0),
    )

    assert result.electron_power_W == pytest.approx(100.0)
    assert result.observables["absorbed_power_W"] == pytest.approx(100.0)
    assert result.observables["delivered_power_W"] == pytest.approx(100.0 / 0.65)


def test_rf_rejects_nonfinite_source_side_diagnostics() -> None:
    port = RFEnvelopePort(
        port_id="source",
        zone_id="plasma",
        frequency_Hz=13.56e6,
        coupling_efficiency=1.0e-320,
        effective_impedance_ohm=50.0,
    )

    with pytest.raises(ModelDomainError, match="observables must be finite"):
        port.evaluate(
            0.0,
            _state(),
            CompiledPowerCommand(kind="power", power_W=100.0),
        )


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
        high.observables["estimated_mean_ion_energy_eV"]
        > low.observables["estimated_mean_ion_energy_eV"]
    )
    assert (
        high.observables["mean_ion_energy_eV"]
        == high.observables["estimated_mean_ion_energy_eV"]
    )
    assert high.observables["self_bias_V"] < 0.0
    assert high.electron_power_W <= high.observables["delivered_power_W"]
    assert high.electron_power_W == high.observables["delivered_power_W"]
    assert high.observables["apparent_power_VA"] >= high.electron_power_W
    assert np.isfinite([high.electron_power_W, *high.observables.values()]).all()


def test_ccp_power_command_is_absorbed_power_not_delivered_power() -> None:
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

    result = port.evaluate(
        0.0,
        _state(),
        CompiledPowerCommand(kind="power", power_W=100.0),
    )

    assert result.electron_power_W == pytest.approx(100.0)
    assert result.observables["absorbed_power_W"] == pytest.approx(100.0)
    assert result.observables["delivered_power_W"] == pytest.approx(100.0)
    assert result.observables["apparent_power_VA"] >= 100.0
    impedance = np.hypot(
        result.observables["bulk_resistance_ohm"],
        result.observables["sheath_reactance_ohm"],
    )
    assert result.observables["rf_voltage_rms_V"] == pytest.approx(
        result.observables["rf_current_rms_A"] * impedance,
        rel=1.0e-9,
    )


def test_ccp_reduced_field_uses_the_resistive_bulk_voltage() -> None:
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
    state = _state()
    result = port.evaluate(
        0.0,
        state,
        CompiledPowerCommand(kind="voltage", voltage_V=300.0),
    )

    bulk_voltage = (
        result.observables["rf_current_rms_A"]
        * result.observables["bulk_resistance_ohm"]
    )
    expected_field_Td = (
        bulk_voltage / port.electrode_gap_m / state.neutral_density_m3 / 1.0e-21
    )
    assert result.reduced_field_Td == pytest.approx(expected_field_Td)


@pytest.mark.parametrize("kind", ["power", "voltage"])
def test_ccp_rejects_positive_commands_without_electrons(kind: str) -> None:
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
    command = (
        CompiledPowerCommand(kind="power", power_W=100.0)
        if kind == "power"
        else CompiledPowerCommand(kind="voltage", voltage_V=100.0)
    )

    with pytest.raises(ModelDomainError, match="positive electron density"):
        port.evaluate(0.0, _state(0.0), command)


def test_ccp_zero_command_has_a_cold_zero_electron_boundary() -> None:
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
    zero = port.evaluate(
        0.0,
        _state(0.0),
        CompiledPowerCommand(kind="power", power_W=0.0),
    )
    assert zero.electron_power_W == 0.0
    assert zero.reduced_field_Td == 0.0
    assert zero.observables["estimated_mean_ion_energy_eV"] == 0.0
    assert zero.observables["plasma_potential_V"] == 0.0
    assert zero.observables["ion_flux_m2_s"] == 0.0


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


def test_icp_has_continuous_zero_density_and_zero_power_limits() -> None:
    port = ICPPowerPort(port_id="icp", zone_id="source")
    vacuum = port.evaluate(
        0.0,
        _state(0.0),
        CompiledPowerCommand(kind="power", power_W=100.0),
    )
    weak = port.evaluate(
        0.0,
        _state(),
        CompiledPowerCommand(kind="power", power_W=1.0e-12),
    )

    assert vacuum.electron_power_W == 0.0
    assert vacuum.reduced_field_Td == 0.0
    assert vacuum.observables["reflected_power_W"] == pytest.approx(100.0)
    assert weak.electron_power_W < 1.0e-12
    assert weak.reduced_field_Td < 1.0e-9
    assert weak.observables["plasma_potential_V"] < 1.0e-9


def test_icp_absorbed_power_is_continuous_across_the_e_h_transition() -> None:
    port = ICPPowerPort(port_id="icp", zone_id="source")
    delivered_power_W = 300.0
    transition_density = (
        -np.log(1.0 - 0.55 / np.sqrt(delivered_power_W / 150.0)) / 2.2 * 1.0e17
    )
    below = port.evaluate(
        0.0,
        _state(transition_density * (1.0 - 1.0e-6)),
        CompiledPowerCommand(kind="power", power_W=delivered_power_W),
    )
    above = port.evaluate(
        0.0,
        _state(transition_density * (1.0 + 1.0e-6)),
        CompiledPowerCommand(kind="power", power_W=delivered_power_W),
    )

    assert below.observables["mode_H"] < 0.5 < above.observables["mode_H"]
    assert above.observables["mode_H"] == pytest.approx(
        below.observables["mode_H"], abs=1.0e-5
    )
    assert above.observables["absorbed_power_W"] == pytest.approx(
        below.observables["absorbed_power_W"], rel=1.0e-5
    )
