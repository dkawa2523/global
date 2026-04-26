from __future__ import annotations

import math

import pytest

from scripts.calibrate_rf_envelope import (
    RFEnvelopeCalibrationInput,
    estimate_rf_envelope_coefficients,
    recipe_snippet,
)


def test_rf_envelope_calibration_estimates_hf_coupling_from_net_power() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role='hf_source',
            frequency_Hz=13.56e6,
            forward_power_W=1000.0,
            reflected_power_W=100.0,
            absorbed_power_W=540.0,
        )
    )

    assert estimate['commanded_power_W'] == pytest.approx(900.0)
    assert estimate['coupling_efficiency'] == pytest.approx(0.6)
    assert estimate['frequency_Hz'] == pytest.approx(13.56e6)
    assert 'warnings' not in estimate


def test_rf_envelope_calibration_estimates_lf_impedance_and_bias_fraction() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role='lf_bias',
            frequency_Hz=2.0e6,
            commanded_power_W=200.0,
            absorbed_power_W=40.0,
            voltage_rms_V=100.0,
            current_rms_A=0.5,
            dc_self_bias_V=-49.497474683,
        )
    )
    snippet = recipe_snippet(estimate)

    assert estimate['coupling_efficiency'] == pytest.approx(0.2)
    assert estimate['effective_impedance_ohm'] == pytest.approx(200.0)
    assert estimate['self_bias_fraction'] == pytest.approx(0.35)
    assert snippet['role'] == 'lf_bias'
    assert snippet['effective_impedance_ohm'] == pytest.approx(200.0)
    assert 'commanded_power_W' not in snippet


def test_rf_envelope_calibration_warns_when_bias_targets_are_incomplete() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role='lf_bias',
            commanded_power_W=100.0,
            absorbed_power_W=20.0,
            voltage_rms_V=100.0,
        )
    )

    warnings = ' '.join(estimate['warnings'])
    assert 'effective_impedance_ohm' in warnings
    assert 'self_bias_fraction' in warnings
    assert math.isclose(estimate['coupling_efficiency'], 0.2)
