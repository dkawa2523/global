from __future__ import annotations

import numpy as np
import pytest
import yaml

from plasma_global.input._resolved_case import (
    ExperimentalRFEnvelopeCommand,
)
from plasma_global.input.schema import RFPower
from tools.calibration.rf_envelope import (
    RFEnvelopeCalibrationInput,
    canonical_fragments,
    estimate_rf_envelope_coefficients,
)


def test_rf_calibration_emits_schema_valid_v4_fragments() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="lf_bias",
            frequency_Hz=2.0e6,
            commanded_power_W=100.0,
            absorbed_power_W=80.0,
            voltage_rms_V=200.0,
            current_rms_A=2.0,
            dc_self_bias_V=-40.0,
        )
    )

    fragments = canonical_fragments(estimate)
    model = RFPower.model_validate(fragments["power_model"])
    command = ExperimentalRFEnvelopeCommand.model_validate(
        {"kind": model.kind, **fragments["power_command"]}
    )

    assert model.kind == "experimental.rf_envelope"
    assert model.control == "voltage"
    assert model.coupling_efficiency == pytest.approx(0.8)
    assert model.effective_impedance_ohm == pytest.approx(100.0)
    assert command.kind == model.kind
    assert command.voltage_rms_V == pytest.approx(200.0)
    assert tuple(estimate) == (
        "role",
        "frequency_Hz",
        "commanded_power_W",
        "absorbed_power_W",
        "coupling_efficiency",
        "voltage_rms_V",
        "effective_impedance_ohm",
        "current_rms_A",
        "self_bias_fraction",
        "dc_self_bias_V",
    )
    assert estimate["self_bias_fraction"] == pytest.approx(1.0 / np.sqrt(50.0))
    assert fragments == {
        "power_model": {
            "kind": "experimental.rf_envelope",
            "role": "bias",
            "frequency_Hz": 2.0e6,
            "control": "voltage",
            "voltage_rms_V": 200.0,
            "effective_impedance_ohm": 100.0,
            "coupling_efficiency": 0.8,
            "self_bias_fraction": pytest.approx(1.0 / np.sqrt(50.0)),
        },
        "power_command": {
            "voltage_rms_V": 200.0,
        },
    }


def test_rf_calibration_normalizes_numpy_scalars_for_yaml_round_trip() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="lf_bias",
            frequency_Hz=np.float32(2.0e6),
            voltage_rms_V=np.float32(200.0),
            current_rms_A=np.float32(2.0),
            dc_self_bias_V=np.float32(-40.0),
        )
    )

    restored = yaml.safe_load(yaml.safe_dump(estimate))
    assert restored["frequency_Hz"] == pytest.approx(2.0e6)
    assert restored["dc_self_bias_V"] == pytest.approx(-40.0)
    assert type(estimate["frequency_Hz"]) is float
    assert type(estimate["dc_self_bias_V"]) is float


@pytest.mark.parametrize(
    ("calibration_input", "expected"),
    [
        pytest.param(
            RFEnvelopeCalibrationInput(
                role="source",
                forward_power_W=100.0,
                reflected_power_W=20.0,
                absorbed_power_W=100.0,
                voltage_rms_V=200.0,
                delivered_power_W=50.0,
            ),
            {
                "role": "source",
                "commanded_power_W": 80.0,
                "absorbed_power_W": 100.0,
                "coupling_efficiency": 1.25,
                "voltage_rms_V": 200.0,
                "effective_impedance_ohm": 800.0,
                "warnings": [
                    "coupling_efficiency is greater than 1; check commanded/absorbed "
                    "power definitions."
                ],
            },
            id="forward-minus-reflected",
        ),
        pytest.param(
            RFEnvelopeCalibrationInput(role="lf_bias", voltage_rms_V=100.0),
            {
                "role": "lf_bias",
                "voltage_rms_V": 100.0,
                "warnings": [
                    "coupling_efficiency was not estimated; provide commanded and "
                    "absorbed power.",
                    "effective_impedance_ohm was not estimated; provide RMS current "
                    "or delivered power.",
                    "self_bias_fraction was not estimated; provide measured DC "
                    "self-bias.",
                ],
            },
            id="missing-bias-measurements",
        ),
    ],
)
def test_rf_calibration_preserves_fallbacks_and_warning_order(
    calibration_input: RFEnvelopeCalibrationInput, expected: dict[str, object]
) -> None:
    assert estimate_rf_envelope_coefficients(calibration_input) == expected
