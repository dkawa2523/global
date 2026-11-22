from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from plasma_global import case_from_mapping, compile_case, load_case, simulate
from plasma_global.input._schema_power import (
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
)
from tools._process import run_process
from tools.calibration.rf_envelope import (
    RFEnvelopeCalibrationInput,
    canonical_fragments,
    estimate_rf_envelope_coefficients,
)

FIXTURE = Path(__file__).parent / "fixtures" / "minimal" / "case.yaml"


def test_rf_calibration_fits_real_power_independently_of_apparent_power() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="bias",
            frequency_Hz=2.0e6,
            commanded_power_W=100.0,
            absorbed_power_W=80.0,
            voltage_rms_V=200.0,
            current_rms_A=2.0,
            dc_self_bias_V=-40.0,
        )
    )
    fragments = canonical_fragments(estimate)
    model = ExperimentalRFEnvelopeModel.model_validate(fragments["power_model"])
    command = ExperimentalRFEnvelopeCommand.model_validate(
        {"kind": model.kind, **fragments["power_command"]}
    )
    assert model.control == "voltage"
    assert model.coupling_efficiency == pytest.approx(0.8)
    assert model.effective_impedance_ohm == pytest.approx(400.0)
    assert command.voltage_rms_V == pytest.approx(200.0)
    assert estimate["apparent_power_VA"] == pytest.approx(400.0)
    assert estimate["impedance_magnitude_ohm"] == pytest.approx(100.0)
    assert estimate["power_factor"] == pytest.approx(0.25)
    assert estimate["self_bias_fraction"] == pytest.approx(1.0 / np.sqrt(50.0))
    assert any("current and phase are not fitted" in x for x in estimate["warnings"])


def test_rf_calibration_cli_fragments_reproduce_measurements_in_public_case(
    tmp_path: Path,
) -> None:
    calibration = tmp_path / "calibration.yaml"
    completed = run_process(
        [
            sys.executable,
            "-m",
            "tools.calibration.rf_envelope",
            "--role",
            "bias",
            "--frequency-Hz",
            "2e6",
            "--commanded-power-W",
            "100",
            "--absorbed-power-W",
            "80",
            "--voltage-rms-V",
            "200",
            "--current-rms-A",
            "2",
            "--dc-self-bias-V=-40",
            "--write",
            str(calibration),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert str(calibration.resolve()) in completed.stdout
    fragments = yaml.safe_load(calibration.read_text(encoding="utf-8"))
    case = load_case(FIXTURE)
    for control in ("voltage", "absorbed_power"):
        data = case.to_dict()
        model = dict(fragments["power_model"])
        command = fragments["power_command"]
        if control == "absorbed_power":
            model.pop("voltage_rms_V")
            model.update(control=control, absorbed_power_W=80.0)
            command = {"absorbed_power_W": 80.0}
        data["reactor"]["power_ports"]["source"]["model"] = model
        data["run"]["steps"][0]["commands"]["power_ports"]["source"] = command
        expected = {
            "port_absorbed_power_W[source]": 80.0,
            "port_delivered_power_W[source]": 100.0,
            "port_voltage_rms_V[source]": 200.0,
            "port_self_bias_V[source]": -40.0,
        }
        data["output"] = {"profile": "minimal", "observables": list(expected)}
        prepared = compile_case(
            case_from_mapping(data, base_directory=case.base_directory)
        )
        result = simulate(prepared)
        assert result.status.success
        for series, value in expected.items():
            np.testing.assert_allclose(result.series(series), value, rtol=1e-12)


def test_rf_calibration_normalizes_numpy_scalars_for_yaml_round_trip() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="bias",
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


def test_rf_calibration_uses_one_net_real_power_definition() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="source",
            commanded_power_W=80.0,
            delivered_power_W=80.0,
            forward_power_W=100.0,
            reflected_power_W=20.0,
            absorbed_power_W=64.0,
            voltage_rms_V=200.0,
        )
    )
    assert estimate["delivered_power_W"] == pytest.approx(80.0)
    assert estimate["coupling_efficiency"] == pytest.approx(0.8)
    assert estimate["effective_impedance_ohm"] == pytest.approx(500.0)
    canonical_fragments(estimate)


def test_rf_calibration_current_only_cannot_identify_real_power_coupling() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(role="bias", voltage_rms_V=200.0, current_rms_A=2.0)
    )
    assert "effective_impedance_ohm" not in estimate
    assert "coupling_efficiency" not in estimate
    assert estimate["impedance_magnitude_ohm"] == pytest.approx(100.0)
    assert any("not identified" in warning for warning in estimate["warnings"])
    with pytest.raises(ValueError, match="requires net delivered and absorbed power"):
        canonical_fragments(estimate)


@pytest.mark.parametrize("role", ["hf_source", "lf_bias"])
def test_rf_calibration_uses_the_public_model_role_names(role: str) -> None:
    with pytest.raises(ValueError, match="role must be source or bias"):
        RFEnvelopeCalibrationInput(role=role)


def test_rf_calibration_zero_absorption_is_an_identified_efficiency() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="source",
            voltage_rms_V=200.0,
            delivered_power_W=100.0,
            absorbed_power_W=0.0,
        )
    )
    assert estimate["coupling_efficiency"] == 0.0
    fragments = canonical_fragments(estimate)
    assert fragments["power_model"]["coupling_efficiency"] == 0.0
    assert fragments["power_command"]["voltage_rms_V"] == 200.0


def test_rf_calibration_off_is_known_zero_with_unidentified_coefficients() -> None:
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="bias",
            commanded_power_W=0.0,
            absorbed_power_W=0.0,
            voltage_rms_V=0.0,
            current_rms_A=0.0,
            dc_self_bias_V=0.0,
        )
    )
    assert estimate["delivered_power_W"] == 0.0
    assert estimate["absorbed_power_W"] == 0.0
    assert estimate["voltage_rms_V"] == 0.0
    assert "coupling_efficiency" not in estimate
    assert "effective_impedance_ohm" not in estimate
    assert "self_bias_fraction" not in estimate
    assert "power_factor" not in estimate
    fragments = canonical_fragments(estimate)
    assert fragments["power_command"]["voltage_rms_V"] == 0.0
    assert "coupling_efficiency" not in fragments["power_model"]
    assert any("zero/off" in warning for warning in estimate["warnings"])


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"frequency_Hz": 0.0}, "frequency_Hz must be positive"),
        ({"delivered_power_W": -1.0}, "delivered_power_W must be nonnegative"),
        ({"current_rms_A": -1.0}, "current_rms_A must be nonnegative"),
        ({"voltage_rms_V": float("inf")}, "voltage_rms_V must be finite"),
        ({"absorbed_power_W": float("nan")}, "absorbed_power_W must be finite"),
        ({"reflected_power_W": 1.0}, "requires forward_power_W"),
        (
            {"forward_power_W": 1.0, "reflected_power_W": 2.0},
            "must not exceed forward_power_W",
        ),
        (
            {"delivered_power_W": 50.0, "commanded_power_W": 100.0},
            "same net real power",
        ),
        (
            {"delivered_power_W": 0.0, "commanded_power_W": 1.0},
            "same net real power",
        ),
        (
            {
                "forward_power_W": 100.0,
                "reflected_power_W": 20.0,
                "delivered_power_W": 50.0,
            },
            "same net real power",
        ),
        (
            {"absorbed_power_W": 81.0, "delivered_power_W": 80.0},
            "must not exceed net delivered",
        ),
        (
            {"absorbed_power_W": 1.0, "delivered_power_W": 0.0},
            "must not exceed net delivered",
        ),
        (
            {"voltage_rms_V": 100.0, "current_rms_A": 0.1, "delivered_power_W": 20.0},
            "must not exceed RMS V",
        ),
        ({"current_rms_A": 1.0}, "requires voltage_rms_V"),
        (
            {"voltage_rms_V": 0.0, "absorbed_power_W": 1.0},
            "positive absorbed power requires positive RMS voltage",
        ),
        (
            {"voltage_rms_V": 0.0, "delivered_power_W": 1.0},
            "positive delivered power requires positive RMS voltage",
        ),
        ({"dc_self_bias_V": -1.0}, "requires voltage_rms_V"),
        (
            {"voltage_rms_V": 100.0, "dc_self_bias_V": 1.0},
            "cannot reproduce positive DC self-bias",
        ),
        (
            {"voltage_rms_V": 0.0, "dc_self_bias_V": -1.0},
            "zero RMS voltage cannot have nonzero DC self-bias",
        ),
    ],
)
def test_rf_calibration_rejects_invalid_or_contradictory_measurements(
    values: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_rf_envelope_coefficients(
            RFEnvelopeCalibrationInput(role="bias", **values)
        )


def test_rf_calibration_source_role_rejects_unrepresentable_nonzero_self_bias() -> None:
    with pytest.raises(ValueError, match="source-role model defines zero self-bias"):
        estimate_rf_envelope_coefficients(
            RFEnvelopeCalibrationInput(
                role="source", voltage_rms_V=100.0, dc_self_bias_V=-20.0
            )
        )


def test_rf_calibration_rejects_power_off_fragment_with_positive_delivered_power() -> (
    None
):
    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role="source", absorbed_power_W=0.0, delivered_power_W=100.0
        )
    )
    assert estimate["coupling_efficiency"] == 0.0
    with pytest.raises(ValueError, match="off command cannot reproduce positive"):
        canonical_fragments(estimate)


def test_rf_calibration_cli_explains_invalid_measurement_without_emitting_yaml() -> (
    None
):
    completed = run_process(
        [
            sys.executable,
            "-m",
            "tools.calibration.rf_envelope",
            "--role",
            "source",
            "--forward-power-W",
            "1",
            "--reflected-power-W",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
    )
    assert completed.returncode == 2
    assert "must not exceed forward_power_W" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not completed.stdout
