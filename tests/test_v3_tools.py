from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from plasma_global.input.schema import (
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
)
from plasma_global.models.kinetics import TabulatedElectronKinetics
from tools.calibration.rf_envelope import (
    RFEnvelopeCalibrationInput,
    canonical_fragments,
    estimate_rf_envelope_coefficients,
)
from tools.importers.rate_table import build_rate_table_h5


def _write_rate_table_inputs(directory: Path) -> None:
    directory.mkdir()
    (directory / "rates.csv").write_text(
        "EoverN_Td,ionize\n20.0,3.0e-16\n10.0,1.0e-16\n",
        encoding="utf-8",
    )
    (directory / "transport.csv").write_text(
        "EoverN_Td,mean_energy_eV,mobility_m2_V_s\n10.0,1.0,0.20\n20.0,3.0,0.10\n",
        encoding="utf-8",
    )
    (directory / "metadata.yaml").write_text(
        "source: regression\nrevision: 2\nvalidated: true\nnested:\n  ignored: true\n",
        encoding="utf-8",
    )


def test_rate_table_importer_writes_v3_electron_kinetics(tmp_path: Path) -> None:
    source = tmp_path / "table"
    _write_rate_table_inputs(source)

    output = build_rate_table_h5(source, tmp_path / "kinetics.h5")
    table = TabulatedElectronKinetics.from_hdf5(
        output,
        lookup="local_field",
        required_rate_ids=("ionize",),
    )
    evaluated = table.evaluate(reduced_field_Td=15.0)

    assert evaluated.mean_energy_eV == pytest.approx(2.0)
    assert evaluated.mobility_m2_V_s == pytest.approx(0.15)
    assert evaluated.rate_coefficients["ionize"] == pytest.approx(2.0e-16)
    with h5py.File(output, "r") as handle:
        assert handle.attrs["format"] == "plasma_global_electron_kinetics"
        assert handle.attrs["format_version"] == 1
        assert handle.attrs["grid_column"] == "EoverN_Td"
        assert handle.attrs["source"] == "regression"
        assert handle.attrs["revision"] == 2
        assert bool(handle.attrs["validated"])
        assert "nested" not in handle.attrs
        np.testing.assert_array_equal(handle["mean_energy_eV"], [1.0, 3.0])
        np.testing.assert_array_equal(handle["effective_field_Td"], [10.0, 20.0])
        np.testing.assert_array_equal(handle["mobility_m2_V_s"], [0.2, 0.1])
        np.testing.assert_array_equal(
            handle["rate_coefficients"]["ionize"], [1.0e-16, 3.0e-16]
        )


def test_rate_table_importer_rejects_mismatched_grids(tmp_path: Path) -> None:
    source = tmp_path / "table"
    _write_rate_table_inputs(source)
    (source / "transport.csv").write_text(
        "EoverN_Td,mean_energy_eV,mobility_m2_V_s\n10.0,1.0,0.20\n30.0,3.0,0.10\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=r"rates\.csv and transport\.csv must contain the same grid"
    ):
        build_rate_table_h5(source, tmp_path / "kinetics.h5")


def test_rf_calibration_emits_schema_valid_v3_fragments() -> None:
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
    model = ExperimentalRFEnvelopeModel.model_validate(fragments["power_model"])
    command = ExperimentalRFEnvelopeCommand.model_validate(fragments["power_command"])

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
            "default_voltage_rms_V": 200.0,
            "effective_impedance_ohm": 100.0,
            "coupling_efficiency": 0.8,
            "self_bias_fraction": pytest.approx(1.0 / np.sqrt(50.0)),
        },
        "power_command": {
            "kind": "experimental.rf_envelope",
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
