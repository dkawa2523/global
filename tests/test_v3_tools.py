from __future__ import annotations

from pathlib import Path

import h5py
import pytest

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
