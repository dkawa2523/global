from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsFloat

import yaml

from plasma_global.input.schema import (
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
)


@dataclass
class RFEnvelopeCalibrationInput:
    role: str
    frequency_Hz: float | None = None
    commanded_power_W: float | None = None
    absorbed_power_W: float | None = None
    forward_power_W: float | None = None
    reflected_power_W: float | None = None
    voltage_rms_V: float | None = None
    current_rms_A: float | None = None
    delivered_power_W: float | None = None
    dc_self_bias_V: float | None = None


def _python_float(value: SupportsFloat) -> float:
    return float(value)


def _positive(value: float | None) -> float | None:
    if value is None:
        return None
    value = _python_float(value)
    return value if value > 0.0 else None


def _net_commanded_power(inp: RFEnvelopeCalibrationInput) -> float | None:
    commanded = _positive(inp.commanded_power_W)
    if commanded is not None:
        return commanded
    if inp.forward_power_W is not None:
        reflected = max(_python_float(inp.reflected_power_W or 0.0), 0.0)
        return max(_python_float(inp.forward_power_W) - reflected, 0.0)
    return None


def _power_coefficients(inp: RFEnvelopeCalibrationInput) -> dict[str, float]:
    commanded = _net_commanded_power(inp)
    coefficients: dict[str, float] = {}
    if inp.frequency_Hz is not None:
        coefficients["frequency_Hz"] = _python_float(inp.frequency_Hz)
    if commanded is not None:
        coefficients["commanded_power_W"] = commanded

    absorbed = _positive(inp.absorbed_power_W)
    if absorbed is not None:
        coefficients["absorbed_power_W"] = absorbed
    if commanded is not None and commanded > 0.0 and absorbed is not None:
        coefficients["coupling_efficiency"] = max(absorbed / commanded, 0.0)
    return coefficients


def _voltage_coefficients(inp: RFEnvelopeCalibrationInput) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    voltage = _positive(inp.voltage_rms_V)
    current = _positive(inp.current_rms_A)
    delivered = _positive(inp.delivered_power_W)
    if voltage is not None:
        coefficients["voltage_rms_V"] = voltage
        if current is not None:
            coefficients["effective_impedance_ohm"] = voltage / current
            coefficients["current_rms_A"] = current
        elif delivered is not None:
            coefficients["effective_impedance_ohm"] = voltage * voltage / delivered
        if inp.dc_self_bias_V is not None:
            dc_self_bias_V = _python_float(inp.dc_self_bias_V)
            coefficients["self_bias_fraction"] = abs(dc_self_bias_V) / (
                math.sqrt(2.0) * voltage
            )
            coefficients["dc_self_bias_V"] = dc_self_bias_V
    return coefficients


def _calibration_warnings(
    inp: RFEnvelopeCalibrationInput, coefficients: Mapping[str, Any]
) -> list[str]:
    warnings: list[str] = []
    if "coupling_efficiency" not in coefficients:
        warnings.append(
            "coupling_efficiency was not estimated; provide commanded and "
            + "absorbed power."
        )
    elif coefficients["coupling_efficiency"] > 1.0:
        warnings.append(
            "coupling_efficiency is greater than 1; check commanded/absorbed "
            + "power definitions."
        )
    if (
        "voltage_rms_V" in coefficients
        and "effective_impedance_ohm" not in coefficients
    ):
        warnings.append(
            "effective_impedance_ohm was not estimated; provide RMS current or "
            + "delivered power."
        )
    role = inp.role.lower()
    if (
        ("bias" in role or "lf" in role)
        and "voltage_rms_V" in coefficients
        and "self_bias_fraction" not in coefficients
    ):
        warnings.append(
            "self_bias_fraction was not estimated; provide measured DC self-bias."
        )
    return warnings


def estimate_rf_envelope_coefficients(
    inp: RFEnvelopeCalibrationInput,
) -> dict[str, Any]:
    coefficients: dict[str, Any] = {"role": inp.role}
    coefficients.update(_power_coefficients(inp))
    coefficients.update(_voltage_coefficients(inp))
    warnings = _calibration_warnings(inp, coefficients)
    if warnings:
        coefficients["warnings"] = warnings
    return coefficients


def canonical_fragments(coefficients: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return schema-validated v3 model and recipe-command fragments."""

    role = "bias" if "bias" in str(coefficients.get("role", "")).lower() else "source"
    shared_keys = (
        "frequency_Hz",
        "coupling_efficiency",
        "effective_impedance_ohm",
        "self_bias_fraction",
    )
    shared = {key: coefficients[key] for key in shared_keys if key in coefficients}
    model_data: dict[str, Any] = {
        "kind": "experimental.rf_envelope",
        "role": role,
        **shared,
    }
    command_data: dict[str, Any] = {"kind": "experimental.rf_envelope"}
    voltage = coefficients.get("voltage_rms_V")
    absorbed = coefficients.get("absorbed_power_W")
    if voltage is not None:
        model_data.update(
            control="voltage",
            default_voltage_rms_V=voltage,
        )
        command_data.update(
            voltage_rms_V=voltage,
        )
    elif absorbed is not None:
        model_data.update(
            control="absorbed_power",
            default_absorbed_power_W=absorbed,
        )
        command_data.update(
            absorbed_power_W=absorbed,
        )

    model = ExperimentalRFEnvelopeModel.model_validate(model_data)
    command = ExperimentalRFEnvelopeCommand.model_validate(command_data)
    return {
        "power_model": model.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
        ),
        "power_command": command.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate rf_envelope calibration coefficients from measured RF "
            + "quantities."
        )
    )
    parser.add_argument(
        "--role", required=True, help="Port role, for example hf_source or lf_bias."
    )
    parser.add_argument("--frequency-Hz", type=float, default=None)
    parser.add_argument(
        "--commanded-power-W",
        type=float,
        default=None,
        help="Generator, net delivered, or recipe command power.",
    )
    parser.add_argument("--forward-power-W", type=float, default=None)
    parser.add_argument("--reflected-power-W", type=float, default=None)
    parser.add_argument("--absorbed-power-W", type=float, default=None)
    parser.add_argument("--voltage-rms-V", type=float, default=None)
    parser.add_argument("--current-rms-A", type=float, default=None)
    parser.add_argument("--delivered-power-W", type=float, default=None)
    parser.add_argument("--dc-self-bias-V", type=float, default=None)
    parser.add_argument(
        "--write", type=Path, default=None, help="Optional YAML output path."
    )
    args = parser.parse_args()

    estimate = estimate_rf_envelope_coefficients(
        RFEnvelopeCalibrationInput(
            role=args.role,
            frequency_Hz=args.frequency_Hz,
            commanded_power_W=args.commanded_power_W,
            absorbed_power_W=args.absorbed_power_W,
            forward_power_W=args.forward_power_W,
            reflected_power_W=args.reflected_power_W,
            voltage_rms_V=args.voltage_rms_V,
            current_rms_A=args.current_rms_A,
            delivered_power_W=args.delivered_power_W,
            dc_self_bias_V=args.dc_self_bias_V,
        )
    )
    payload = {
        "estimated_coefficients": estimate,
        **canonical_fragments(estimate),
    }
    text = yaml.safe_dump(payload, sort_keys=False)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
        print(args.write.resolve())
    else:
        print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
