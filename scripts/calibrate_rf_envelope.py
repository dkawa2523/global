from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


def _positive(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if value > 0.0 else None


def _net_commanded_power(inp: RFEnvelopeCalibrationInput) -> float | None:
    if _positive(inp.commanded_power_W) is not None:
        return float(inp.commanded_power_W)
    if inp.forward_power_W is not None:
        reflected = max(float(inp.reflected_power_W or 0.0), 0.0)
        return max(float(inp.forward_power_W) - reflected, 0.0)
    return None


def estimate_rf_envelope_coefficients(inp: RFEnvelopeCalibrationInput) -> dict[str, Any]:
    commanded = _net_commanded_power(inp)
    out: dict[str, Any] = {
        'role': inp.role,
    }
    if inp.frequency_Hz is not None:
        out['frequency_Hz'] = float(inp.frequency_Hz)
    if commanded is not None:
        out['commanded_power_W'] = commanded

    absorbed = _positive(inp.absorbed_power_W)
    if commanded is not None and commanded > 0.0 and absorbed is not None:
        out['coupling_efficiency'] = max(absorbed / commanded, 0.0)

    voltage = _positive(inp.voltage_rms_V)
    current = _positive(inp.current_rms_A)
    delivered = _positive(inp.delivered_power_W)
    if voltage is not None:
        out['voltage_rms_V'] = voltage
        if current is not None:
            out['effective_impedance_ohm'] = voltage / current
            out['current_rms_A'] = current
        elif delivered is not None:
            out['effective_impedance_ohm'] = voltage * voltage / delivered
        if inp.dc_self_bias_V is not None:
            out['self_bias_fraction'] = abs(float(inp.dc_self_bias_V)) / (math.sqrt(2.0) * voltage)
            out['dc_self_bias_V'] = float(inp.dc_self_bias_V)

    warnings: list[str] = []
    if 'coupling_efficiency' not in out:
        warnings.append('coupling_efficiency was not estimated; provide commanded and absorbed power.')
    elif out['coupling_efficiency'] > 1.0:
        warnings.append('coupling_efficiency is greater than 1; check commanded/absorbed power definitions.')
    if voltage is not None and 'effective_impedance_ohm' not in out:
        warnings.append('effective_impedance_ohm was not estimated; provide RMS current or delivered power.')
    if ('bias' in inp.role.lower() or 'lf' in inp.role.lower()) and voltage is not None and 'self_bias_fraction' not in out:
        warnings.append('self_bias_fraction was not estimated; provide measured DC self-bias.')
    if warnings:
        out['warnings'] = warnings
    return out


def recipe_snippet(coefficients: dict[str, Any]) -> dict[str, Any]:
    keys = (
        'role',
        'frequency_Hz',
        'voltage_rms_V',
        'coupling_efficiency',
        'effective_impedance_ohm',
        'self_bias_fraction',
    )
    return {key: coefficients[key] for key in keys if key in coefficients}


def main() -> int:
    parser = argparse.ArgumentParser(description='Estimate rf_envelope calibration coefficients from measured RF quantities.')
    parser.add_argument('--role', required=True, help='Port role, for example hf_source or lf_bias.')
    parser.add_argument('--frequency-Hz', type=float, default=None)
    parser.add_argument('--commanded-power-W', type=float, default=None, help='Generator, net delivered, or recipe command power.')
    parser.add_argument('--forward-power-W', type=float, default=None)
    parser.add_argument('--reflected-power-W', type=float, default=None)
    parser.add_argument('--absorbed-power-W', type=float, default=None)
    parser.add_argument('--voltage-rms-V', type=float, default=None)
    parser.add_argument('--current-rms-A', type=float, default=None)
    parser.add_argument('--delivered-power-W', type=float, default=None)
    parser.add_argument('--dc-self-bias-V', type=float, default=None)
    parser.add_argument('--write', type=Path, default=None, help='Optional YAML output path.')
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
        'estimated_coefficients': estimate,
        'recipe_port_snippet': recipe_snippet(estimate),
    }
    text = yaml.safe_dump(payload, sort_keys=False)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding='utf-8')
        print(args.write.resolve())
    else:
        print(text.strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
