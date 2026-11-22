from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from plasma_global.input._schema_power import (
    ExperimentalRFEnvelopeCommand,
    ExperimentalRFEnvelopeModel,
)


def _model_role(role: str) -> str:
    if role not in {"source", "bias"}:
        raise ValueError("role must be source or bias")
    return role


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

    def __post_init__(self) -> None:
        _model_role(self.role)
        for name in self.__dataclass_fields__:
            if name == "role" or (value := getattr(self, name)) is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            if name != "dc_self_bias_V" and number < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            setattr(self, name, number)
        if self.frequency_Hz == 0.0:
            raise ValueError("frequency_Hz must be positive")


def _forward_net_power(inp: RFEnvelopeCalibrationInput) -> float | None:
    if inp.forward_power_W is None:
        if inp.reflected_power_W is not None:
            raise ValueError("reflected_power_W requires forward_power_W")
        return None
    net = inp.forward_power_W - (inp.reflected_power_W or 0.0)
    if net < 0.0:
        raise ValueError("reflected_power_W must not exceed forward_power_W")
    return net


def _net_delivered_power(inp: RFEnvelopeCalibrationInput) -> float | None:
    """Resolve one source-side real power; V*I is not a real-power substitute."""
    candidates = [
        value
        for value in (
            inp.delivered_power_W,
            inp.commanded_power_W,
            _forward_net_power(inp),
        )
        if value is not None
    ]
    if not candidates:
        return None
    if any(
        not math.isclose(value, candidates[0], rel_tol=1e-9, abs_tol=0.0)
        for value in candidates[1:]
    ):
        raise ValueError(
            "delivered, commanded and forward-minus-reflected powers "
            "must describe the same net real power"
        )
    return candidates[0]


def _power_coefficients(inp: RFEnvelopeCalibrationInput) -> dict[str, float]:
    coefficients = {}
    for name in (
        "frequency_Hz",
        "commanded_power_W",
        "forward_power_W",
        "reflected_power_W",
        "absorbed_power_W",
    ):
        value = getattr(inp, name)
        if value is not None:
            coefficients[name] = value
    delivered = _net_delivered_power(inp)
    if delivered is None:
        return coefficients
    coefficients["delivered_power_W"] = delivered
    absorbed = inp.absorbed_power_W
    if absorbed is not None:
        if absorbed > delivered:
            raise ValueError(
                "absorbed_power_W must not exceed net delivered real power"
            )
        if delivered > 0.0:
            coefficients["coupling_efficiency"] = absorbed / delivered
    return coefficients


def _self_bias_coefficients(inp: RFEnvelopeCalibrationInput) -> dict[str, float]:
    bias = inp.dc_self_bias_V
    if bias is None:
        return {}
    if _model_role(inp.role) == "source" and bias != 0.0:
        raise ValueError(
            "the source-role model defines zero self-bias; "
            "use role=bias for nonzero self-bias"
        )
    if bias > 0.0:
        raise ValueError(
            "the RF envelope bias model cannot reproduce positive DC self-bias"
        )
    voltage = inp.voltage_rms_V
    if voltage is None:
        raise ValueError("dc_self_bias_V requires voltage_rms_V")
    if voltage == 0.0 and bias != 0.0:
        raise ValueError("zero RMS voltage cannot have nonzero DC self-bias")
    result = {"dc_self_bias_V": bias}
    if voltage > 0.0:
        result["self_bias_fraction"] = -bias / (math.sqrt(2.0) * voltage)
    return result


def _current_diagnostics(
    voltage: float | None, current: float | None, delivered: float | None
) -> dict[str, float]:
    if current is None:
        return {}
    if voltage is None:
        raise ValueError("current_rms_A requires voltage_rms_V")
    apparent = voltage * current
    coefficients = {"current_rms_A": current, "apparent_power_VA": apparent}
    if current > 0.0:
        coefficients["impedance_magnitude_ohm"] = voltage / current
    if delivered is not None:
        if delivered > apparent and not math.isclose(
            delivered, apparent, rel_tol=1e-9, abs_tol=0.0
        ):
            raise ValueError(
                "delivered real power must not exceed RMS V*I apparent power "
                "at the same measurement plane"
            )
        if apparent > 0.0:
            coefficients["power_factor"] = delivered / apparent
    return coefficients


def _voltage_coefficients(
    inp: RFEnvelopeCalibrationInput, delivered: float | None
) -> dict[str, float]:
    coefficients = _self_bias_coefficients(inp)
    voltage = inp.voltage_rms_V
    coefficients.update(_current_diagnostics(voltage, inp.current_rms_A, delivered))
    if voltage is None:
        return coefficients
    coefficients["voltage_rms_V"] = voltage
    if voltage == 0.0 and (inp.absorbed_power_W or 0.0) > 0.0:
        raise ValueError("positive absorbed power requires positive RMS voltage")
    if delivered is not None and delivered > 0.0:
        if voltage == 0.0:
            raise ValueError("positive delivered power requires positive RMS voltage")
        coefficients["effective_impedance_ohm"] = voltage * voltage / delivered
    return coefficients


def _calibration_warnings(
    inp: RFEnvelopeCalibrationInput, coefficients: dict[str, Any]
) -> list[str]:
    warnings = []
    if "coupling_efficiency" not in coefficients:
        warnings.append(
            "coupling_efficiency was not identified; provide positive net delivered "
            "real power and absorbed power. "
            "A zero/off point does not identify efficiency."
        )
    if inp.voltage_rms_V is not None and "effective_impedance_ohm" not in coefficients:
        warnings.append(
            "effective_impedance_ohm was not identified; positive RMS voltage and "
            "positive net delivered real power are required. V/I is an impedance "
            "magnitude, not this model parameter."
        )
    if inp.current_rms_A is not None:
        warnings.append(
            "RMS current and phase are not fitted by rf_envelope; V*I and V/I are "
            "reported only as apparent power and impedance magnitude."
        )
    if (
        _model_role(inp.role) == "bias"
        and inp.voltage_rms_V is not None
        and "self_bias_fraction" not in coefficients
    ):
        warnings.append(
            "self_bias_fraction was not identified; "
            "provide DC self-bias at positive RMS voltage."
        )
    return warnings


def estimate_rf_envelope_coefficients(
    inp: RFEnvelopeCalibrationInput,
) -> dict[str, Any]:
    """Identify eta=P_abs/P_del and R_eff=V_rms^2/P_del at one measured point.

    All source powers must refer to the same net real-power measurement plane.
    commanded_power_W is usable only when that generator setpoint equals this
    net delivered power; it is not the public absorbed-power recipe command.
    """
    coefficients: dict[str, Any] = {"role": _model_role(inp.role)}
    coefficients.update(_power_coefficients(inp))
    coefficients.update(
        _voltage_coefficients(inp, coefficients.get("delivered_power_W"))
    )
    for key, value in coefficients.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} is not finite for the supplied calibration point")
    warnings = _calibration_warnings(inp, coefficients)
    if warnings:
        coefficients["warnings"] = warnings
    return coefficients


def canonical_fragments(coefficients: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return public port/command fragments that reproduce the fitted point."""
    shared_keys = (
        "frequency_Hz",
        "coupling_efficiency",
        "effective_impedance_ohm",
        "self_bias_fraction",
    )
    shared = {key: coefficients[key] for key in shared_keys if key in coefficients}
    model_data = {
        "kind": "experimental.rf_envelope",
        "role": _model_role(str(coefficients["role"])),
        **shared,
    }
    command_data: dict[str, Any] = {"kind": "experimental.rf_envelope"}
    voltage = coefficients.get("voltage_rms_V")
    absorbed = coefficients.get("absorbed_power_W")
    if voltage is not None:
        if (
            voltage > 0.0
            and not {"effective_impedance_ohm", "coupling_efficiency"}
            <= coefficients.keys()
        ):
            raise ValueError(
                "a calibrated voltage fragment requires net delivered and absorbed "
                "power; RMS current alone cannot identify real-power coupling"
            )
        model_data.update(control="voltage", voltage_rms_V=voltage)
        command_data["voltage_rms_V"] = voltage
    elif absorbed is not None:
        if absorbed == 0.0 and coefficients.get("delivered_power_W", 0.0) > 0.0:
            raise ValueError(
                "an absorbed-power off command cannot reproduce positive delivered "
                "power; provide positive RMS voltage for voltage control"
            )
        model_data.update(control="absorbed_power", absorbed_power_W=absorbed)
        command_data["absorbed_power_W"] = absorbed
    else:
        raise ValueError(
            "provide absorbed power or a fully identified voltage calibration point"
        )
    model = ExperimentalRFEnvelopeModel.model_validate(model_data)
    command = ExperimentalRFEnvelopeCommand.model_validate(command_data)
    return {
        "power_model": model.model_dump(
            mode="json", exclude_none=True, exclude_unset=True
        ),
        "power_command": command.model_dump(
            mode="json", exclude={"kind"}, exclude_none=True, exclude_unset=True
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Identify one-point rf_envelope coefficients; this does not validate "
            "apparatus predictions away from that point."
        )
    )
    parser.add_argument("--role", required=True, choices=("source", "bias"))
    parser.add_argument("--frequency-Hz", type=float, default=None)
    parser.add_argument(
        "--commanded-power-W",
        type=float,
        default=None,
        help=(
            "Net delivered real source power only when the generator setpoint "
            "equals that measurement; not an absorbed-power recipe command."
        ),
    )
    parser.add_argument(
        "--forward-power-W",
        type=float,
        default=None,
        help="Incident real power; subtract reflected power (assumed zero if omitted).",
    )
    parser.add_argument("--reflected-power-W", type=float, default=None)
    parser.add_argument("--absorbed-power-W", type=float, default=None)
    parser.add_argument("--voltage-rms-V", type=float, default=None)
    parser.add_argument("--current-rms-A", type=float, default=None)
    parser.add_argument(
        "--delivered-power-W",
        type=float,
        default=None,
        help="Net real power delivered at the RMS-voltage measurement plane, in W.",
    )
    parser.add_argument("--dc-self-bias-V", type=float, default=None)
    parser.add_argument("--write", type=Path, default=None)
    args = parser.parse_args()
    try:
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
        payload = {"estimated_coefficients": estimate, **canonical_fragments(estimate)}
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
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
