"""Numerical gas-rate evaluators compiled from canonical chemistry data."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from plasma_global.chemistry._maxwell import compile_maxwell_rate
from plasma_global.chemistry.data import CrossSectionData, RateModelData
from plasma_global.errors import ChemistryError, ModelDomainError

E_CHARGE = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
EV_TO_K = E_CHARGE / 1.380649e-23


RateEvaluator = Callable[[Any], float]
RateTableLoader = Callable[[Any], tuple[np.ndarray, np.ndarray]]


def _rate_table_rows(path: Any) -> list[tuple[float, float]]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if header != ["x", "value"]:
                raise ChemistryError(
                    f"rate table {path} must use the exact header x,value"
                )
            rows: list[tuple[float, float]] = []
            for line, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ChemistryError(
                        f"rate table {path}:{line} must contain exactly two fields"
                    )
                rows.append((float(row[0]), float(row[1])))
    except (OSError, ValueError) as exc:
        raise ChemistryError(f"cannot read rate table {path}: {exc}") from exc
    return rows


def load_rate_table(path: Any) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact two-column table used by ``tabulated_1d`` rates."""

    values = np.asarray(_rate_table_rows(path), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ChemistryError(
            f"rate table {path} must have exactly two columns and at least two rows"
        )
    axis, rates = np.asarray(values[:, 0], float), np.asarray(values[:, 1], float)
    if (
        not np.all(np.isfinite(values))
        or np.any(np.diff(axis) <= 0.0)
        or np.any(rates < 0.0)
    ):
        raise ChemistryError(
            f"rate table {path} must be finite, nonnegative, strictly increasing, "
            "and unique"
        )
    return axis, rates


def _context_value(context: object, name: str) -> float:
    if name == "pressure_Pa":
        value = getattr(context, "pressure_Pa", None)
    elif name == "electron_temperature_eV":
        value = getattr(context, "electron_temperature_eV", None)
    else:
        value = getattr(context, name, None)
    if value is None:
        raise ModelDomainError(f"rate evaluation requires {name}")
    return float(value)


def _bounded_interp(
    axis: np.ndarray, values: np.ndarray, x: float, *, name: str, bounds: str = "error"
) -> float:
    if x < axis[0] or x > axis[-1]:
        if bounds == "clip":
            x = float(np.clip(x, axis[0], axis[-1]))
        else:
            raise ModelDomainError(
                f"{name}={x:g} is outside [{axis[0]:g}, {axis[-1]:g}]"
            )
    return float(np.interp(x, axis, values))


def _electron_impact_evaluator(
    model: RateModelData, cross_sections: Mapping[str, CrossSectionData]
) -> tuple[RateEvaluator, float | None]:
    cross_section_id = str(model.parameters["cross_section"])
    if cross_section_id not in cross_sections:
        raise ChemistryError(
            f"rate model {model.id} references unknown cross section "
            f"{cross_section_id!r}"
        )
    cross_section = cross_sections[cross_section_id]
    rate = compile_maxwell_rate(cross_section)
    branch = float(model.parameters["branching_yield"])
    if not np.isfinite(branch) or branch < 0.0:
        raise ChemistryError(f"rate model {model.id} has invalid branching_yield")

    def electron_impact(context: Any) -> float:
        supplied = getattr(context, "rate_coefficients", None)
        if supplied is not None and cross_section_id in supplied:
            return branch * float(supplied[cross_section_id])
        return branch * rate(context.mean_energy_eV)

    return electron_impact, cross_section.resolved_electron_energy_transfer_eV


def _arrhenius_evaluator(model: RateModelData) -> RateEvaluator:
    amplitude = float(model.parameters["A"])
    exponent = float(model.parameters["beta"])
    activation = float(model.parameters["activation_eV"])

    def arrhenius(context: Any) -> float:
        temperature = max(context.gas_temperature_K, 1.0e-12)
        thermal_eV = temperature / EV_TO_K
        return float(
            amplitude
            * (temperature / 300.0) ** exponent
            * np.exp(-activation / thermal_eV)
        )

    return arrhenius


def _electron_temperature_power_evaluator(model: RateModelData) -> RateEvaluator:
    amplitude = float(model.parameters["A"])
    reference = float(model.parameters["reference_temperature_K"])
    exponent = float(model.parameters["exponent"])

    def electron_power(context: Any) -> float:
        temperature_K = max(context.electron_temperature_eV * EV_TO_K, 1.0e-12)
        return float(amplitude * (temperature_K / reference) ** exponent)

    return electron_power


def _tabulated_evaluator(
    model: RateModelData, table_loader: RateTableLoader
) -> RateEvaluator:
    axis_name = str(model.parameters["axis"])
    if axis_name not in {
        "gas_temperature_K",
        "mean_energy_eV",
        "electron_temperature_eV",
        "reduced_field_Td",
        "pressure_Pa",
    }:
        raise ChemistryError(
            f"rate model {model.id} has unsupported table axis {axis_name!r}"
        )
    axis, values = table_loader(model.parameters["file"])
    bounds = str(model.parameters["bounds"])
    if bounds not in {"error", "clip"}:
        raise ChemistryError(f"rate model {model.id} bounds must be error or clip")

    def tabulated(context: Any) -> float:
        return _bounded_interp(
            axis,
            values,
            _context_value(context, axis_name),
            name=axis_name,
            bounds=bounds,
        )

    return tabulated


def compile_rate_evaluator(
    model: RateModelData,
    cross_sections: Mapping[str, CrossSectionData],
    *,
    table_loader: RateTableLoader,
) -> tuple[RateEvaluator, float | None]:
    """Compile one supported gas-rate model into a scalar evaluator."""

    if model.kind == "electron_impact":
        return _electron_impact_evaluator(model, cross_sections)
    if model.kind == "electron_table":
        rate_id = str(model.parameters["rate_id"])
        branch = float(model.parameters.get("branching_yield", 1.0))

        def supplied_rate(context: Any) -> float:
            supplied = getattr(context, "rate_coefficients", {})
            if rate_id not in supplied:
                raise ModelDomainError(f"electron table requires rate {rate_id!r}")
            return branch * float(supplied[rate_id])

        return supplied_rate, None
    if model.kind == "arrhenius":
        return _arrhenius_evaluator(model), None
    if model.kind == "constant":
        value = float(model.parameters["value"])
        return lambda _context: value, None
    if model.kind == "first_order":
        value = float(model.parameters["rate_s_inv"])
        return lambda _context: value, None
    if model.kind == "experimental.electron_temperature_power_law":
        return _electron_temperature_power_evaluator(model), None
    if model.kind == "tabulated_1d":
        return _tabulated_evaluator(model, table_loader), None
    raise ChemistryError(
        f"rate model {model.id} kind {model.kind!r} is not a gas-rate model"
    )
