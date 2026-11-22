"""Finite chemistry rate contracts, evaluators, and state dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from plasma_global.chemistry._rate_evaluators import (
    RateEvaluator,
    RateTableLoader,
    _arrhenius_evaluator,
    _constant_evaluator,
    _electron_impact_evaluator,
    _electron_table_evaluator,
    _electron_temperature_power_evaluator,
    _first_order_evaluator,
    _tabulated_evaluator,
)
from plasma_global.chemistry.data import CrossSectionData, RateModelData
from plasma_global.errors import ChemistryError


def _mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ChemistryError(f"{where} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _only_keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ChemistryError(f"{where} contains unknown keys: {', '.join(unknown)}")


def _required_text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChemistryError(f"{where} must be a non-empty string")
    return value


def _finite_number(
    model: Mapping[str, Any], name: str, where: str, *, minimum: float | None = None
) -> float:
    value = model[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChemistryError(f"{where}.{name} must be a number")
    number = float(value)
    if not np.isfinite(number) or (minimum is not None and number < minimum):
        suffix = "finite" if minimum is None else f"finite and >= {minimum:g}"
        raise ChemistryError(f"{where}.{name} must be {suffix}")
    return number


def _validate_coverage(value: object, where: str) -> None:
    if value is None:
        return
    config = _mapping(value, where)
    kind = config.get("kind")
    if kind == "constant":
        _only_keys(config, {"kind"}, where)
        return
    if kind not in {"site_blocking", "species_power"}:
        raise ChemistryError(f"{where}.kind is unsupported: {kind!r}")
    species_key = "site_species" if kind == "site_blocking" else "species"
    _only_keys(config, {"kind", species_key, "exponent"}, where)
    missing = [name for name in (species_key, "exponent") if name not in config]
    if missing:
        raise ChemistryError(f"{where} is missing: {', '.join(missing)}")
    if not isinstance(config[species_key], str) or not config[species_key].strip():
        raise ChemistryError(f"{where}.{species_key} must be a non-empty species ID")
    _finite_number(config, "exponent", where, minimum=0.0)


def _validate_tabulated_rate_parameters(model: Mapping[str, Any], where: str) -> None:
    if model["axis"] not in {
        "gas_temperature_K",
        "mean_energy_eV",
        "electron_temperature_eV",
        "reduced_field_Td",
        "pressure_Pa",
    }:
        raise ChemistryError(f"{where}.axis is unsupported: {model['axis']!r}")
    if model["bounds"] not in {"error", "hold"}:
        raise ChemistryError(f"{where}.bounds must be error or hold")


def _electron_impact_parameters(model: Mapping[str, Any], where: str) -> None:
    _required_text(model["cross_section"], f"{where}.cross_section")
    _finite_number(model, "branching_yield", where, minimum=0.0)


def _electron_table_parameters(model: Mapping[str, Any], where: str) -> None:
    _required_text(model["rate_id"], f"{where}.rate_id")
    if "branching_yield" in model:
        _finite_number(model, "branching_yield", where, minimum=0.0)


def _arrhenius_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "A", where, minimum=0.0)
    _finite_number(model, "beta", where)
    _finite_number(model, "activation_eV", where, minimum=0.0)


def _constant_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "value", where, minimum=0.0)


def _first_order_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "rate_s_inv", where, minimum=0.0)


def _electron_power_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "A", where, minimum=0.0)
    reference = _finite_number(model, "reference_temperature_K", where, minimum=0.0)
    if reference == 0.0:
        raise ChemistryError(f"{where}.reference_temperature_K must be positive")
    _finite_number(model, "exponent", where)


def _sticking_parameters(model: Mapping[str, Any], where: str) -> None:
    value = _finite_number(model, "value", where, minimum=0.0)
    if value > 1.0:
        raise ChemistryError(f"{where}.value must not exceed one")
    _validate_coverage(model.get("coverage"), f"{where}.coverage")


def _ion_assisted_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "yield", where, minimum=0.0)
    threshold = _finite_number(model, "threshold_eV", where, minimum=0.0)
    reference = _finite_number(model, "reference_energy_eV", where, minimum=0.0)
    if reference <= threshold:
        raise ChemistryError(f"{where}.reference_energy_eV must exceed threshold_eV")
    _finite_number(model, "exponent", where, minimum=0.0)
    _validate_coverage(model.get("coverage"), f"{where}.coverage")


def _desorption_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "frequency_s_inv", where, minimum=0.0)
    _finite_number(model, "activation_eV", where, minimum=0.0)


def _surface_reaction_parameters(model: Mapping[str, Any], where: str) -> None:
    _finite_number(model, "A_m2_s", where, minimum=0.0)
    _finite_number(model, "activation_eV", where, minimum=0.0)


@dataclass(frozen=True, slots=True)
class RateDefinition:
    required: frozenset[str]
    validate: Callable[[Mapping[str, Any], str], None]
    optional: frozenset[str] = frozenset()
    family: Literal["gas", "surface"] = "gas"
    dimensional_order: bool = False
    dependencies: Literal["none", "electron", "gas", "table_axis"] = "none"
    evaluator: Callable[[RateModelData], RateEvaluator] | None = None


RATE_DEFINITIONS: Mapping[str, RateDefinition] = {
    "electron_table": RateDefinition(
        frozenset({"rate_id"}),
        _electron_table_parameters,
        frozenset({"branching_yield", "overall_order"}),
        dimensional_order=True,
        dependencies="electron",
        evaluator=_electron_table_evaluator,
    ),
    "electron_impact": RateDefinition(
        frozenset({"cross_section", "branching_yield"}),
        _electron_impact_parameters,
        dependencies="electron",
    ),
    "arrhenius": RateDefinition(
        frozenset({"A", "beta", "activation_eV"}),
        _arrhenius_parameters,
        frozenset({"overall_order"}),
        dimensional_order=True,
        dependencies="gas",
        evaluator=_arrhenius_evaluator,
    ),
    "constant": RateDefinition(
        frozenset({"value"}),
        _constant_parameters,
        frozenset({"overall_order"}),
        dimensional_order=True,
        evaluator=_constant_evaluator,
    ),
    "first_order": RateDefinition(
        frozenset({"rate_s_inv"}),
        _first_order_parameters,
        evaluator=_first_order_evaluator,
    ),
    "experimental.electron_temperature_power_law": RateDefinition(
        frozenset({"A", "reference_temperature_K", "exponent"}),
        _electron_power_parameters,
        frozenset({"overall_order"}),
        dimensional_order=True,
        dependencies="electron",
        evaluator=_electron_temperature_power_evaluator,
    ),
    "tabulated_1d": RateDefinition(
        frozenset({"axis", "file", "bounds"}),
        _validate_tabulated_rate_parameters,
        frozenset({"overall_order"}),
        dimensional_order=True,
        dependencies="table_axis",
    ),
    "sticking": RateDefinition(
        frozenset({"value"}),
        _sticking_parameters,
        frozenset({"coverage"}),
        family="surface",
    ),
    "ion_assisted": RateDefinition(
        frozenset({"yield", "threshold_eV", "reference_energy_eV", "exponent"}),
        _ion_assisted_parameters,
        frozenset({"coverage"}),
        family="surface",
    ),
    "desorption": RateDefinition(
        frozenset({"frequency_s_inv", "activation_eV"}),
        _desorption_parameters,
        family="surface",
    ),
    "langmuir_hinshelwood": RateDefinition(
        frozenset({"A_m2_s", "activation_eV"}),
        _surface_reaction_parameters,
        family="surface",
    ),
}


def rate_definition(kind: str) -> RateDefinition:
    try:
        return RATE_DEFINITIONS[kind]
    except KeyError as exc:
        raise ChemistryError(f"unsupported rate kind {kind!r}") from exc


def validate_rate_model(model: RateModelData) -> RateDefinition:
    definition = rate_definition(model.kind)
    parameters = model.parameters
    where = f"rate model {model.id!r}"
    _only_keys(parameters, set(definition.required | definition.optional), where)
    if missing := sorted(definition.required - parameters.keys()):
        raise ChemistryError(f"{where} is missing: {', '.join(missing)}")
    if "overall_order" in parameters:
        _finite_number(parameters, "overall_order", where, minimum=0.0)
    definition.validate(parameters, where)
    return definition


def rate_dependency(model: RateModelData) -> Literal["none", "electron", "gas"]:
    dependency = rate_definition(model.kind).dependencies
    if dependency != "table_axis":
        return dependency
    return (
        "gas"
        if model.parameters["axis"] in {"pressure_Pa", "gas_temperature_K"}
        else "electron"
    )


def compile_rate_evaluator(
    model: RateModelData,
    cross_sections: Mapping[str, CrossSectionData],
    *,
    table_loader: RateTableLoader,
) -> tuple[RateEvaluator, float | None]:
    """Select only declared gas evaluators after the shared contract check."""
    definition = validate_rate_model(model)
    if definition.family != "gas":
        raise ChemistryError(
            f"rate model {model.id} kind {model.kind!r} is not a gas-rate model"
        )
    if model.kind == "electron_impact":
        return _electron_impact_evaluator(model, cross_sections)
    if model.kind == "tabulated_1d":
        return _tabulated_evaluator(model, table_loader), None
    if definition.evaluator is None:
        raise ChemistryError(f"rate model {model.id} has no gas evaluator")
    return definition.evaluator(model), None
