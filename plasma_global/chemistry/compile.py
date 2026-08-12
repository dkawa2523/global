"""Compile canonical chemistry into immutable arrays used by the solver."""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from plasma_global.chemistry.data import (
    ChemistryData,
    CrossSectionData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.errors import ChemistryError, ModelDomainError

E_CHARGE = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
EV_TO_K = E_CHARGE / 1.380649e-23

# Cross sections are never extrapolated past their final energy node.  Stop
# each Maxwellian lookup where the unresolved energy-weighted distribution
# tail is negligible instead of silently treating a truncated curve as zero.
_MAXWELL_UNRESOLVED_TAIL_FRACTION = 1.0e-6
_MAXWELL_MIN_MEAN_ENERGY_EV = 1.0e-3
_MAXWELL_TABLE_POINTS = 512


def _maxwell_tail_cutoff(tolerance: float) -> float:
    """Return ``x`` where ``integral_x^inf u exp(-u) du`` reaches tolerance."""

    lower, upper = 0.0, 64.0
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        tail = (1.0 + middle) * math.exp(-middle)
        if tail > tolerance:
            lower = middle
        else:
            upper = middle
    return upper


_MAXWELL_TAIL_CUTOFF = _maxwell_tail_cutoff(_MAXWELL_UNRESOLVED_TAIL_FRACTION)


class RateContextLike(Protocol):
    @property
    def mean_energy_eV(self) -> float: ...

    @property
    def electron_temperature_eV(self) -> float: ...

    @property
    def reduced_field_Td(self) -> float | None: ...

    @property
    def gas_temperature_K(self) -> float: ...


RateEvaluator = Callable[[RateContextLike], float]


@dataclass(frozen=True, slots=True)
class CompiledBoundaryReaction:
    id: str
    incident_species: str
    products: Mapping[str, float]
    probability: float
    zones: tuple[str, ...]
    surfaces: tuple[str, ...]
    wall_charge_per_event: float


@dataclass(frozen=True, slots=True)
class CompiledChemistry:
    species_ids: tuple[str, ...]
    charges: np.ndarray
    masses_kg: np.ndarray
    reaction_ids: tuple[str, ...]
    stoichiometry: np.ndarray
    reactant_orders: np.ndarray
    electron_orders: np.ndarray
    rate_evaluators: tuple[RateEvaluator, ...]
    energy_loss_eV: np.ndarray
    gas_heating_eV: np.ndarray
    reaction_zones: tuple[tuple[str, ...], ...]
    jacobian_species_pattern: np.ndarray
    element_names: tuple[str, ...]
    element_matrix: np.ndarray
    boundary_reactions: tuple[CompiledBoundaryReaction, ...]
    surface_reactions: tuple[ReactionData, ...]
    cross_sections: Mapping[str, CrossSectionData]
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CompiledGasReactions:
    stoichiometry: np.ndarray
    reactant_orders: np.ndarray
    electron_orders: np.ndarray
    rate_evaluators: tuple[RateEvaluator, ...]
    energy_loss_eV: np.ndarray
    gas_heating_eV: np.ndarray


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _readonly_bool(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    result.setflags(write=False)
    return result


def _unique(values: list[str], kind: str) -> None:
    seen: set[str] = set()
    duplicate = sorted({value for value in values if value in seen or seen.add(value)})
    if duplicate:
        raise ChemistryError(f"duplicate {kind} ids: {', '.join(duplicate)}")


def _normalized_residual(delta: float, scale: float) -> float:
    return abs(delta) / max(abs(scale), 1.0)


def _validate_reaction_balance(
    reaction: ReactionData,
    species: Mapping[str, SpeciesData],
    *,
    boundary: bool,
) -> None:
    unknown = sorted((set(reaction.reactants) | set(reaction.products)) - set(species))
    if unknown:
        raise ChemistryError(
            f"reaction {reaction.id} references unknown species: {', '.join(unknown)}"
        )

    elements = sorted(
        {
            element
            for species_id in (*reaction.reactants, *reaction.products)
            for element in species[species_id].elements
        }
    )
    for element in elements:
        before = sum(
            order * species[species_id].elements.get(element, 0.0)
            for species_id, order in reaction.reactants.items()
        )
        after = sum(
            order * species[species_id].elements.get(element, 0.0)
            for species_id, order in reaction.products.items()
        )
        if _normalized_residual(after - before, before) > 1.0e-12:
            raise ChemistryError(f"reaction {reaction.id} does not conserve {element}")

    before_charge = sum(
        order * species[species_id].charge
        for species_id, order in reaction.reactants.items()
    )
    after_charge = sum(
        order * species[species_id].charge
        for species_id, order in reaction.products.items()
    )
    if (
        not boundary
        and _normalized_residual(after_charge - before_charge, before_charge) > 1.0e-12
    ):
        raise ChemistryError(f"reaction {reaction.id} does not conserve charge")


def _table(path: Any) -> tuple[np.ndarray, np.ndarray]:
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
    values = np.asarray(rows, dtype=float)
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


def maxwell_rate_table(
    cross_section: CrossSectionData,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute Maxwellian ``<sigma v>`` on a tail-safe mean-energy range."""

    maximum_cross_section_energy = float(cross_section.energy_eV[-1])
    maximum_mean_energy = 1.5 * maximum_cross_section_energy / _MAXWELL_TAIL_CUTOFF
    if maximum_mean_energy <= _MAXWELL_MIN_MEAN_ENERGY_EV:
        raise ChemistryError(
            f"cross section {cross_section.id!r} ends at "
            f"{maximum_cross_section_energy:g} eV and cannot support the minimum "
            f"Maxwellian mean energy {_MAXWELL_MIN_MEAN_ENERGY_EV:g} eV with "
            f"unresolved tail <= {_MAXWELL_UNRESOLVED_TAIL_FRACTION:g}"
        )
    mean_energy = np.geomspace(
        _MAXWELL_MIN_MEAN_ENERGY_EV,
        maximum_mean_energy,
        _MAXWELL_TABLE_POINTS,
    )
    temperature = 2.0 * mean_energy / 3.0
    energy = cross_section.energy_eV
    sigma = cross_section.sigma_m2
    prefactor = 2.0 / np.sqrt(np.pi) * np.sqrt(2.0 * E_CHARGE / ELECTRON_MASS_KG)
    rates = np.empty_like(mean_energy)
    for index, te_eV in enumerate(temperature):
        integrand = sigma * energy * np.exp(-energy / te_eV)
        rates[index] = prefactor * np.trapezoid(integrand, energy) / te_eV**1.5
    mean_energy.setflags(write=False)
    rates.setflags(write=False)
    return mean_energy, rates


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
    axis, values = maxwell_rate_table(cross_section)
    branch = float(model.parameters["branching_yield"])
    if not np.isfinite(branch) or branch < 0.0:
        raise ChemistryError(f"rate model {model.id} has invalid branching_yield")

    def electron_impact(context: RateContextLike) -> float:
        supplied = getattr(context, "rate_coefficients", None)
        if supplied is not None and cross_section_id in supplied:
            return branch * float(supplied[cross_section_id])
        return branch * _bounded_interp(
            axis,
            values,
            context.mean_energy_eV,
            name="mean_energy_eV (Maxwellian tail-safe range)",
        )

    return electron_impact, cross_section.energy_loss_eV


def _arrhenius_evaluator(model: RateModelData) -> RateEvaluator:
    amplitude = float(model.parameters["A"])
    exponent = float(model.parameters["beta"])
    activation = float(model.parameters["activation_eV"])

    def arrhenius(context: RateContextLike) -> float:
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

    def electron_power(context: RateContextLike) -> float:
        temperature_K = max(context.electron_temperature_eV * EV_TO_K, 1.0e-12)
        return float(amplitude * (temperature_K / reference) ** exponent)

    return electron_power


def _tabulated_evaluator(model: RateModelData) -> RateEvaluator:
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
    axis, values = _table(model.parameters["file"])
    bounds = str(model.parameters["bounds"])
    if bounds not in {"error", "clip"}:
        raise ChemistryError(f"rate model {model.id} bounds must be error or clip")

    def tabulated(context: RateContextLike) -> float:
        return _bounded_interp(
            axis,
            values,
            _context_value(context, axis_name),
            name=axis_name,
            bounds=bounds,
        )

    return tabulated


def _rate_evaluator(
    model: RateModelData, cross_sections: Mapping[str, CrossSectionData]
) -> tuple[RateEvaluator, float | None]:
    if model.kind == "electron_impact":
        return _electron_impact_evaluator(model, cross_sections)
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
        return _tabulated_evaluator(model), None

    # Surface-only evaluators are compiled by the surface model and must not
    # accidentally appear in the gas reaction matrix.
    raise ChemistryError(
        f"rate model {model.id} kind {model.kind!r} is not a gas-rate model"
    )


def _compile_boundary(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> CompiledBoundaryReaction:
    incident = [
        species_id
        for species_id, order in reaction.reactants.items()
        if species[species_id].phase == "gas"
        and species[species_id].charge > 0
        and order > 0.0
    ]
    if (
        len(incident) != 1
        or len(reaction.reactants) != 1
        or reaction.reactants[incident[0]] != 1.0
    ):
        raise ChemistryError(
            f"boundary reaction {reaction.id} must have one unit positive-ion reactant"
        )
    if any(species[species_id].phase != "gas" for species_id in reaction.products):
        raise ChemistryError(
            f"boundary reaction {reaction.id} products must be gas species"
        )
    if any(species[species_id].charge != 0 for species_id in reaction.products):
        raise ChemistryError(
            f"boundary reaction {reaction.id} products must be neutral"
        )
    before_charge = float(species[incident[0]].charge)
    after_charge = sum(
        order * species[species_id].charge
        for species_id, order in reaction.products.items()
    )
    return CompiledBoundaryReaction(
        id=reaction.id,
        incident_species=incident[0],
        products=MappingProxyType(dict(reaction.products)),
        probability=1.0,
        zones=reaction.zones,
        surfaces=reaction.surfaces,
        wall_charge_per_event=before_charge - after_charge,
    )


def _validate_cross_section_targets(
    cross_sections: Mapping[str, CrossSectionData],
    species_by_id: Mapping[str, SpeciesData],
) -> None:
    for cross_section in cross_sections.values():
        target = species_by_id.get(cross_section.target)
        if target is None or target.phase != "gas" or target.id == "e":
            raise ChemistryError(
                f"cross section {cross_section.id} target {cross_section.target!r} "
                "must be a declared heavy gas species"
            )


def _validate_species(
    data: ChemistryData,
) -> tuple[dict[str, SpeciesData], tuple[SpeciesData, ...]]:
    species_by_id = {item.id: item for item in data.species}
    electron = species_by_id.get("e")
    if electron is None or electron.phase != "gas" or electron.charge != -1:
        raise ChemistryError(
            "canonical chemistry must contain exactly one gas electron species 'e' "
            "with charge -1"
        )
    _validate_cross_section_targets(data.cross_sections, species_by_id)
    gas_species = tuple(
        item for item in data.species if item.phase == "gas" and item.id != "e"
    )
    return species_by_id, gas_species


def _validate_electron_impact_reaction(
    reaction: ReactionData,
    rate_model: RateModelData,
    cross_sections: Mapping[str, CrossSectionData],
) -> None:
    cross_section = cross_sections.get(str(rate_model.parameters["cross_section"]))
    if cross_section is None:
        raise ChemistryError(
            f"reaction {reaction.id} references an unknown cross section"
        )
    if cross_section.kind == "momentum_transfer":
        raise ChemistryError(
            f"reaction {reaction.id} cannot use momentum-transfer cross section "
            f"{cross_section.id!r} as a reactive rate"
        )
    if reaction.reactants.get("e", 0.0) <= 0.0:
        raise ChemistryError(
            f"electron-impact reaction {reaction.id} must consume an electron"
        )
    if reaction.reactants.get(cross_section.target, 0.0) <= 0.0:
        raise ChemistryError(
            f"electron-impact reaction {reaction.id} must consume cross-section "
            f"target {cross_section.target!r}"
        )


def _validated_gas_rate_model(
    data: ChemistryData,
    reaction: ReactionData,
    species_by_id: Mapping[str, SpeciesData],
) -> RateModelData:
    _validate_reaction_balance(reaction, species_by_id, boundary=False)
    if any(
        species_by_id[item].phase != "gas"
        for item in (*reaction.reactants, *reaction.products)
    ):
        raise ChemistryError(f"gas reaction {reaction.id} contains a surface species")
    if reaction.rate_model not in data.rate_models:
        raise ChemistryError(
            f"reaction {reaction.id} references unknown rate model "
            f"{reaction.rate_model!r}"
        )
    rate_model = data.rate_models[reaction.rate_model]
    if rate_model.kind == "electron_impact":
        _validate_electron_impact_reaction(reaction, rate_model, data.cross_sections)
    return rate_model


def _compile_gas_reactions(
    data: ChemistryData,
    species_by_id: Mapping[str, SpeciesData],
    gas_species: tuple[SpeciesData, ...],
) -> _CompiledGasReactions:
    gas_ids = tuple(item.id for item in gas_species)
    gas_index = {species_id: index for index, species_id in enumerate(gas_ids)}
    reaction_count, species_count = len(data.gas_reactions), len(gas_species)
    stoichiometry = np.zeros((reaction_count, species_count))
    reactant_orders = np.zeros_like(stoichiometry)
    electron_orders = np.zeros(reaction_count)
    evaluators: list[RateEvaluator] = []
    evaluator_by_model_id: dict[str, tuple[RateEvaluator, float | None]] = {}
    energy_losses = np.zeros(reaction_count)
    gas_heating = np.zeros(reaction_count)

    for row, reaction in enumerate(data.gas_reactions):
        rate_model = _validated_gas_rate_model(data, reaction, species_by_id)
        compiled_rate = evaluator_by_model_id.get(rate_model.id)
        if compiled_rate is None:
            compiled_rate = _rate_evaluator(rate_model, data.cross_sections)
            evaluator_by_model_id[rate_model.id] = compiled_rate
        evaluator, default_loss = compiled_rate
        evaluators.append(evaluator)
        energy_losses[row] = (
            reaction.energy_loss_eV
            if reaction.energy_loss_eV is not None
            else default_loss or 0.0
        )
        gas_heating[row] = reaction.gas_heating_eV
        electron_orders[row] = reaction.reactants.get("e", 0.0)
        for species_id, index in gas_index.items():
            reactant_orders[row, index] = reaction.reactants.get(species_id, 0.0)
            stoichiometry[row, index] = reaction.products.get(
                species_id, 0.0
            ) - reaction.reactants.get(species_id, 0.0)

    return _CompiledGasReactions(
        stoichiometry=stoichiometry,
        reactant_orders=reactant_orders,
        electron_orders=electron_orders,
        rate_evaluators=tuple(evaluators),
        energy_loss_eV=energy_losses,
        gas_heating_eV=gas_heating,
    )


def _validate_non_gas_reactions(
    data: ChemistryData, species_by_id: Mapping[str, SpeciesData]
) -> None:
    for reaction in data.boundary_reactions:
        _validate_reaction_balance(reaction, species_by_id, boundary=True)
    for reaction in data.surface_reactions:
        _validate_reaction_balance(reaction, species_by_id, boundary=False)
        if reaction.rate_model not in data.rate_models:
            raise ChemistryError(
                f"surface reaction {reaction.id} references unknown rate model "
                f"{reaction.rate_model!r}"
            )


def _compile_element_matrix(
    gas_species: tuple[SpeciesData, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    element_names = tuple(
        sorted(
            {
                element
                for item in gas_species
                for element in item.elements
                if element != "site"
            }
        )
    )
    element_matrix = np.array(
        [
            [item.elements.get(element, 0.0) for item in gas_species]
            for element in element_names
        ],
        dtype=float,
    )
    return element_names, element_matrix


def compile_chemistry(data: ChemistryData) -> CompiledChemistry:
    """Validate conservation and compile all gas reactions once."""

    _unique([item.id for item in data.species], "species")
    all_reactions = [
        *data.gas_reactions,
        *data.boundary_reactions,
        *data.surface_reactions,
    ]
    _unique([item.id for item in all_reactions], "reaction")
    species_by_id, gas_species = _validate_species(data)
    gas = _compile_gas_reactions(data, species_by_id, gas_species)
    _validate_non_gas_reactions(data, species_by_id)
    element_names, element_matrix = _compile_element_matrix(gas_species)
    jacobian_species_pattern = _compile_jacobian_species_pattern(
        data,
        gas_species,
        gas.stoichiometry,
        gas.reactant_orders,
        gas.electron_orders,
    )

    return CompiledChemistry(
        species_ids=tuple(item.id for item in gas_species),
        charges=_readonly(np.array([item.charge for item in gas_species])),
        masses_kg=_readonly(np.array([item.mass_kg for item in gas_species])),
        reaction_ids=tuple(item.id for item in data.gas_reactions),
        stoichiometry=_readonly(gas.stoichiometry),
        reactant_orders=_readonly(gas.reactant_orders),
        electron_orders=_readonly(gas.electron_orders),
        rate_evaluators=gas.rate_evaluators,
        energy_loss_eV=_readonly(gas.energy_loss_eV),
        gas_heating_eV=_readonly(gas.gas_heating_eV),
        reaction_zones=tuple(item.zones for item in data.gas_reactions),
        jacobian_species_pattern=_readonly_bool(jacobian_species_pattern),
        element_names=element_names,
        element_matrix=_readonly(element_matrix),
        boundary_reactions=tuple(
            _compile_boundary(item, species_by_id) for item in data.boundary_reactions
        ),
        surface_reactions=data.surface_reactions,
        cross_sections=data.cross_sections,
        provenance=data.provenance,
    )


def _compile_jacobian_species_pattern(
    data: ChemistryData,
    gas_species: tuple[SpeciesData, ...],
    stoichiometry: np.ndarray,
    reactant_orders: np.ndarray,
    electron_orders: np.ndarray,
) -> np.ndarray:
    """Compile a conservative species dependency graph for finite differences."""

    dependencies = reactant_orders != 0.0
    charged = np.asarray([item.charge != 0 for item in gas_species], dtype=bool)
    for row, reaction in enumerate(data.gas_reactions):
        model = data.rate_models[str(reaction.rate_model)]
        if electron_orders[row] != 0.0 or model.kind in {
            "electron_impact",
            "experimental.electron_temperature_power_law",
        }:
            dependencies[row, charged] = True
        if model.kind == "arrhenius":
            dependencies[row, :] = True
        if model.kind == "tabulated_1d":
            axis = str(model.parameters["axis"])
            if axis in {"pressure_Pa", "gas_temperature_K"}:
                dependencies[row, :] = True
            elif axis in {"mean_energy_eV", "electron_temperature_eV"}:
                dependencies[row, charged] = True
    changed = stoichiometry != 0.0
    return (changed.T.astype(np.int8) @ dependencies.astype(np.int8)) > 0


__all__ = [
    "CompiledBoundaryReaction",
    "CompiledChemistry",
    "RateContextLike",
    "RateEvaluator",
    "compile_chemistry",
    "maxwell_rate_table",
]
