"""Compile canonical chemistry into immutable solver arrays.

This module owns the public compiled representation and the final array
assembly.  Scientific validation and scalar rate construction live in focused
private modules, keeping the main path readable from ``ChemistryData`` to
``CompiledChemistry``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from plasma_global.chemistry._compile_validation import (
    validate_gas_rate_dimensions,
    validate_non_gas_reactions,
    validate_species,
    validate_unique_ids,
    validated_gas_rate_models,
)
from plasma_global.chemistry._rate_evaluators import E_CHARGE as E_CHARGE
from plasma_global.chemistry._rate_evaluators import (
    ELECTRON_MASS_KG as ELECTRON_MASS_KG,
)
from plasma_global.chemistry._rate_evaluators import EV_TO_K as EV_TO_K
from plasma_global.chemistry._rate_evaluators import (
    compile_rate_evaluator,
    load_rate_table,
)
from plasma_global.chemistry._rate_evaluators import (
    maxwell_rate_table as _build_maxwell_rate_table,
)
from plasma_global.chemistry.data import (
    ChemistryData,
    CrossSectionData,
    RateModelData,
    ReactionData,
    SpeciesData,
)
from plasma_global.errors import ChemistryError


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


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _readonly_bool(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    result.setflags(write=False)
    return result


def _legacy_energy_loss_view(chemistry: CompiledChemistry) -> np.ndarray:
    """Expose the historic nonnegative loss array without mutable storage."""

    return _readonly(np.maximum(-chemistry.electron_energy_transfer_eV, 0.0))


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
    electron_energy_transfer_eV: np.ndarray
    gas_heating_eV: np.ndarray
    reaction_zones: tuple[tuple[str, ...], ...]
    jacobian_species_pattern: np.ndarray
    element_names: tuple[str, ...]
    element_matrix: np.ndarray
    boundary_reactions: tuple[CompiledBoundaryReaction, ...]
    surface_reactions: tuple[ReactionData, ...]
    cross_sections: Mapping[str, CrossSectionData]
    provenance: Mapping[str, Any]

    energy_loss_eV = property(_legacy_energy_loss_view)


@dataclass(frozen=True, slots=True)
class _CompiledGasReactions:
    stoichiometry: np.ndarray
    reactant_orders: np.ndarray
    electron_orders: np.ndarray
    rate_evaluators: tuple[RateEvaluator, ...]
    electron_energy_transfer_eV: np.ndarray
    gas_heating_eV: np.ndarray


def maxwell_rate_table(
    cross_section: CrossSectionData,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute Maxwellian ``<sigma v>`` on a tail-safe mean-energy range."""

    return _build_maxwell_rate_table(cross_section)


def _rate_evaluator(
    model: RateModelData, cross_sections: Mapping[str, CrossSectionData]
) -> tuple[RateEvaluator, float | None]:
    evaluator, transfer = compile_rate_evaluator(
        model,
        cross_sections,
        table_loader=load_rate_table,
    )
    return evaluator, transfer


def _boundary_incident_species(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> str:
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
    return incident[0]


def _validate_boundary_products(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> None:
    if any(species[species_id].phase != "gas" for species_id in reaction.products):
        raise ChemistryError(
            f"boundary reaction {reaction.id} products must be gas species"
        )
    if any(species[species_id].charge != 0 for species_id in reaction.products):
        raise ChemistryError(
            f"boundary reaction {reaction.id} products must be neutral"
        )


def _compile_boundary(
    reaction: ReactionData, species: Mapping[str, SpeciesData]
) -> CompiledBoundaryReaction:
    incident_species = _boundary_incident_species(reaction, species)
    _validate_boundary_products(reaction, species)
    before_charge = float(species[incident_species].charge)
    after_charge = sum(
        order * species[species_id].charge
        for species_id, order in reaction.products.items()
    )
    return CompiledBoundaryReaction(
        id=reaction.id,
        incident_species=incident_species,
        products=MappingProxyType(dict(reaction.products)),
        probability=1.0,
        zones=reaction.zones,
        surfaces=reaction.surfaces,
        wall_charge_per_event=before_charge - after_charge,
    )


def _explicit_reaction_electron_transfer(reaction: ReactionData) -> float | None:
    legacy_loss = reaction.energy_loss_eV
    signed_transfer = reaction.electron_energy_transfer_eV
    if legacy_loss is not None and signed_transfer is not None:
        raise ChemistryError(
            f"reaction {reaction.id!r} must define at most one of "
            "energy_loss_eV or electron_energy_transfer_eV"
        )
    if signed_transfer is not None:
        if not math.isfinite(signed_transfer):
            raise ChemistryError(
                f"reaction {reaction.id!r} electron_energy_transfer_eV must be finite"
            )
        return signed_transfer
    if legacy_loss is not None:
        if not math.isfinite(legacy_loss) or legacy_loss < 0.0:
            raise ChemistryError(
                f"reaction {reaction.id!r} energy_loss_eV must be finite and "
                "nonnegative"
            )
        return -legacy_loss
    return None


def _reaction_electron_energy_transfer(
    reaction: ReactionData,
    default_transfer_eV: float | None,
) -> float:
    """Resolve positive electron gain while retaining legacy positive loss."""

    explicit_transfer = _explicit_reaction_electron_transfer(reaction)
    transfer = explicit_transfer
    if transfer is None:
        transfer = default_transfer_eV or 0.0
    _validate_electron_transfer_topology(reaction, transfer)
    return transfer


def _validate_electron_transfer_topology(
    reaction: ReactionData, transfer_eV: float
) -> None:
    has_electron = any(
        values.get("e", 0.0) > 0.0 for values in (reaction.reactants, reaction.products)
    )
    if transfer_eV != 0.0 and not has_electron:
        raise ChemistryError(
            f"reaction {reaction.id!r} defines electron-energy transfer but does "
            "not contain an electron reactant or product"
        )


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
    electron_energy_transfers = np.zeros(reaction_count)
    gas_heating = np.zeros(reaction_count)
    validated_rate_models = validated_gas_rate_models(data, species_by_id)
    validate_gas_rate_dimensions(data.gas_reactions, validated_rate_models)

    for row, (reaction, rate_model) in enumerate(
        zip(data.gas_reactions, validated_rate_models, strict=True)
    ):
        compiled_rate = evaluator_by_model_id.get(rate_model.id)
        if compiled_rate is None:
            compiled_rate = _rate_evaluator(rate_model, data.cross_sections)
            evaluator_by_model_id[rate_model.id] = compiled_rate
        evaluator, default_transfer = compiled_rate
        evaluators.append(evaluator)
        electron_energy_transfers[row] = _reaction_electron_energy_transfer(
            reaction,
            default_transfer,
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
        electron_energy_transfer_eV=electron_energy_transfers,
        gas_heating_eV=gas_heating,
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
            elif axis in {
                "mean_energy_eV",
                "electron_temperature_eV",
                "reduced_field_Td",
            }:
                dependencies[row, charged] = True
    changed = stoichiometry != 0.0
    return changed.T @ dependencies


def compile_chemistry(data: ChemistryData) -> CompiledChemistry:
    """Validate conservation and compile all gas reactions once."""

    validate_unique_ids([item.id for item in data.species], "species")
    all_reactions = [
        *data.gas_reactions,
        *data.boundary_reactions,
        *data.surface_reactions,
    ]
    validate_unique_ids([item.id for item in all_reactions], "reaction")
    species_by_id, gas_species = validate_species(data)
    gas = _compile_gas_reactions(data, species_by_id, gas_species)
    validate_non_gas_reactions(data, species_by_id)
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
        electron_energy_transfer_eV=_readonly(gas.electron_energy_transfer_eV),
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


__all__ = [
    "CompiledBoundaryReaction",
    "CompiledChemistry",
    "RateContextLike",
    "RateEvaluator",
    "compile_chemistry",
    "maxwell_rate_table",
]
