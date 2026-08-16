"""Compile the chemistry boundary into immutable arrays used by the RHS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np

from plasma_global.errors import ModelConfigurationError
from plasma_global.models.rates import ConstantRate, RateEvaluator

ReactantPowers = tuple[tuple[int, float], ...]


class ChemistrySource(Protocol):
    """Structural input accepted by :func:`compile_chemistry_data`."""

    @property
    def species_ids(self) -> Sequence[object]: ...

    @property
    def charges(self) -> object: ...

    @property
    def masses_kg(self) -> object: ...

    @property
    def reaction_ids(self) -> Sequence[object]: ...

    @property
    def stoichiometry(self) -> object: ...

    @property
    def reactant_orders(self) -> object: ...

    @property
    def electron_orders(self) -> object: ...

    @property
    def rate_evaluators(self) -> Sequence[object]: ...

    @property
    def energy_loss_eV(self) -> object: ...

    @property
    def gas_heating_eV(self) -> object: ...

    @property
    def reaction_zones(self) -> Sequence[Sequence[object]]: ...

    @property
    def jacobian_species_pattern(self) -> object: ...


@dataclass(frozen=True, slots=True)
class CompiledChemistryData:
    """Validated, immutable chemistry arrays in the runtime's fixed order."""

    species_ids: tuple[str, ...]
    species_index: Mapping[str, int]
    reaction_ids: tuple[str, ...]
    charges: np.ndarray
    masses_kg: np.ndarray
    stoichiometry: np.ndarray
    reactant_orders: np.ndarray
    reactant_powers_by_reaction: tuple[ReactantPowers, ...]
    electron_orders: np.ndarray
    electron_energy_transfer_eV: np.ndarray
    energy_loss_eV: np.ndarray
    gas_heating_eV: np.ndarray
    jacobian_species_pattern: np.ndarray
    reaction_zones: tuple[tuple[str, ...], ...]
    rate_evaluators: tuple[RateEvaluator, ...]


@dataclass(frozen=True, slots=True)
class _ChemistryArrays:
    charges: np.ndarray
    masses_kg: np.ndarray
    stoichiometry: np.ndarray
    reactant_orders: np.ndarray
    electron_orders: np.ndarray
    electron_energy_transfer_eV: np.ndarray
    gas_heating_eV: np.ndarray
    jacobian_pattern: np.ndarray


def _readonly_float_array(
    values: object, *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if array.shape != shape:
        raise ModelConfigurationError(
            f"{name} has shape {array.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ModelConfigurationError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_jacobian_pattern(values: object, species_count: int) -> np.ndarray:
    pattern = np.asarray(values, dtype=bool)
    expected_shape = (species_count, species_count)
    if pattern.shape != expected_shape:
        raise ModelConfigurationError(
            "jacobian_species_pattern has shape "
            f"{pattern.shape}, expected {expected_shape}"
        )
    result = np.array(pattern, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _electron_energy_transfer(
    source: ChemistrySource, reaction_count: int
) -> np.ndarray:
    signed = getattr(source, "electron_energy_transfer_eV", None)
    if signed is not None:
        return _readonly_float_array(
            signed,
            shape=(reaction_count,),
            name="electron_energy_transfer_eV",
        )
    legacy_loss = _readonly_float_array(
        source.energy_loss_eV,
        shape=(reaction_count,),
        name="energy_loss_eV",
    )
    if np.any(legacy_loss < 0.0):
        raise ModelConfigurationError("Electron energy losses must be non-negative")
    result = -legacy_loss
    result.setflags(write=False)
    return result


def _legacy_energy_loss(electron_energy_transfer_eV: np.ndarray) -> np.ndarray:
    """Build the immutable positive-loss compatibility view once."""

    result = np.maximum(-electron_energy_transfer_eV, 0.0)
    result.setflags(write=False)
    return result


def _compiled_rate_evaluators(
    values: Sequence[object], reaction_count: int
) -> tuple[RateEvaluator, ...]:
    if len(values) != reaction_count:
        raise ModelConfigurationError("rate_evaluators length must match reaction_ids")
    evaluators: list[RateEvaluator] = []
    for evaluator in values:
        if isinstance(evaluator, (float, int)):
            evaluators.append(ConstantRate(float(evaluator)))
        elif callable(evaluator):
            evaluators.append(cast(RateEvaluator, evaluator))
        else:
            raise ModelConfigurationError(
                "Every reaction rate evaluator must be callable or numeric"
            )
    return tuple(evaluators)


def _compiled_identifiers(
    source: ChemistrySource,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    species_ids = tuple(str(value) for value in source.species_ids)
    reaction_ids = tuple(str(value) for value in source.reaction_ids)
    if not species_ids or len(set(species_ids)) != len(species_ids):
        raise ModelConfigurationError(
            "Chemistry needs unique, non-empty heavy-species IDs"
        )
    if len(set(reaction_ids)) != len(reaction_ids):
        raise ModelConfigurationError("Reaction IDs must be unique")
    return species_ids, reaction_ids


def _compiled_arrays(
    source: ChemistrySource, *, species_count: int, reaction_count: int
) -> _ChemistryArrays:
    arrays = _ChemistryArrays(
        charges=_readonly_float_array(
            source.charges, shape=(species_count,), name="charges"
        ),
        masses_kg=_readonly_float_array(
            source.masses_kg, shape=(species_count,), name="masses_kg"
        ),
        stoichiometry=_readonly_float_array(
            source.stoichiometry,
            shape=(reaction_count, species_count),
            name="stoichiometry",
        ),
        reactant_orders=_readonly_float_array(
            source.reactant_orders,
            shape=(reaction_count, species_count),
            name="reactant_orders",
        ),
        electron_orders=_readonly_float_array(
            source.electron_orders,
            shape=(reaction_count,),
            name="electron_orders",
        ),
        electron_energy_transfer_eV=_electron_energy_transfer(source, reaction_count),
        gas_heating_eV=_readonly_float_array(
            source.gas_heating_eV,
            shape=(reaction_count,),
            name="gas_heating_eV",
        ),
        jacobian_pattern=_readonly_jacobian_pattern(
            source.jacobian_species_pattern, species_count
        ),
    )
    _validate_physical_arrays(arrays)
    return arrays


def _validate_physical_arrays(arrays: _ChemistryArrays) -> None:
    if np.any(arrays.masses_kg <= 0.0):
        raise ModelConfigurationError(
            "Every evolved species must have positive mass_kg"
        )
    if np.any(arrays.reactant_orders < 0.0) or np.any(arrays.electron_orders < 0.0):
        raise ModelConfigurationError(
            "Mass-action reaction orders must be non-negative"
        )
    if np.any(arrays.gas_heating_eV < 0.0):
        raise ModelConfigurationError("Gas reaction heating must be non-negative")


def _compiled_reaction_zones(
    source: ChemistrySource, reaction_count: int
) -> tuple[tuple[str, ...], ...]:
    raw_reaction_zones = source.reaction_zones
    if len(raw_reaction_zones) != reaction_count:
        raise ModelConfigurationError("reaction_zones length must match reaction_ids")
    return tuple(
        tuple(str(zone_id) for zone_id in zones) for zones in raw_reaction_zones
    )


def _reactant_powers_by_reaction(
    reactant_orders: np.ndarray,
) -> tuple[ReactantPowers, ...]:
    """Compile only nonzero heavy-reactant powers in species order."""

    return tuple(
        tuple(
            (species_index, float(order))
            for species_index, order in enumerate(reaction_orders)
            if order != 0.0
        )
        for reaction_orders in reactant_orders
    )


def compile_chemistry_data(source: ChemistrySource) -> CompiledChemistryData:
    """Validate and copy one chemistry source without mutating the runtime model."""

    species_ids, reaction_ids = _compiled_identifiers(source)
    species_count = len(species_ids)
    reaction_count = len(reaction_ids)
    arrays = _compiled_arrays(
        source,
        species_count=species_count,
        reaction_count=reaction_count,
    )

    electron_transfer = arrays.electron_energy_transfer_eV
    return CompiledChemistryData(
        species_ids=species_ids,
        species_index=MappingProxyType(
            {species_id: index for index, species_id in enumerate(species_ids)}
        ),
        reaction_ids=reaction_ids,
        charges=arrays.charges,
        masses_kg=arrays.masses_kg,
        stoichiometry=arrays.stoichiometry,
        reactant_orders=arrays.reactant_orders,
        reactant_powers_by_reaction=_reactant_powers_by_reaction(
            arrays.reactant_orders
        ),
        electron_orders=arrays.electron_orders,
        electron_energy_transfer_eV=electron_transfer,
        energy_loss_eV=_legacy_energy_loss(electron_transfer),
        gas_heating_eV=arrays.gas_heating_eV,
        jacobian_species_pattern=arrays.jacobian_pattern,
        reaction_zones=_compiled_reaction_zones(source, reaction_count),
        rate_evaluators=_compiled_rate_evaluators(
            tuple(source.rate_evaluators), reaction_count
        ),
    )


__all__ = ["CompiledChemistryData", "compile_chemistry_data"]
