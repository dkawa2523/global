"""Compile static wall topology outside the ODE hot path."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from plasma_global.errors import ModelConfigurationError
from plasma_global.models.walls import (
    CompiledWallBoundary,
    WallBoundary,
    _CompiledBranch,
)


def _readonly(values: object, *, dtype: type) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _compiled_branches(
    boundary: WallBoundary,
    species_ids: tuple[str, ...],
    charge_values: np.ndarray,
    ion_indices: np.ndarray,
) -> tuple[tuple[_CompiledBranch, ...], ...]:
    index = {species_id: i for i, species_id in enumerate(species_ids)}
    branches: dict[int, list[_CompiledBranch]] = {
        int(species_index): [] for species_index in ion_indices
    }
    for reaction in boundary.reactions:
        incident_index = index.get(reaction.incident_species)
        if incident_index is None:
            raise ModelConfigurationError(
                f"Boundary reaction {reaction.reaction_id!r} references unknown "
                f"incident species {reaction.incident_species!r}"
            )
        if charge_values[incident_index] <= 0.0:
            raise ModelConfigurationError(
                f"Boundary incident species {reaction.incident_species!r} must be "
                "a positive ion"
            )
        branches[incident_index].append(
            _compile_branch(
                reaction.reaction_id, reaction.probability, reaction.products, index
            )
        )
    return tuple(tuple(branches[int(species_index)]) for species_index in ion_indices)


def _compile_branch(
    reaction_id: str,
    probability: float,
    products: Mapping[str, float],
    index: dict[str, int],
) -> _CompiledBranch:
    product_indices: list[int] = []
    product_yields: list[float] = []
    for product, yield_per_ion in products.items():
        product_index = index.get(product)
        if product_index is None:
            raise ModelConfigurationError(
                f"Boundary reaction {reaction_id!r} references unknown product "
                f"{product!r}"
            )
        product_indices.append(product_index)
        product_yields.append(yield_per_ion)
    return _CompiledBranch(
        reaction_id=reaction_id,
        probability=probability,
        product_indices=_readonly(product_indices, dtype=int),
        product_yields=_readonly(product_yields, dtype=float),
    )


def _positive_ion_masses(
    species_ids: tuple[str, ...],
    mass_values: np.ndarray,
    ion_indices: np.ndarray,
) -> np.ndarray:
    selected = mass_values[ion_indices]
    if np.any(selected <= 0.0):
        bad = [
            species_ids[int(species_index)]
            for species_index, mass in zip(ion_indices, selected, strict=True)
            if mass <= 0.0
        ]
        raise ModelConfigurationError(
            f"Positive ions must have positive mass_kg: {bad}"
        )
    return selected


def compile_wall_boundary_data(
    *,
    boundary: WallBoundary,
    species_ids: tuple[str, ...],
    charges: np.ndarray,
    masses_kg: np.ndarray,
) -> CompiledWallBoundary:
    """Resolve one wall's species indices and reaction branches."""

    charge_values = np.asarray(charges, dtype=float)
    mass_values = np.asarray(masses_kg, dtype=float)
    expected = (len(species_ids),)
    if charge_values.shape != expected or mass_values.shape != expected:
        raise ModelConfigurationError(
            "Wall species, charge, and mass arrays must use the same order"
        )
    ion_indices = np.flatnonzero(charge_values > 0.0)
    neutral_indices = np.flatnonzero(charge_values == 0.0)
    branches_by_ion = _compiled_branches(
        boundary, species_ids, charge_values, ion_indices
    )
    selected_masses = _positive_ion_masses(species_ids, mass_values, ion_indices)
    return CompiledWallBoundary(
        boundary=boundary,
        ion_indices=_readonly(ion_indices, dtype=int),
        charges=_readonly(charge_values[ion_indices], dtype=float),
        masses_kg=_readonly(selected_masses, dtype=float),
        neutral_indices=_readonly(neutral_indices, dtype=int),
        branches_by_ion=branches_by_ion,
        standard_floating_wall=(
            boundary.transport_kind == "bohm"
            and boundary.sheath_energy_eV == 0.0
            and ion_indices.size > 0
        ),
    )
