"""Pure reaction-shape contracts shared by chemistry compilation and models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plasma_global.chemistry.data import (
        CrossSectionData,
        RateModelData,
        ReactionData,
        SpeciesData,
    )


_ONSET_THRESHOLD_KINDS = frozenset({"dissociation", "excitation", "ionization"})


def has_onset_threshold(cross_section: CrossSectionData) -> bool:
    """Return whether threshold denotes zero support below a reaction onset."""

    return (
        cross_section.threshold_eV > 0.0
        and cross_section.kind in _ONSET_THRESHOLD_KINDS
    )


def cross_section_support_error(cross_section: CrossSectionData) -> str | None:
    """Describe curve data that contradicts a declared physical onset."""

    if not has_onset_threshold(cross_section):
        return None
    threshold = cross_section.threshold_eV
    tolerance = max(abs(float(np.spacing(threshold))), np.finfo(float).tiny)
    nodes = tuple(zip(cross_section.energy_eV, cross_section.sigma_m2, strict=True))
    if any(
        float(energy) < threshold - tolerance and float(sigma) != 0.0
        for energy, sigma in nodes
    ):
        return (
            f"cross section {cross_section.id!r} is nonzero below threshold "
            f"{threshold:g} eV"
        )
    if float(cross_section.energy_eV[-1]) < threshold - tolerance:
        return (
            f"cross section {cross_section.id!r} must contain an explicit node at "
            f"threshold {threshold:g} eV or above"
        )
    return None


def _reactants_in_phase(
    reaction: ReactionData,
    species_by_id: Mapping[str, SpeciesData],
    phase: str,
) -> tuple[tuple[SpeciesData, float], ...]:
    return tuple(
        (species_by_id[species_id], order)
        for species_id, order in reaction.reactants.items()
        if species_by_id[species_id].phase == phase
    )


def _thermal_surface_shape_error(
    reaction: ReactionData,
    rate_model: RateModelData,
    species_by_id: Mapping[str, SpeciesData],
) -> str | None:
    surface_reactants = _reactants_in_phase(reaction, species_by_id, "surface")
    adsorbates = tuple(
        (species, order)
        for species, order in surface_reactants
        if "site" not in species.state_tags
    )
    expected_order = 1.0 if rate_model.kind == "desorption" else 2.0
    actual_order = sum(order for _species, order in adsorbates)
    contains_only_adsorbates = len(adsorbates) == len(reaction.reactants) and len(
        surface_reactants
    ) == len(reaction.reactants)
    if contains_only_adsorbates and adsorbates and actual_order == expected_order:
        return None
    return (
        f"{rate_model.kind} surface reaction {reaction.id} must contain only "
        f"adsorbed surface reactants with total order {expected_order:g}"
    )


def _flux_surface_shape_error(
    reaction: ReactionData,
    rate_model: RateModelData,
    species_by_id: Mapping[str, SpeciesData],
) -> str | None:
    gas_reactants = _reactants_in_phase(reaction, species_by_id, "gas")
    if rate_model.kind == "sticking":
        valid = (
            len(gas_reactants) == 1
            and gas_reactants[0][1] == 1.0
            and gas_reactants[0][0].charge == 0
        )
        expected = "neutral gas reactant"
    else:
        valid = (
            len(gas_reactants) == 1
            and gas_reactants[0][1] == 1.0
            and gas_reactants[0][0].charge > 0
        )
        expected = "positive-ion gas reactant"
    if valid:
        return None
    return (
        f"{rate_model.kind} surface reaction {reaction.id} must have exactly one "
        f"unit {expected}"
    )


def surface_reaction_shape_error(
    reaction: ReactionData,
    rate_model: RateModelData,
    species_by_id: Mapping[str, SpeciesData],
) -> str | None:
    """Describe an unsupported surface-rate reactant shape, if present."""

    if rate_model.kind in {"desorption", "langmuir_hinshelwood"}:
        return _thermal_surface_shape_error(reaction, rate_model, species_by_id)
    if rate_model.kind in {"sticking", "ion_assisted"}:
        return _flux_surface_shape_error(reaction, rate_model, species_by_id)
    return None


__all__ = [
    "cross_section_support_error",
    "has_onset_threshold",
    "surface_reaction_shape_error",
]
