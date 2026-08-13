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


def _momentum_support_error(cross_section: CrossSectionData) -> str | None:
    if (
        cross_section.kind == "momentum_transfer"
        and float(cross_section.energy_eV[0]) > 0.0
        and float(cross_section.sigma_m2[0]) != 0.0
    ):
        return (
            f"momentum cross section {cross_section.id!r} must explicitly cover "
            "0 eV; low-energy support is not extrapolated"
        )
    return None


def _onset_support_error(cross_section: CrossSectionData) -> str | None:
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


def cross_section_support_error(cross_section: CrossSectionData) -> str | None:
    """Describe curve data that contradicts declared low-energy support."""

    return _momentum_support_error(cross_section) or _onset_support_error(cross_section)


def _readonly_curve(
    energy_eV: np.ndarray, sigma_m2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    energy_eV.setflags(write=False)
    sigma_m2.setflags(write=False)
    return energy_eV, sigma_m2


def _threshold_node_index(energy_eV: np.ndarray, threshold_eV: float) -> int | None:
    tolerance = max(abs(float(np.spacing(threshold_eV))), np.finfo(float).tiny)
    candidates = np.flatnonzero(np.abs(energy_eV - threshold_eV) <= tolerance)
    if candidates.size == 0:
        return None
    distances = np.abs(energy_eV[candidates] - threshold_eV)
    return int(candidates[int(np.argmin(distances))])


def _normalized_onset_curve(
    cross_section: CrossSectionData,
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.array(cross_section.energy_eV, dtype=float, copy=True)
    sigma = np.array(cross_section.sigma_m2, dtype=float, copy=True)
    threshold = cross_section.threshold_eV
    sigma[energy < threshold] = 0.0
    threshold_index = _threshold_node_index(energy, threshold)
    if threshold_index is None:
        threshold_index = int(np.searchsorted(energy, threshold))
        energy = np.insert(energy, threshold_index, threshold)
        sigma = np.insert(sigma, threshold_index, 0.0)
    else:
        energy[threshold_index] = threshold
        if float(sigma[threshold_index]) != 0.0:
            lower_edge = np.nextafter(threshold, 0.0)
            previous = (
                0.0 if threshold_index == 0 else float(energy[threshold_index - 1])
            )
            if lower_edge > previous:
                energy = np.insert(energy, threshold_index, lower_edge)
                sigma = np.insert(sigma, threshold_index, 0.0)
    if float(energy[0]) > 0.0:
        energy = np.insert(energy, 0, 0.0)
        sigma = np.insert(sigma, 0, 0.0)
    return _readonly_curve(energy, sigma)


def _normalized_continuous_curve(
    cross_section: CrossSectionData,
) -> tuple[np.ndarray, np.ndarray]:
    energy = np.array(cross_section.energy_eV, dtype=float, copy=True)
    sigma = np.array(cross_section.sigma_m2, dtype=float, copy=True)
    first_energy = float(energy[0])
    if first_energy == 0.0:
        return _readonly_curve(energy, sigma)
    return _readonly_curve(
        np.insert(energy, 0, 0.0),
        np.insert(sigma, 0, 0.0),
    )


def normalized_cross_section_curve(
    cross_section: CrossSectionData,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one immutable lower-support policy for all electron closures."""

    support_error = cross_section_support_error(cross_section)
    if support_error is not None:
        raise ValueError(support_error)
    if has_onset_threshold(cross_section):
        return _normalized_onset_curve(cross_section)
    return _normalized_continuous_curve(cross_section)


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
    "normalized_cross_section_curve",
    "surface_reaction_shape_error",
]
