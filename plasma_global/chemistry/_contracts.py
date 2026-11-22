"""Pure reaction-shape contracts shared by chemistry compilation and models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Literal

import numpy as np

from plasma_global._identifiers import validate_identifier
from plasma_global.errors import ChemistryError

if TYPE_CHECKING:
    from plasma_global.chemistry.data import (
        CrossSectionData,
        RateModelData,
        ReactionData,
        SpeciesData,
    )


_ONSET_THRESHOLD_KINDS = frozenset({"dissociation", "excitation", "ionization"})
_CROSS_SECTION_KINDS = frozenset(
    {
        "momentum_transfer",
        "attachment",
        "dissociation",
        "deexcitation",
        "excitation",
        "ionization",
    }
)


def _validate_electron_energy_transfer(
    family: Literal["gas", "boundary", "surface"],
    where: str,
    signed_transfer_eV: float | None,
) -> None:
    if signed_transfer_eV is None:
        return
    if not math.isfinite(signed_transfer_eV):
        raise ChemistryError(f"{where}: electron_energy_transfer_eV must be finite")
    if family != "gas" and signed_transfer_eV != 0.0:
        raise ChemistryError(
            f"{where}: {family} reactions do not support electron-energy fields"
        )


def validate_reaction_energy_fields(
    family: Literal["gas", "boundary", "surface"],
    where: str,
    signed_transfer_eV: float | None,
    gas_heating_eV: float,
) -> None:
    _validate_electron_energy_transfer(family, where, signed_transfer_eV)
    if not math.isfinite(gas_heating_eV):
        raise ChemistryError(f"{where}: gas_heating_eV must be finite")
    if family == "gas" and gas_heating_eV < 0.0:
        raise ChemistryError(f"{where}: gas_heating_eV must be nonnegative")
    if family == "boundary" and gas_heating_eV != 0.0:
        raise ChemistryError(
            f"{where}: boundary reactions do not support gas_heating_eV"
        )


def _finite_nonnegative_scalar(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return bool(np.isfinite(number) and number >= 0.0)


def _cross_section_label(where: str | None, cross_section_id: object) -> str:
    return f"cross section {cross_section_id!r}" if where is None else where


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _numeric_curve(
    energy_eV: object, sigma_m2: object
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        raw_energy = np.asarray(energy_eV)
        raw_sigma = np.asarray(sigma_m2)
    except (TypeError, ValueError, OverflowError):
        return None
    raw_arrays = (raw_energy, raw_sigma)
    if any(values.dtype.kind in {"b", "c", "S", "U", "V"} for values in raw_arrays):
        return None
    if any(
        values.dtype.kind == "O"
        and any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in values.flat
        )
        for values in raw_arrays
    ):
        return None
    try:
        return (
            np.asarray(raw_energy, dtype=float),
            np.asarray(raw_sigma, dtype=float),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def _curve_data_error(label: str, energy: np.ndarray, sigma: np.ndarray) -> str | None:
    if not all((energy.ndim == 1, sigma.shape == energy.shape, energy.size >= 2)):
        return (
            f"{label} energy_eV and sigma_m2 must be equal one-dimensional "
            "arrays with at least two values"
        )
    if not all((np.all(np.isfinite(energy)), np.all(np.isfinite(sigma)))):
        return f"{label} contains non-finite values"
    if any((np.any(energy < 0.0), np.any(np.diff(energy) <= 0.0))):
        return f"{label} energy must be nonnegative, strictly increasing, and unique"
    if np.any(sigma < 0.0):
        return f"{label} contains negative cross sections"
    return None


def cross_section_data_error(
    *,
    cross_section_id: object,
    kind: object,
    target: object,
    threshold_eV: object,
    energy_eV: object,
    sigma_m2: object,
    where: str | None = None,
) -> str | None:
    """Describe malformed canonical cross-section data without mutating it."""

    label = _cross_section_label(where, cross_section_id)
    curve = _numeric_curve(energy_eV, sigma_m2)
    if curve is None:
        return f"{label} energy_eV and sigma_m2 must be numeric arrays"
    energy, sigma = curve
    curve_error = _curve_data_error(label, energy, sigma)
    if curve_error is not None:
        return curve_error
    if not _nonempty_string(cross_section_id):
        return f"{label} id must be a non-empty string"
    if not isinstance(kind, str) or kind not in _CROSS_SECTION_KINDS:
        return f"{label} has unsupported kind {kind!r}"
    if not _nonempty_string(target):
        return f"{label} target must be a non-empty string"
    if not _finite_nonnegative_scalar(threshold_eV):
        return f"{label} has invalid threshold"
    return None


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

    data_error = cross_section_data_error(
        cross_section_id=cross_section.id,
        kind=cross_section.kind,
        target=cross_section.target,
        threshold_eV=cross_section.threshold_eV,
        energy_eV=cross_section.energy_eV,
        sigma_m2=cross_section.sigma_m2,
    )
    if data_error is not None:
        raise ValueError(data_error)
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
    if not valid:
        return (
            f"{rate_model.kind} surface reaction {reaction.id} must have exactly one "
            f"unit {expected}"
        )
    ion = gas_reactants[0][0]
    if rate_model.kind == "ion_assisted" and reaction.products.get(ion.id) != 1.0:
        return (
            f"ion_assisted surface reaction {reaction.id} must preserve the "
            f"incident ion {ion.id!r} with unit stoichiometry on both sides; "
            "ion-assisted rates use wall flux as a catalyst, not ion capture"
        )
    return None


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
    "cross_section_data_error",
    "cross_section_support_error",
    "has_onset_threshold",
    "normalized_cross_section_curve",
    "surface_reaction_shape_error",
]


def chemistry_identifier(value: object, where: str) -> str:
    """Use the series component contract at each canonical chemistry boundary."""
    if not isinstance(value, str):
        raise ChemistryError(f"{where}: identifier must be a string")
    try:
        return validate_identifier(value)
    except ValueError as exc:
        raise ChemistryError(f"{where}: {exc}") from exc


def validate_rate_identifiers(parameters: Mapping[str, object], where: str) -> None:
    for name in ("cross_section", "rate_id"):
        if name in parameters:
            chemistry_identifier(parameters[name], f"{where}.{name}")
    coverage = parameters.get("coverage")
    if isinstance(coverage, Mapping):
        for name in ("species", "site_species"):
            if name in coverage:
                chemistry_identifier(coverage[name], f"{where}.coverage.{name}")


def _validate_species_identity(species: SpeciesData) -> None:
    label = f"species {species.id!r}"
    chemistry_identifier(species.id, f"{label}.id")
    for surface in species.surfaces:
        chemistry_identifier(surface, f"{label}.surfaces")
    if species.phase not in {"gas", "surface"}:
        raise ChemistryError(f"{label}.phase must be gas or surface")
    if isinstance(species.charge, bool) or not isinstance(species.charge, int):
        raise ChemistryError(f"{label}.charge must be an integer")
    if any(
        not isinstance(value, str) or not value.strip() for value in species.state_tags
    ):
        raise ChemistryError(f"{label} state tags must be non-empty strings")


def _validate_species_thermodynamics(species: SpeciesData) -> None:
    label = f"species {species.id!r}"
    if not _finite_nonnegative_scalar(species.mass_amu) or (
        species.phase == "gas" and species.mass_amu == 0.0
    ):
        raise ChemistryError(
            f"{label}.mass_amu must be finite and positive for gas, "
            "nonnegative for surface"
        )
    if species.cv_over_kb is not None and (
        not _finite_nonnegative_scalar(species.cv_over_kb) or species.cv_over_kb == 0.0
    ):
        raise ChemistryError(f"{label}.cv_over_kb must be finite and positive")


def _validate_species_composition(species: SpeciesData) -> None:
    label = f"species {species.id!r}"
    if not isinstance(species.elements, Mapping):
        raise ChemistryError(f"{label}.elements must map element names to counts")
    for name, count in species.elements.items():
        if not isinstance(name, str) or not name.strip():
            raise ChemistryError(f"{label}.elements requires non-empty element names")
        if not _finite_nonnegative_scalar(count) or count == 0.0:
            raise ChemistryError(f"{label}.elements.{name} must be finite and positive")


def validate_species_record(species: SpeciesData) -> None:
    """Apply the same species contract to CSV, YAML, and Python construction."""
    _validate_species_identity(species)
    _validate_species_thermodynamics(species)
    _validate_species_composition(species)
