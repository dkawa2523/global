"""Single-source ion wall flux and boundary-reaction bookkeeping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from plasma_global.core.exceptions import ModelConfigurationError
from plasma_global.models.electrons import ELEMENTARY_CHARGE_C, ElectronState

ELECTRON_MASS_KG = 9.1093837139e-31


@dataclass(frozen=True)
class BoundaryReaction:
    """One product branch fed by an incident positive-ion wall flux."""

    reaction_id: str
    incident_species: str
    products: Mapping[str, float] = field(default_factory=dict)
    probability: float = 1.0

    def __post_init__(self) -> None:
        if not self.reaction_id or not self.incident_species:
            raise ModelConfigurationError(
                "Boundary reaction and incident species IDs must not be empty"
            )
        products = {str(key): float(value) for key, value in self.products.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in products.values()):
            raise ModelConfigurationError(
                "Boundary product yields must be finite and non-negative"
            )
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ModelConfigurationError(
                "Boundary reaction probability must be between zero and one"
            )
        object.__setattr__(self, "products", MappingProxyType(products))


def _validate_prescribed_transport(
    *,
    area_m2: float,
    prescribed_frequency_s_inv: float | None,
) -> None:
    if (
        prescribed_frequency_s_inv is None
        or not math.isfinite(prescribed_frequency_s_inv)
        or prescribed_frequency_s_inv < 0.0
    ):
        raise ModelConfigurationError(
            "prescribed_frequency wall transport requires a finite nonnegative "
            "frequency"
        )
    if area_m2 <= 0.0:
        raise ModelConfigurationError(
            "prescribed_frequency wall transport requires positive area_m2"
        )


def _validate_ambipolar_transport(
    *,
    area_m2: float,
    diffusion_coefficient_m2_s: float | None,
    diffusion_length_m: float | None,
) -> None:
    for name, value in (
        ("diffusion_coefficient_m2_s", diffusion_coefficient_m2_s),
        ("diffusion_length_m", diffusion_length_m),
    ):
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise ModelConfigurationError(
                f"ambipolar wall transport requires positive {name}"
            )
    if area_m2 <= 0.0:
        raise ModelConfigurationError(
            "ambipolar wall transport requires positive area_m2"
        )


def _validate_wall_transport(
    *,
    kind: str,
    area_m2: float,
    prescribed_frequency_s_inv: float | None,
    diffusion_coefficient_m2_s: float | None,
    diffusion_length_m: float | None,
) -> None:
    if kind not in {"bohm", "prescribed_frequency", "ambipolar", "off"}:
        raise ModelConfigurationError(f"Unknown wall transport kind {kind!r}")
    if kind == "prescribed_frequency":
        _validate_prescribed_transport(
            area_m2=area_m2,
            prescribed_frequency_s_inv=prescribed_frequency_s_inv,
        )
    if kind == "ambipolar":
        _validate_ambipolar_transport(
            area_m2=area_m2,
            diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
            diffusion_length_m=diffusion_length_m,
        )


def _validate_branch_probabilities(reactions: tuple[BoundaryReaction, ...]) -> None:
    totals: dict[str, float] = {}
    for reaction in reactions:
        totals[reaction.incident_species] = (
            totals.get(reaction.incident_species, 0.0) + reaction.probability
        )
    overfull = {
        species: total for species, total in totals.items() if total > 1.0 + 1.0e-12
    }
    if overfull:
        raise ModelConfigurationError(
            f"Boundary branch probabilities exceed one: {overfull}"
        )


@dataclass(frozen=True)
class WallBoundary:
    """One surface's positive-ion transport and boundary reaction branches."""

    zone_id: str
    area_m2: float
    bohm_factor: float = 0.61
    sheath_energy_eV: float = 0.0
    reactions: tuple[BoundaryReaction, ...] = ()
    surface_id: str | None = None
    transport_kind: str = "bohm"
    prescribed_frequency_s_inv: float | None = None
    diffusion_coefficient_m2_s: float | None = None
    diffusion_length_m: float | None = None

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ModelConfigurationError("Wall boundary zone_id must not be empty")
        if not math.isfinite(self.area_m2) or self.area_m2 < 0.0:
            raise ModelConfigurationError(
                "Wall boundary area_m2 must be finite and non-negative"
            )
        if not math.isfinite(self.bohm_factor) or self.bohm_factor < 0.0:
            raise ModelConfigurationError(
                "Wall boundary bohm_factor must be finite and non-negative"
            )
        if not math.isfinite(self.sheath_energy_eV) or self.sheath_energy_eV < 0.0:
            raise ModelConfigurationError(
                "Wall boundary sheath_energy_eV must be finite and non-negative"
            )
        if self.surface_id is not None and not self.surface_id:
            raise ModelConfigurationError("Wall boundary surface_id must not be empty")
        _validate_wall_transport(
            kind=self.transport_kind,
            area_m2=self.area_m2,
            prescribed_frequency_s_inv=self.prescribed_frequency_s_inv,
            diffusion_coefficient_m2_s=self.diffusion_coefficient_m2_s,
            diffusion_length_m=self.diffusion_length_m,
        )
        _validate_branch_probabilities(self.reactions)


@dataclass(frozen=True)
class WallFluxRecord:
    zone_id: str
    surface_id: str | None
    transport_kind: str
    incident_species: str
    incident_flux_m2_s: float
    incident_rate_m3_s: float
    bohm_speed_m_s: float
    electron_energy_loss_J_m3_s: float
    branch_rates_m3_s: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "branch_rates_m3_s", MappingProxyType(dict(self.branch_rates_m3_s))
        )


@dataclass(frozen=True)
class WallEvaluation:
    species_derivative_m3_s: np.ndarray
    electron_energy_loss_J_m3_s: float
    records: tuple[WallFluxRecord, ...]
    incident_flux_m2_s: np.ndarray
    sheath_energy_eV: float


@dataclass(frozen=True, slots=True)
class _CompiledBranch:
    reaction_id: str
    probability: float
    product_indices: np.ndarray
    product_yields: np.ndarray


@dataclass(frozen=True, slots=True)
class CompiledWallBoundary:
    """A wall boundary with every species and product ID resolved once."""

    boundary: WallBoundary
    ion_indices: np.ndarray
    charges: np.ndarray
    masses_kg: np.ndarray
    branches_by_ion: tuple[tuple[_CompiledBranch, ...], ...]
    standard_floating_wall: bool


def _readonly(values: object, *, dtype: type) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def compile_wall_boundary(
    *,
    boundary: WallBoundary,
    species_ids: tuple[str, ...],
    charges: np.ndarray,
    masses_kg: np.ndarray,
) -> CompiledWallBoundary:
    """Resolve one wall's static species topology outside the ODE hot path."""

    index = {species_id: i for i, species_id in enumerate(species_ids)}
    charge_values = np.asarray(charges, dtype=float)
    mass_values = np.asarray(masses_kg, dtype=float)
    expected = (len(species_ids),)
    if charge_values.shape != expected or mass_values.shape != expected:
        raise ModelConfigurationError(
            "Wall species, charge, and mass arrays must use the same order"
        )
    ion_indices = np.flatnonzero(charge_values > 0.0)
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
                f"Boundary incident species {reaction.incident_species!r} must be a positive ion"
            )
        product_indices: list[int] = []
        product_yields: list[float] = []
        for product, yield_per_ion in reaction.products.items():
            product_index = index.get(product)
            if product_index is None:
                raise ModelConfigurationError(
                    f"Boundary reaction {reaction.reaction_id!r} references "
                    f"unknown product {product!r}"
                )
            product_indices.append(product_index)
            product_yields.append(yield_per_ion)
        branches[incident_index].append(
            _CompiledBranch(
                reaction_id=reaction.reaction_id,
                probability=reaction.probability,
                product_indices=_readonly(product_indices, dtype=int),
                product_yields=_readonly(product_yields, dtype=float),
            )
        )
    selected_masses = mass_values[ion_indices]
    if np.any(selected_masses <= 0.0):
        bad = [
            species_ids[int(species_index)]
            for species_index, mass in zip(ion_indices, selected_masses)
            if mass <= 0.0
        ]
        raise ModelConfigurationError(
            f"Positive ions must have positive mass_kg: {bad}"
        )
    return CompiledWallBoundary(
        boundary=boundary,
        ion_indices=_readonly(ion_indices, dtype=int),
        charges=_readonly(charge_values[ion_indices], dtype=float),
        masses_kg=_readonly(selected_masses, dtype=float),
        branches_by_ion=tuple(
            tuple(branches[int(species_index)]) for species_index in ion_indices
        ),
        standard_floating_wall=(
            boundary.transport_kind == "bohm"
            and boundary.sheath_energy_eV == 0.0
            and ion_indices.size == 1
            and float(charge_values[ion_indices[0]]) == 1.0
            and not np.any(charge_values < 0.0)
        ),
    )


def _resolved_sheath_energy_eV(
    compiled: CompiledWallBoundary, electrons: ElectronState
) -> float:
    """Return an explicit override or the standard floating-wall potential drop."""

    boundary = compiled.boundary
    if not compiled.standard_floating_wall:
        return boundary.sheath_energy_eV
    ion_mass_kg = float(compiled.masses_kg[0])
    mass_ratio = ion_mass_kg / (2.0 * math.pi * ELECTRON_MASS_KG)
    if mass_ratio <= 1.0:
        raise ModelConfigurationError(
            "Standard floating-wall closure requires ion mass > 2*pi*electron mass"
        )
    return 0.5 * electrons.temperature_eV * math.log(mass_ratio)


def _compiled_loss_frequency(boundary: WallBoundary) -> float:
    if (
        boundary.transport_kind == "prescribed_frequency"
        and boundary.prescribed_frequency_s_inv is not None
    ):
        return boundary.prescribed_frequency_s_inv
    if (
        boundary.transport_kind == "ambipolar"
        and boundary.diffusion_coefficient_m2_s is not None
        and boundary.diffusion_length_m is not None
    ):
        return boundary.diffusion_coefficient_m2_s / boundary.diffusion_length_m**2
    raise RuntimeError("compiled wall transport parameters are inconsistent")


def evaluate_compiled_wall_boundary(
    *,
    compiled: CompiledWallBoundary,
    volume_m3: float,
    species_ids: tuple[str, ...],
    densities_m3: np.ndarray,
    electrons: ElectronState,
    collect_records: bool = True,
) -> WallEvaluation:
    """Evaluate an indexed wall kernel without rebuilding static mappings."""

    boundary = compiled.boundary
    derivative = np.zeros_like(densities_m3, dtype=float)
    flux_by_species = np.zeros_like(densities_m3, dtype=float)
    records: list[WallFluxRecord] = []
    total_energy_loss = 0.0
    area_over_volume = boundary.area_m2 / volume_m3
    if boundary.transport_kind == "off":
        return WallEvaluation(derivative, 0.0, (), flux_by_species, 0.0)

    sheath_energy_eV = _resolved_sheath_energy_eV(compiled, electrons)

    for ion_slot, species_index_raw in enumerate(compiled.ion_indices):
        species_index = int(species_index_raw)
        species_id = species_ids[species_index]
        charge = float(compiled.charges[ion_slot])
        mass = float(compiled.masses_kg[ion_slot])
        # Only the mass-action/transport view is clipped. The integrated state
        # remains untouched and is rejected by the caller below -10*atol.
        ion_density = max(float(densities_m3[species_index]), 0.0)
        if boundary.transport_kind == "bohm":
            transport_speed = boundary.bohm_factor * math.sqrt(
                charge * ELEMENTARY_CHARGE_C * electrons.temperature_eV / mass
            )
            incident_flux = ion_density * transport_speed
            incident_rate = incident_flux * area_over_volume
        else:
            loss_frequency = _compiled_loss_frequency(boundary)
            incident_rate = ion_density * loss_frequency
            incident_flux = incident_rate / area_over_volume
            transport_speed = loss_frequency / area_over_volume
        flux_by_species[species_index] = incident_flux
        derivative[species_index] -= incident_rate

        branch_rates: dict[str, float] = {}
        for branch in compiled.branches_by_ion[ion_slot]:
            branch_rate = incident_rate * branch.probability
            if collect_records:
                branch_rates[branch.reaction_id] = branch_rate
            if branch.product_indices.size:
                np.add.at(
                    derivative,
                    branch.product_indices,
                    branch.product_yields * branch_rate,
                )

        loss_per_ion_eV = charge * 2.0 * electrons.temperature_eV + sheath_energy_eV
        energy_loss = incident_rate * loss_per_ion_eV * ELEMENTARY_CHARGE_C
        total_energy_loss += energy_loss
        if collect_records:
            records.append(
                WallFluxRecord(
                    zone_id=boundary.zone_id,
                    surface_id=boundary.surface_id,
                    transport_kind=boundary.transport_kind,
                    incident_species=species_id,
                    incident_flux_m2_s=incident_flux,
                    incident_rate_m3_s=incident_rate,
                    bohm_speed_m_s=transport_speed,
                    electron_energy_loss_J_m3_s=energy_loss,
                    branch_rates_m3_s=branch_rates,
                )
            )
    return WallEvaluation(
        derivative,
        total_energy_loss,
        tuple(records),
        flux_by_species,
        sheath_energy_eV,
    )


__all__ = [
    "BoundaryReaction",
    "CompiledWallBoundary",
    "WallBoundary",
    "WallEvaluation",
    "WallFluxRecord",
    "compile_wall_boundary",
    "evaluate_compiled_wall_boundary",
]
