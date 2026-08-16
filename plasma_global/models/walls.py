"""Single-source ion wall flux and boundary-reaction bookkeeping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from plasma_global.errors import ModelConfigurationError, StateDomainError
from plasma_global.models.electrons import ElectronState

ELECTRON_MASS_KG = 9.1093837139e-31
# Bump whenever numeric h-factor semantics or the automatic closure equation changes.
BOHM_H_FACTOR_CLOSURE_VERSION = "direct-multiplier-v2"


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


@dataclass(frozen=True, slots=True)
class AutoBohmHFactor:
    """Density-dependent sheath-edge to bulk ion-density ratio."""

    characteristic_length_m: float
    ion_neutral_cross_section_m2: float
    min_h_factor: float
    max_h_factor: float

    def __post_init__(self) -> None:
        values = (
            self.characteristic_length_m,
            self.ion_neutral_cross_section_m2,
            self.min_h_factor,
            self.max_h_factor,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ModelConfigurationError(
                "Automatic Bohm h-factor parameters must be finite and positive"
            )
        if self.min_h_factor > self.max_h_factor or self.max_h_factor > 1.0:
            raise ModelConfigurationError(
                "Automatic Bohm h-factor limits must satisfy 0 < min <= max <= 1"
            )

    def evaluate(self, neutral_density_m3: float) -> float:
        """Evaluate the collisional edge-to-bulk closure at current density."""

        if not math.isfinite(neutral_density_m3) or neutral_density_m3 < 0.0:
            raise StateDomainError(
                "Automatic Bohm h-factor requires finite nonnegative neutral density"
            )
        collisionality = (
            self.characteristic_length_m
            * neutral_density_m3
            * self.ion_neutral_cross_section_m2
            / 2.0
        )
        value = 0.86 / math.sqrt(3.0 + collisionality)
        return min(max(value, self.min_h_factor), self.max_h_factor)


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


def _validate_auto_bohm_h_factor(kind: str, closure: AutoBohmHFactor | None) -> None:
    if closure is not None and kind != "bohm":
        raise ModelConfigurationError(
            "Automatic Bohm h-factor is only valid for Bohm wall transport"
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


def _validate_wall_id(name: str, value: str) -> None:
    if not value:
        raise ModelConfigurationError(f"Wall boundary {name} must not be empty")


def _validate_wall_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ModelConfigurationError(
            f"Wall boundary {name} must be finite and non-negative"
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
    auto_bohm_h_factor: AutoBohmHFactor | None = None

    def __post_init__(self) -> None:
        _validate_wall_id("zone_id", self.zone_id)
        _validate_wall_nonnegative("area_m2", self.area_m2)
        _validate_wall_nonnegative("bohm_factor", self.bohm_factor)
        _validate_wall_nonnegative("sheath_energy_eV", self.sheath_energy_eV)
        if self.surface_id is not None:
            _validate_wall_id("surface_id", self.surface_id)
        _validate_auto_bohm_h_factor(
            self.transport_kind,
            self.auto_bohm_h_factor,
        )
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
    neutral_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def compile_wall_boundary(
    *,
    boundary: WallBoundary,
    species_ids: tuple[str, ...],
    charges: np.ndarray,
    masses_kg: np.ndarray,
) -> CompiledWallBoundary:
    """Resolve one wall's static species topology outside the ODE hot path."""

    from plasma_global.models._wall_compile import compile_wall_boundary_data

    return compile_wall_boundary_data(
        boundary=boundary,
        species_ids=species_ids,
        charges=charges,
        masses_kg=masses_kg,
    )


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

    from plasma_global.models._wall_runtime import evaluate_wall

    return evaluate_wall(
        compiled=compiled,
        volume_m3=volume_m3,
        species_ids=species_ids,
        densities_m3=densities_m3,
        electrons=electrons,
        collect_records=collect_records,
    )


__all__ = [
    "BOHM_H_FACTOR_CLOSURE_VERSION",
    "AutoBohmHFactor",
    "BoundaryReaction",
    "CompiledWallBoundary",
    "WallBoundary",
    "WallEvaluation",
    "WallFluxRecord",
    "compile_wall_boundary",
    "evaluate_compiled_wall_boundary",
]
