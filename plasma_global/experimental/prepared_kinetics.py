"""Compile the exploratory two-term closure into immutable local-field tables.

The approximation is deliberately confined to case compilation.  The ODE core
only receives :class:`~plasma_global.models.kinetics.TabulatedElectronKinetics`
objects and therefore never solves an EEDF while evaluating the RHS.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global.chemistry.data import ChemistryData, CrossSectionData
from plasma_global.errors import CaseValidationError
from plasma_global.experimental.eedf import ApproximateTwoTermEEDF, ElectronCollision
from plasma_global.models.kinetics import TabulatedElectronKinetics

_MOMENTUM_KINDS = frozenset(
    {
        "elastic",
        "elastic_momentum",
        "momentum",
        "momentum_transfer",
        "mt",
    }
)


@dataclass(frozen=True, slots=True)
class PreparedApproximateTwoTermKinetics:
    """Zone bindings and auditable assumptions produced during compilation."""

    by_zone: Mapping[str, TabulatedElectronKinetics]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_zone", MappingProxyType(dict(self.by_zone)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _collision_kind(cross_section: CrossSectionData) -> str:
    return (
        "momentum"
        if cross_section.kind.strip().lower() in _MOMENTUM_KINDS
        else "inelastic"
    )


def _compile_collisions(chemistry: ChemistryData) -> tuple[ElectronCollision, ...]:
    if not chemistry.cross_sections:
        raise CaseValidationError(
            "experimental.approximate_two_term requires electron-neutral cross sections"
        )
    species = {item.id: item for item in chemistry.species}
    invalid_targets: list[str] = []
    collisions: list[ElectronCollision] = []
    for cross_section in chemistry.cross_sections.values():
        target = species.get(cross_section.target)
        if target is None or target.phase != "gas" or target.charge != 0:
            invalid_targets.append(cross_section.target)
            continue
        collisions.append(
            ElectronCollision(
                collision_id=cross_section.id,
                target_species=cross_section.target,
                kind=_collision_kind(cross_section),
                energy_eV=cross_section.energy_eV,
                cross_section_m2=cross_section.sigma_m2,
                energy_loss_eV=(
                    0.0
                    if _collision_kind(cross_section) == "momentum"
                    else cross_section.energy_loss_eV
                ),
            )
        )
    if invalid_targets:
        raise CaseValidationError(
            "experimental.approximate_two_term only accepts neutral gas targets; "
            f"invalid targets: {', '.join(sorted(set(invalid_targets)))}"
        )
    if not any(collision.kind == "momentum" for collision in collisions):
        raise CaseValidationError(
            "experimental.approximate_two_term requires a momentum-transfer cross section"
        )
    return tuple(collisions)


def _fixed_mixture(
    zone_id: str,
    densities_m3: Mapping[str, float],
    targets: tuple[str, ...],
    momentum_targets: frozenset[str],
) -> tuple[dict[str, float], dict[str, float]]:
    values = {target: float(densities_m3.get(target, 0.0)) for target in targets}
    invalid = [
        target
        for target, density in values.items()
        if not math.isfinite(density) or density < 0.0
    ]
    if invalid:
        raise CaseValidationError(
            f"zone {zone_id!r} has invalid initial collision-target densities: "
            f"{', '.join(invalid)}"
        )
    total = sum(values.values())
    if total <= 0.0:
        raise CaseValidationError(
            f"zone {zone_id!r} has no positive initial neutral collision-target density"
        )
    missing_momentum = sorted(
        target
        for target, density in values.items()
        if density > 0.0 and target not in momentum_targets
    )
    if missing_momentum:
        raise CaseValidationError(
            f"zone {zone_id!r} has positive collision targets without momentum-transfer "
            f"cross sections: {', '.join(missing_momentum)}"
        )
    fractions = {
        target: density / total for target, density in values.items() if density > 0.0
    }
    return values, fractions


def _mixture_identity(
    densities_m3: Mapping[str, float], targets: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    # Exact float identities prevent two physically different initial mixtures
    # from silently sharing mobility or rate data.  Human-facing rounded keys
    # remain provenance only.
    return tuple((target, float(densities_m3[target]).hex()) for target in targets)


def _readonly(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array.setflags(write=False)
    return array


def _prepare_table(
    *,
    source: Path,
    model: ApproximateTwoTermEEDF,
    field_grid_Td: np.ndarray,
    target_densities_m3: Mapping[str, float],
) -> TabulatedElectronKinetics:
    results = tuple(
        model.evaluate(float(field), target_densities_m3) for field in field_grid_Td
    )
    rate_ids = tuple(collision.collision_id for collision in model.collisions)
    rates = {
        rate_id: _readonly(
            [result.rate_coefficients_m3_s[rate_id] for result in results]
        )
        for rate_id in rate_ids
    }
    return TabulatedElectronKinetics(
        source=source,
        lookup="local_field",
        # Experimental power ports return E/N=0 while switched off.  The
        # configured positive grid represents the lowest resolved swarm state,
        # so endpoint clipping is explicit in provenance instead of extrapolation.
        bounds="clip",
        axis=_readonly(field_grid_Td),
        mean_energy_eV=_readonly([result.mean_energy_eV for result in results]),
        mobility_m2_V_s=_readonly([result.mobility_m2_V_s for result in results]),
        effective_field_Td=_readonly(field_grid_Td),
        rate_tables=MappingProxyType(rates),
    )


def prepare_approximate_two_term_kinetics(
    chemistry: ChemistryData,
    initial_densities_m3_by_zone: Mapping[str, Mapping[str, float]],
    *,
    mixture_key_species: Sequence[str] = (),
    cache_max_entries: int = 12,
    fraction_decimals: int = 3,
    energy_min_eV: float = 1.0e-3,
    energy_max_eV: float = 160.0,
    energy_points: int = 360,
    field_min_Td: float = 0.2,
    field_max_Td: float = 2500.0,
    field_points: int = 48,
    max_iterations: int = 48,
) -> PreparedApproximateTwoTermKinetics:
    """Prepare one strict local-field table per distinct initial neutral mixture."""

    if not initial_densities_m3_by_zone:
        raise CaseValidationError("approximate two-term preparation requires zones")
    if cache_max_entries < 1 or fraction_decimals < 0:
        raise CaseValidationError("approximate two-term cache settings are invalid")
    collisions = _compile_collisions(chemistry)
    targets = tuple(sorted({collision.target_species for collision in collisions}))
    requested_key = tuple(str(value) for value in mixture_key_species)
    if len(set(requested_key)) != len(requested_key):
        raise CaseValidationError("mixture_key_species must be unique")
    unknown_key = sorted(set(requested_key) - set(targets))
    if unknown_key:
        raise CaseValidationError(
            "mixture_key_species are not electron-neutral collision targets: "
            f"{', '.join(unknown_key)}"
        )

    try:
        field_grid = np.geomspace(
            float(field_min_Td), float(field_max_Td), int(field_points)
        )
        model = ApproximateTwoTermEEDF(
            collisions=collisions,
            energy_min_eV=float(energy_min_eV),
            energy_max_eV=float(energy_max_eV),
            energy_points=int(energy_points),
            max_iterations=int(max_iterations),
        )
    except (TypeError, ValueError) as exc:
        raise CaseValidationError(
            f"invalid approximate two-term preparation grid: {exc}"
        ) from exc
    if (
        field_grid.ndim != 1
        or field_grid.size < 2
        or not np.isfinite(field_grid).all()
        or np.any(np.diff(field_grid) <= 0.0)
    ):
        raise CaseValidationError(
            "approximate two-term reduced-field grid must be finite and strictly increasing"
        )
    field_grid.setflags(write=False)

    momentum_targets = frozenset(
        collision.target_species
        for collision in collisions
        if collision.kind == "momentum"
    )
    tables: dict[tuple[tuple[str, str], ...], TabulatedElectronKinetics] = {}
    zone_tables: dict[str, TabulatedElectronKinetics] = {}
    zone_provenance: dict[str, Any] = {}
    for zone_id, raw_densities in initial_densities_m3_by_zone.items():
        densities, fractions = _fixed_mixture(
            str(zone_id), raw_densities, targets, momentum_targets
        )
        identity = _mixture_identity(densities, targets)
        if identity not in tables:
            if len(tables) >= cache_max_entries:
                raise CaseValidationError(
                    "distinct initial neutral mixtures exceed cache.max_entries "
                    f"({cache_max_entries})"
                )
            try:
                tables[identity] = _prepare_table(
                    source=chemistry.source,
                    model=model,
                    field_grid_Td=field_grid,
                    target_densities_m3=densities,
                )
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise CaseValidationError(
                    f"failed to prepare approximate two-term kinetics for zone "
                    f"{zone_id!r}: {exc}"
                ) from exc
        table_index = tuple(tables).index(identity)
        zone_tables[str(zone_id)] = tables[identity]
        zone_provenance[str(zone_id)] = {
            "table_index": table_index,
            "target_densities_m3": dict(densities),
            "target_fractions": dict(fractions),
            "rounded_mixture_key": {
                species: round(float(fractions.get(species, 0.0)), fraction_decimals)
                for species in (requested_key or targets)
            },
        }

    provenance = {
        "kind": "experimental.approximate_two_term",
        "classification": "experimental_fixed_initial_neutral_mixture",
        "fixed_mixture": True,
        "runtime_eedf_solve": False,
        "closure": "local_field",
        "preparation": "compile_time_strict_geometric_grid",
        "bounds_policy": "clip_to_prepared_grid",
        "chemistry_manifest": str(chemistry.source),
        "cross_section_ids": tuple(collision.collision_id for collision in collisions),
        "mixture_key_species": requested_key or targets,
        "energy_grid_eV": {
            "minimum": float(energy_min_eV),
            "maximum": float(energy_max_eV),
            "count": int(energy_points),
        },
        "reduced_field_grid_Td": {
            "minimum": float(field_grid[0]),
            "maximum": float(field_grid[-1]),
            "count": int(field_grid.size),
        },
        "unique_table_count": len(tables),
        "zones": zone_provenance,
    }
    return PreparedApproximateTwoTermKinetics(zone_tables, provenance)


__all__ = [
    "PreparedApproximateTwoTermKinetics",
    "prepare_approximate_two_term_kinetics",
]
