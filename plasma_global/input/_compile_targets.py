"""Bind chemistry and profile selectors to the case's concrete geometry."""

from __future__ import annotations

from plasma_global.chemistry.data import ChemistryData, ReactionData
from plasma_global.errors import CaseValidationError
from plasma_global.input._resolved_case import ResolvedCase


def validate_chemistry_targets(case: ResolvedCase, chemistry: ChemistryData) -> None:
    """Reject misspelled selectors and empty zone/surface intersections."""

    zones = {zone.zone_id for zone in case.reactor.zones}
    surface_zones = {
        surface.surface_id: surface.zone_id for surface in case.reactor.surfaces
    }
    families = (
        ("gas", chemistry.gas_reactions),
        ("boundary", chemistry.boundary_reactions),
        (
            "surface",
            chemistry.surface_reactions
            if case.models.surface_kinetics is not None
            else (),
        ),
    )
    for family, reactions in families:
        for reaction in reactions:
            _validate_selection(
                reaction.zones, zones, f"reaction {reaction.id!r} zones"
            )
            _validate_selection(
                reaction.surfaces,
                set(surface_zones),
                f"reaction {reaction.id!r} surfaces",
            )
            if family != "gas":
                _validate_surface_intersection(reaction, surface_zones)


def _validate_selection(
    selected: tuple[str, ...], available: set[str], where: str
) -> None:
    if len(set(selected)) != len(selected):
        raise CaseValidationError(f"{where} has duplicate targets")
    if unknown := set(selected) - available:
        raise CaseValidationError(f"{where} selects unknown targets: {sorted(unknown)}")


def _validate_surface_intersection(
    reaction: ReactionData, surface_zones: dict[str, str]
) -> None:
    if not reaction.zones and not reaction.surfaces:
        return
    if not any(
        not reaction.zones or surface_zones[surface] in reaction.zones
        for surface in reaction.surfaces or surface_zones
    ):
        raise CaseValidationError(
            f"reaction {reaction.id!r} has no surface in its selected zones"
        )
