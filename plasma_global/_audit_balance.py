"""Compute static chemistry and dynamic ledger audit findings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plasma_global._audit_types import AuditIssue

BALANCE_TOLERANCE = 1.0e-12
DYNAMIC_LEDGER_NAMES = (
    "particle_ledger_normalized",
    "electron_energy_ledger_normalized",
    "heavy_energy_ledger_normalized",
)


def _normalized_balance(delta: float, before: float) -> float:
    return abs(delta) / max(abs(before), 1.0)


def _reaction_element_residuals(
    reaction: Any, species: dict[str, Any]
) -> dict[str, float]:
    elements = {
        element
        for species_id in (*reaction.reactants, *reaction.products)
        for element in species[species_id].elements
    }
    residuals: dict[str, float] = {}
    for element in elements:
        before = sum(
            order * species[species_id].elements.get(element, 0.0)
            for species_id, order in reaction.reactants.items()
        )
        after = sum(
            order * species[species_id].elements.get(element, 0.0)
            for species_id, order in reaction.products.items()
        )
        residuals[element] = _normalized_balance(after - before, before)
    return residuals


def _reaction_charge_residual(reaction: Any, species: dict[str, Any]) -> float:
    before = sum(
        order * species[species_id].charge
        for species_id, order in reaction.reactants.items()
    )
    after = sum(
        order * species[species_id].charge
        for species_id, order in reaction.products.items()
    )
    return _normalized_balance(after - before, before)


def static_balance_maxima(chemistry_data: Any) -> dict[str, float]:
    species = {item.id: item for item in chemistry_data.species}
    maxima = {
        "normalized_element": 0.0,
        "normalized_charge": 0.0,
        "normalized_site": 0.0,
    }
    for family, reactions in (
        ("gas", chemistry_data.gas_reactions),
        ("boundary", chemistry_data.boundary_reactions),
        ("surface", chemistry_data.surface_reactions),
    ):
        for reaction in reactions:
            for element, residual in _reaction_element_residuals(
                reaction, species
            ).items():
                quantity = (
                    "normalized_site" if element == "site" else "normalized_element"
                )
                maxima[quantity] = max(maxima[quantity], residual)
            if family != "boundary":
                maxima["normalized_charge"] = max(
                    maxima["normalized_charge"],
                    _reaction_charge_residual(reaction, species),
                )
    return maxima


def balance_issues(
    case: Any,
    static: Mapping[str, float],
    dynamic: Mapping[str, float],
) -> list[AuditIssue]:
    issues = [
        AuditIssue(
            "ERROR",
            "STATIC_REACTION_BALANCE_EXCEEDED",
            (
                f"Static reaction balance {name!r} is {value}, "
                f"above {BALANCE_TOLERANCE}."
            ),
            name,
        )
        for name, value in static.items()
        if value > BALANCE_TOLERANCE
    ]
    issues.extend(
        AuditIssue(
            "ERROR",
            "DYNAMIC_LEDGER_RESIDUAL_EXCEEDED",
            (
                f"Dynamic ledger residual {name!r} is {dynamic[name]}, "
                f"above {BALANCE_TOLERANCE}."
            ),
            name,
        )
        for name in DYNAMIC_LEDGER_NAMES
        if dynamic.get(name, 0.0) > BALANCE_TOLERANCE
    )
    if (
        case.models.electron_density.kind == "quasineutral"
        and dynamic["charge_closure_normalized"] > BALANCE_TOLERANCE
    ):
        issues.append(
            AuditIssue(
                "ERROR",
                "CHARGE_CLOSURE_RESIDUAL_EXCEEDED",
                "Quasineutral charge-closure residual exceeds 1e-12.",
                "charge_closure_normalized",
            )
        )
    return issues


__all__ = ["balance_issues", "static_balance_maxima"]
