"""Compile-and-run audit orchestration for one schema-v3 case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plasma_global._audit_result import audit_result
from plasma_global._audit_types import AuditIssue, CaseAuditReport
from plasma_global.core.result import SimulationResult, to_plain_mapping

_BALANCE_TOLERANCE = 1.0e-12
_DYNAMIC_LEDGER_NAMES = (
    "particle_ledger_normalized",
    "electron_energy_ledger_normalized",
    "heavy_energy_ledger_normalized",
)


def _normalized_balance(delta: float, before: float) -> float:
    return abs(delta) / max(abs(before), 1.0)


def _heavy_ions(chemistry_data: Any) -> tuple[Any, ...]:
    return tuple(
        item
        for item in chemistry_data.species
        if item.phase == "gas" and item.id != "e" and item.charge != 0
    )


def _standard_wall_transport(
    surface: Any,
    positive_ions: tuple[Any, ...],
    has_negative_ion: bool,
) -> bool:
    if surface.wall_transport.kind == "off":
        return True
    return (
        all(
            (
                surface.wall_transport.kind == "bohm",
                len(positive_ions) == 1,
                not has_negative_ion,
                surface.ion_impact_energy_eV == 0.0,
            )
        )
        and positive_ions[0].charge == 1
    )


def _is_applicable_wall_branch(reaction: Any, surface: Any, ion_id: str) -> bool:
    return all(
        (
            reaction.reactants == {ion_id: 1.0},
            bool(reaction.products),
            not reaction.zones or surface.zone_id in reaction.zones,
            not reaction.surfaces or surface.surface_id in reaction.surfaces,
        )
    )


def _has_standard_wall_branch(
    surface: Any,
    chemistry_data: Any,
    positive_ions: tuple[Any, ...],
    has_negative_ion: bool,
) -> bool:
    if surface.wall_transport.kind == "off":
        return True
    if not _standard_wall_transport(surface, positive_ions, has_negative_ion):
        return False
    ion_id = positive_ions[0].id
    branches = [
        reaction
        for reaction in chemistry_data.boundary_reactions
        if _is_applicable_wall_branch(reaction, surface, ion_id)
    ]
    return len(branches) == 1


def _electron_reactions_use_cross_sections(chemistry_data: Any) -> bool:
    return all(
        "e" not in reaction.reactants
        or (
            reaction.rate_model is not None
            and chemistry_data.rate_models[str(reaction.rate_model)].kind
            == "electron_impact"
        )
        for reaction in chemistry_data.gas_reactions
    )


def _case_classification(case: Any, chemistry_data: Any) -> str:
    """Classify the deliberately small, validated standard-model envelope."""

    heavy_ions = _heavy_ions(chemistry_data)
    positive_ions = tuple(item for item in heavy_ions if item.charge > 0)
    has_negative_ion = any(item.charge < 0 for item in heavy_ions)
    standard_walls = all(
        _has_standard_wall_branch(
            surface, chemistry_data, positive_ions, has_negative_ion
        )
        for surface in case.reactor.surfaces
    )
    one_singly_charged_positive_ion = (
        len(positive_ions) == 1 and positive_ions[0].charge == 1
    )
    standard = all(
        (
            case.models.electrons.kind == "maxwellian",
            case.models.electron_closure.kind == "electron_energy",
            case.models.electron_density.kind == "quasineutral",
            case.models.gas_energy.kind == "fixed",
            all(
                port.model.kind == "prescribed_power"
                for port in case.reactor.power_ports
            ),
            case.models.surface_kinetics is None,
            not chemistry_data.surface_reactions,
            standard_walls,
            _electron_reactions_use_cross_sections(chemistry_data),
            not has_negative_ion,
            one_singly_charged_positive_ion,
            not any(
                str(model.kind).startswith("experimental.")
                for model in chemistry_data.rate_models.values()
            ),
            not chemistry_data.experimental,
            case.experimental is None,
        )
    )
    return "standard" if standard else "experimental"


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


def _static_balance_maxima(chemistry_data: Any) -> dict[str, float]:
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


def _model_ids(case: Any, compiled: Any) -> dict[str, Any]:
    return {
        "electron_kinetics": case.models.electrons.kind,
        "electron_closure": case.models.electron_closure.kind,
        "electron_density": case.models.electron_density.kind,
        "gas_energy": case.models.gas_energy.kind,
        "power_ports": {
            item.port_id: item.model.kind for item in case.reactor.power_ports
        },
        "wall_transport": {
            item.surface_id: item.wall_transport.kind for item in case.reactor.surfaces
        },
        "elastic_heating": dict(compiled.metadata.get("model_ids", {})).get(
            "elastic_heating"
        ),
    }


def _experimental_model_ids(case: Any, chemistry_data: Any) -> tuple[str, ...]:
    identifiers = [
        str(value)
        for value in (
            case.models.electrons.kind,
            case.models.electron_density.kind,
            *(item.model.kind for item in case.reactor.power_ports),
            *(item.kind for item in chemistry_data.rate_models.values()),
        )
        if str(value).startswith("experimental.")
    ]
    if case.experimental is not None:
        identifiers.extend(
            f"experimental.{field_name}"
            for field_name in ("film", "wall_inventory", "extensions")
            if getattr(case.experimental, field_name) is not None
        )
    return tuple(dict.fromkeys(identifiers))


def _experimental_issues(case: Any, chemistry_data: Any) -> list[AuditIssue]:
    return [
        AuditIssue(
            "WARNING",
            "EXPERIMENTAL_MODEL",
            (
                f"{identifier} is an explicit experimental model; only "
                "finite behavior, conservation, and qualitative trends are claimed."
            ),
            identifier,
        )
        for identifier in _experimental_model_ids(case, chemistry_data)
    ]


def _missing_momentum_issues(provenance: Mapping[str, Any]) -> list[AuditIssue]:
    targets = dict.fromkeys(
        str(target)
        for target in provenance.get("missing_momentum_cross_section_targets", ())
    )
    return [
        AuditIssue(
            "WARNING",
            "MISSING_MOMENTUM_CROSS_SECTION",
            (
                f"Neutral target {target!r} has no momentum-transfer cross section; "
                "electron elastic gas heating is omitted for this target."
            ),
            target,
        )
        for target in targets
    ]


def _balance_issues(
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
                f"above {_BALANCE_TOLERANCE}."
            ),
            name,
        )
        for name, value in static.items()
        if value > _BALANCE_TOLERANCE
    ]
    issues.extend(
        AuditIssue(
            "ERROR",
            "DYNAMIC_LEDGER_RESIDUAL_EXCEEDED",
            (
                f"Dynamic ledger residual {name!r} is {dynamic[name]}, "
                f"above {_BALANCE_TOLERANCE}."
            ),
            name,
        )
        for name in _DYNAMIC_LEDGER_NAMES
        if dynamic.get(name, 0.0) > _BALANCE_TOLERANCE
    )
    if (
        case.models.electron_density.kind == "quasineutral"
        and dynamic["charge_closure_normalized"] > _BALANCE_TOLERANCE
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


def _inventory(case: Any, compiled: Any) -> dict[str, int]:
    chemistry_data = compiled.chemistry_data
    return {
        "zones": len(case.reactor.zones),
        "species": len(chemistry_data.species),
        "gas_reactions": len(chemistry_data.gas_reactions),
        "boundary_reactions": len(chemistry_data.boundary_reactions),
        "surface_reactions": len(chemistry_data.surface_reactions),
        "recipe_segments": len(compiled.segments),
        "state_variables": compiled.model.layout.size,
    }


def _simulation_summary(result: SimulationResult) -> dict[str, Any]:
    return {
        "status": {
            "success": result.status.success,
            "code": result.status.code,
            "message": result.status.message,
        },
        "time_count": result.n_times,
        "start_s": float(result.time_s[0]) if result.n_times else None,
        "end_s": float(result.time_s[-1]) if result.n_times else None,
        "solver": to_plain_mapping(result.solver_stats),
    }


def audit_case(case: Any) -> CaseAuditReport:
    """Compile, integrate, and audit one case without writing output files."""

    from plasma_global.build import _simulate_compiled_case, compile_case
    from plasma_global.input.schema import CaseSpec

    if not isinstance(case, CaseSpec):
        raise TypeError("case must be a schema-v3 CaseSpec")
    compiled = compile_case(case)
    result = _simulate_compiled_case(compiled, detailed_audit=True)
    chemistry_data = compiled.chemistry_data

    issues = _experimental_issues(case, chemistry_data)
    issues.extend(audit_result(result).issues)
    compiled_provenance = to_plain_mapping(result.metadata)["provenance"]
    issues.extend(_missing_momentum_issues(compiled_provenance))

    provenance = dict(compiled_provenance)
    dynamic = dict(provenance["runtime_diagnostics"]["conservation_max_abs_residual"])
    static = _static_balance_maxima(chemistry_data)
    issues.extend(_balance_issues(case, static, dynamic))
    classification = _case_classification(case, chemistry_data)

    return CaseAuditReport(
        issues=tuple(issues),
        model_ids=_model_ids(case, compiled),
        provenance=provenance,
        inventory=_inventory(case, compiled),
        conservation_max_abs_residual={**static, **dynamic},
        classification=classification,
        production_qualified=(classification == "standard" and not issues),
        simulation=_simulation_summary(result),
    )
