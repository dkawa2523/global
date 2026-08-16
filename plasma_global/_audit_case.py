"""Compile-and-run audit orchestration for one schema-v3 case."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plasma_global._audit_balance import (
    balance_issues as _balance_issues,
)
from plasma_global._audit_balance import (
    static_balance_maxima as _static_balance_maxima,
)
from plasma_global._audit_classification import (
    case_classification as _case_classification,
)
from plasma_global._audit_result import audit_result
from plasma_global._audit_types import AuditIssue, CaseAuditReport
from plasma_global.core.result import SimulationResult, to_plain_mapping


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
