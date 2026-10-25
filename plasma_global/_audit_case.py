"""Run the shared case assessment without replaying algebraic ledger identities."""

from __future__ import annotations

from typing import Any

from plasma_global._audit_types import CaseAuditReport
from plasma_global.core.result import to_plain_mapping


def audit_case(case: Any) -> CaseAuditReport:
    """Compile and integrate a case, reporting the same assessment as normal runs."""

    from plasma_global.build import _simulate_compiled_case, compile_case

    compiled = compile_case(case)
    result = _simulate_compiled_case(compiled)
    chemistry = compiled.chemistry_data
    return CaseAuditReport(
        assessment=compiled.assessment,
        model_ids=compiled.metadata["model_ids"],
        provenance=compiled.metadata["provenance"],
        inventory={
            "zones": len(compiled.case.reactor.zones),
            "species": len(chemistry.species),
            "gas_reactions": len(chemistry.gas_reactions),
            "boundary_reactions": len(chemistry.boundary_reactions),
            "surface_reactions": len(chemistry.surface_reactions),
            "recipe_segments": len(compiled.segments),
            "state_variables": compiled.model.layout.size,
        },
        simulation={
            "status": {
                "success": result.status.success,
                "code": result.status.code,
                "message": result.status.message,
            },
            "time_count": result.n_times,
            "start_s": float(result.time_s[0]),
            "end_s": float(result.time_s[-1]),
            "solver": to_plain_mapping(result.solver_stats),
        },
    )
