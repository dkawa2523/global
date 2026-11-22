"""Record model-specific assumptions and omitted physical contributions."""

from __future__ import annotations

from typing import Any

from plasma_global._audit_types import AuditIssue, CaseAssessment


def _model_assumptions(case: Any) -> tuple[str, ...]:
    assumptions = [
        "Gas species and energy are spatial averages within each modeled zone.",
        f"Electron kinetics use {case.models.electrons.kind} closure.",
    ]
    if case.models.electrons.kind == "energy_balance":
        coefficients = case.models.electrons.coefficients
        if getattr(coefficients, "kind", None) == "maxwellian":
            assumptions.append(
                "Electron-impact cross-section integration uses a Maxwellian "
                "electron distribution."
            )
    if case.models.electron_density.kind == "quasineutral":
        assumptions.append("Electron density follows heavy-species charge neutrality.")
    if case.models.gas_energy.kind == "fixed":
        assumptions.append(
            "Gas temperature is prescribed; heavy-gas energy is not integrated."
        )
    for identity, port in case.reactor.power_ports.items():
        assumptions.append(f"Power port {identity!r} uses model {port.model.kind!r}.")
    for identity, surface in case.reactor.surfaces.items():
        assumptions.append(
            f"Surface {identity!r} uses wall transport {surface.wall_transport.kind!r}."
        )
    return tuple(assumptions)


def assess_case(
    case: Any,
    chemistry_data: Any,
    missing_momentum_targets: tuple[str, ...],
) -> CaseAssessment:
    """Explain configured models without assigning a whole-case accuracy label."""
    issues: list[AuditIssue] = []
    if missing_momentum_targets:
        targets = ", ".join(missing_momentum_targets)
        issues.append(
            AuditIssue(
                "WARNING",
                "MISSING_MOMENTUM_CROSS_SECTION",
                f"Missing neutral momentum-transfer data: {targets}; "
                "elastic transfer is omitted.",
                targets,
            )
        )
    if chemistry_data.surface_reactions and case.models.surface_kinetics is None:
        issues.append(
            AuditIssue(
                "WARNING",
                "SURFACE_REACTIONS_DISABLED",
                "Chemistry declares surface reactions, but "
                "models.surface_kinetics is absent; "
                "these reactions are not integrated.",
                "models.surface_kinetics",
            )
        )
    for identity, port in case.reactor.power_ports.items():
        if str(port.model.kind).startswith("experimental."):
            issues.append(
                AuditIssue(
                    "WARNING",
                    "POWER_MODEL_ASSUMPTIONS",
                    f"Power port {identity!r} uses {port.model.kind!r}; "
                    "interpret its outputs within that model's documented assumptions.",
                    f"reactor.power_ports.{identity}",
                )
            )
    empirical = tuple(
        f"Reaction coefficient model {identity!r} uses {model.kind!r}."
        for identity, model in chemistry_data.rate_models.items()
        if str(model.kind).startswith("experimental.")
    )
    return CaseAssessment((*_model_assumptions(case), *empirical), tuple(issues))


__all__ = ["assess_case"]
