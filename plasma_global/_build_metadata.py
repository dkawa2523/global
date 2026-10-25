"""Metadata and solver-setting construction for compiled cases."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import yaml

from plasma_global._audit_types import CaseAssessment
from plasma_global._provenance import build_artifact_provenance
from plasma_global.chemistry._maxwell import maxwell_integration_policy
from plasma_global.chemistry.compile import CompiledChemistry
from plasma_global.chemistry.data import ChemistryData
from plasma_global.core.domain import SolverSettings
from plasma_global.input._resolved_case import ResolvedCase
from plasma_global.input._serialize import case_document
from plasma_global.models.elastic import momentum_cross_section_ids
from plasma_global.models.walls import BOHM_H_FACTOR_CLOSURE_VERSION


def _effective_case_yaml(case: ResolvedCase) -> str:
    return yaml.safe_dump(
        case_document(case),
        sort_keys=False,
        allow_unicode=True,
    )


def _wall_transport_closure_provenance(case: ResolvedCase) -> dict[str, Any] | None:
    surface_modes = {
        surface.surface_id: (
            "auto" if surface.wall_transport.h_factor == "auto" else "numeric"
        )
        for surface in case.reactor.surfaces
        if surface.wall_transport.kind == "bohm"
    }
    if not surface_modes:
        return None
    return {
        "bohm_h_factor": {
            "version": BOHM_H_FACTOR_CLOSURE_VERSION,
            "surface_modes": surface_modes,
        }
    }


def _model_ids(
    case: ResolvedCase,
    chemistry: CompiledChemistry,
    experimental_features: tuple[str, ...],
    electron_kinetics_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "electrons": case.models.electrons.kind,
        "electron_closure": case.models.electron_closure.kind,
        "electron_density": case.models.electron_density.kind,
        "gas_energy": case.models.gas_energy.kind,
        "surface_kinetics": (
            "compiled" if case.models.surface_kinetics is not None else None
        ),
        "power_ports": {
            port.port_id: port.model.kind for port in case.reactor.power_ports
        },
        "wall_transport": {
            surface.surface_id: surface.wall_transport.kind
            for surface in case.reactor.surfaces
        },
        "elastic_heating": (
            (
                "maxwellian_momentum_cross_sections"
                if case.models.electrons.kind == "maxwellian"
                else "tabulated_momentum_rates"
            )
            if momentum_cross_section_ids(chemistry)
            or electron_kinetics_provenance.get("momentum_rate_ids")
            else None
        ),
        "experimental_accumulator": experimental_features or None,
    }


def build_metadata(
    case: ResolvedCase,
    chemistry_data: ChemistryData,
    chemistry: CompiledChemistry,
    missing_momentum_targets: tuple[str, ...],
    electron_kinetics_provenance: Mapping[str, Any],
    experimental_features: tuple[str, ...],
    assessment: CaseAssessment,
) -> Mapping[str, Any]:
    """Build immutable, self-describing artifact metadata."""

    domain_provenance: dict[str, Any] = {
        "case_source": None if case.source_path is None else str(case.source_path),
        "included_files": [str(path) for path in case.included_files],
        "chemistry_source": str(chemistry_data.source),
        "chemistry": dict(chemistry.provenance),
        "missing_momentum_cross_section_targets": missing_momentum_targets,
    }
    wall_transport_closure = _wall_transport_closure_provenance(case)
    if wall_transport_closure is not None:
        domain_provenance["wall_transport_closure"] = wall_transport_closure
    if electron_kinetics_provenance:
        domain_provenance["electron_kinetics"] = dict(electron_kinetics_provenance)
    provenance = build_artifact_provenance(case, chemistry_data, domain_provenance)
    return MappingProxyType(
        {
            "effective_case_yaml": _effective_case_yaml(case),
            "model_ids": _model_ids(
                case, chemistry, experimental_features, electron_kinetics_provenance
            ),
            "provenance": provenance,
            "summary_series": tuple(case.output.summary_series),
            "assessment": assessment.to_dict(),
            **(
                {"maxwellian_integration": maxwell_integration_policy()}
                if case.models.electrons.kind == "maxwellian"
                else {}
            ),
        }
    )


def solver_settings(case: ResolvedCase) -> SolverSettings:
    return SolverSettings(
        rtol=case.solver.rtol,
        atol=case.solver.atol,
        first_step_s=case.solver.first_step_s,
        max_step_s=case.solver.max_step_s,
        sample_interval_s=case.solver.sample_interval_s,
        save_at_s=(
            None if case.solver.save_at_s is None else tuple(case.solver.save_at_s)
        ),
    )
