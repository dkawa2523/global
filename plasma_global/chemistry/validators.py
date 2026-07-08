from __future__ import annotations

from typing import Any

from plasma_global.chemistry.extension_validation import validate_processes, validate_state_variables
from plasma_global.chemistry.models import MechanismBundle
from plasma_global.chemistry.provenance import validate_provenance
from plasma_global.chemistry.reaction_validation import ReactionValidator, validate_reactions
from plasma_global.validation import ValidationReport

__all__ = ['ReactionValidator', 'ValidationReport', 'validate_mechanism']


def _validate_species(mechanism: MechanismBundle, report: ValidationReport) -> None:
    alias_seen: dict[str, str] = {}
    for species in mechanism.species:
        for alias in [species.canonical_id, *species.aliases]:
            if alias in alias_seen and alias_seen[alias] != species.canonical_id:
                report.add('ERROR', 'ALIAS_COLLISION', f'Alias {alias} used by both {alias_seen[alias]} and {species.canonical_id}', species.canonical_id)
            else:
                alias_seen[alias] = species.canonical_id
        if species.phase == 'surface':
            if not species.surfaces:
                report.add('ERROR', 'SURFACE_SPECIES_NO_SURFACE', f'Surface species {species.canonical_id} has no surfaces list', species.canonical_id)
            site_occ = float(species.elements.get('site', 0.0))
            if site_occ <= 0.0:
                report.add('ERROR', 'SURFACE_SPECIES_NO_SITE', f'Surface species {species.canonical_id} is missing a positive site stoichiometry in elements', species.canonical_id)
        elif species.phase == 'gas' and 'site' in species.elements:
            report.add('WARNING', 'GAS_SPECIES_SITE_ELEMENT', f'Gas species {species.canonical_id} includes site stoichiometry; check if this is intentional', species.canonical_id)


def _validate_cross_sections(mechanism: MechanismBundle, report: ValidationReport) -> None:
    for cs_id, cs in mechanism.cross_sections.items():
        validate_provenance(report, getattr(cs, 'metadata', {}) or {}, 'Cross section', cs_id)
        if not cs.has_tabulated_data():
            report.add('ERROR', 'CROSS_SECTION_NO_DATA', f'Cross section {cs_id} has no tabulated data', cs_id)
            continue
        if cs.energy_eV[0] < 0.0 or (cs.sigma_m2 < 0.0).any():
            report.add('ERROR', 'CROSS_SECTION_INVALID', f'Cross section {cs_id} contains negative energy or sigma values', cs_id)
        if (cs.energy_eV[1:] < cs.energy_eV[:-1]).any():
            report.add('ERROR', 'CROSS_SECTION_NONMONOTONIC', f'Cross section {cs_id} energy grid is not monotonic increasing', cs_id)
        if cs.threshold_eV < 0.0:
            report.add('ERROR', 'CROSS_SECTION_NEGATIVE_THRESHOLD', f'Cross section {cs_id} has a negative threshold', cs_id)
        if cs.threshold_eV > float(cs.energy_eV[-1]):
            report.add('WARNING', 'CROSS_SECTION_THRESHOLD_OUTSIDE_RANGE', f'Cross section {cs_id} threshold exceeds maximum tabulated energy', cs_id)
        if cs.target_species and cs.target_species not in mechanism.species_by_id:
            report.add('ERROR', 'CROSS_SECTION_UNKNOWN_TARGET', f'Cross section {cs_id} targets unknown species {cs.target_species}', cs_id)


def _validate_global_checks(mechanism: MechanismBundle, report: ValidationReport) -> None:
    if not mechanism.momentum_transfer_cross_sections:
        report.add(
            'WARNING',
            'NO_MOMENTUM_XS',
            'No momentum-transfer cross sections were provided; this is only required when using swarm.model_name: boltzmann_2term.',
        )


def validate_mechanism(mechanism: MechanismBundle, chamber: Any | None = None) -> ValidationReport:
    report = ValidationReport()
    _validate_species(mechanism, report)
    _validate_cross_sections(mechanism, report)
    validate_reactions(mechanism, report)
    validate_state_variables(mechanism, report, chamber)
    validate_processes(mechanism, report, chamber)
    _validate_global_checks(mechanism, report)
    return report
