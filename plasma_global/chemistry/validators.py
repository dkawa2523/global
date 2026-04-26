from __future__ import annotations

from dataclasses import dataclass, field

from plasma_global.chemistry.models import MechanismBundle


@dataclass
class ValidationMessage:
    level: str
    code: str
    message: str
    entity_id: str = ''


@dataclass
class ValidationReport:
    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(m.level.upper() == 'ERROR' for m in self.messages)

    def add(self, level: str, code: str, message: str, entity_id: str = '') -> None:
        self.messages.append(ValidationMessage(level=level, code=code, message=message, entity_id=entity_id))


def _normalized_side(side: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((sp, float(nu)) for sp, nu in side.items() if abs(float(nu)) > 0.0))


def validate_mechanism(mechanism: MechanismBundle) -> ValidationReport:
    report = ValidationReport()

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

    for cs_id, cs in mechanism.cross_sections.items():
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

    duplicate_signatures: dict[tuple, str] = {}

    def _check_reaction(rxn):
        signature = (
            rxn.phase,
            _normalized_side(rxn.reactants),
            _normalized_side(rxn.products),
            tuple(sorted(rxn.zone_filter)),
            tuple(sorted(rxn.surface_filter)),
            rxn.rate_model_key,
        )
        prev = duplicate_signatures.get(signature)
        if prev is not None:
            report.add('WARNING', 'DUPLICATE_REACTION', f'Reaction {rxn.reaction_id} duplicates {prev} under the same filters', rxn.reaction_id)
        else:
            duplicate_signatures[signature] = rxn.reaction_id

        for sp in set(rxn.reactants) | set(rxn.products):
            if sp not in mechanism.species_by_id:
                report.add('ERROR', 'UNKNOWN_SPECIES', f'Unknown species {sp} in reaction {rxn.equation}', rxn.reaction_id)
        if rxn.rate_model_key not in mechanism.rate_models:
            report.add('ERROR', 'UNKNOWN_RATE_MODEL', f'Unknown rate model {rxn.rate_model_key}', rxn.reaction_id)
        if rxn.energy_model_key and rxn.energy_model_key not in mechanism.rate_models:
            report.add('ERROR', 'UNKNOWN_ENERGY_MODEL', f'Unknown energy model {rxn.energy_model_key}', rxn.reaction_id)

        if rxn.phase == 'surface' and not rxn.surface_filter:
            report.add('ERROR', 'SURFACE_REACTION_NO_SURFACE_FILTER', f'Surface reaction {rxn.reaction_id} has no surface_filter', rxn.reaction_id)
        if rxn.phase == 'gas' and rxn.surface_filter:
            report.add('WARNING', 'GAS_REACTION_SURFACE_FILTER', f'Gas reaction {rxn.reaction_id} has a surface_filter that will be ignored', rxn.reaction_id)

        if rxn.rate_model_key in mechanism.rate_models:
            model = mechanism.rate_models[rxn.rate_model_key]
            backend = str(model.get('backend', '')).lower()
            if backend == 'electron_impact_xsec':
                cs_id = model.get('cross_section_id')
                if cs_id not in mechanism.cross_sections:
                    report.add('ERROR', 'UNKNOWN_CROSS_SECTION', f'Unknown cross section {cs_id}', rxn.reaction_id)
            if backend == 'first_order_loss':
                rate = model.get('rate_s_inv', model.get('value'))
                try:
                    invalid_rate = rate is None or float(rate) < 0.0
                except (TypeError, ValueError):
                    invalid_rate = True
                if invalid_rate:
                    report.add(
                        'ERROR',
                        'FIRST_ORDER_LOSS_RATE_INVALID',
                        'first_order_loss requires a non-negative rate_s_inv value',
                        rxn.reaction_id,
                    )
                reactants = [(sp, float(nu)) for sp, nu in rxn.reactants.items() if abs(float(nu)) > 1.0e-12]
                valid_reactant = (
                    rxn.phase == 'gas'
                    and len(reactants) == 1
                    and abs(reactants[0][1] - 1.0) <= 1.0e-12
                    and reactants[0][0] != mechanism.electron_species_id
                )
                if not valid_reactant:
                    report.add(
                        'ERROR',
                        'FIRST_ORDER_LOSS_REACTION_FORM',
                        'first_order_loss is only for gas reactions with one non-electron reactant of stoichiometry 1',
                        rxn.reaction_id,
                    )
            coverage_factor = model.get('coverage_factor') or {}
            coverage_species = coverage_factor.get('species') or coverage_factor.get('site_species')
            if coverage_species and coverage_species not in mechanism.species_by_id:
                report.add('ERROR', 'UNKNOWN_COVERAGE_SPECIES', f'Coverage model references unknown species {coverage_species}', rxn.reaction_id)

        charge_l = sum(mechanism.species_by_id[s].charge * nu for s, nu in rxn.reactants.items() if s in mechanism.species_by_id)
        charge_r = sum(mechanism.species_by_id[s].charge * nu for s, nu in rxn.products.items() if s in mechanism.species_by_id)
        if abs(charge_l - charge_r) > 1e-12:
            report.add('ERROR', 'CHARGE_NOT_CONSERVED', f'Charge balance failed in {rxn.equation}', rxn.reaction_id)

        def _elem_sum(side):
            out: dict[str, float] = {}
            for sp, nu in side.items():
                if sp not in mechanism.species_by_id:
                    continue
                for el, cnt in mechanism.species_by_id[sp].elements.items():
                    out[el] = out.get(el, 0.0) + cnt * nu
            return out

        left = _elem_sum(rxn.reactants)
        right = _elem_sum(rxn.products)
        for el in set(left) | set(right):
            if abs(left.get(el, 0.0) - right.get(el, 0.0)) > 1e-9:
                report.add('ERROR', 'ELEMENT_NOT_CONSERVED', f'Element {el} not conserved in {rxn.equation}', rxn.reaction_id)

        if rxn.phase == 'surface':
            left_site = sum(float(mechanism.species_by_id[s].elements.get('site', 0.0)) * nu for s, nu in rxn.reactants.items() if s in mechanism.species_by_id)
            right_site = sum(float(mechanism.species_by_id[s].elements.get('site', 0.0)) * nu for s, nu in rxn.products.items() if s in mechanism.species_by_id)
            if abs(left_site - right_site) > 1e-9:
                report.add('ERROR', 'SITE_NOT_CONSERVED', f'Surface site balance failed in {rxn.equation}', rxn.reaction_id)

    for rxn in mechanism.gas_reactions + mechanism.surface_reactions:
        _check_reaction(rxn)

    if not mechanism.momentum_transfer_cross_sections:
        report.add('WARNING', 'NO_MOMENTUM_XS', 'No momentum-transfer cross sections were provided; swarm models will fall back to synthetic transport channels')

    return report
