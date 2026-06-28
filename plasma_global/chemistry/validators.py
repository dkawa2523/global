from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plasma_global.chemistry.models import MechanismBundle
from plasma_global.chemistry.provenance import RANGE_PROVENANCE_FIELDS, provenance_from_mapping


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


def _validate_provenance(report: ValidationReport, raw: Any, entity_kind: str, entity_id: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        report.add(
            'WARNING',
            'PROVENANCE_FORMAT_INVALID',
            f'{entity_kind} {entity_id} provenance should be a mapping',
            entity_id,
        )
        return {}
    provenance = provenance_from_mapping(raw)
    if '_invalid_provenance' in provenance:
        report.add(
            'WARNING',
            'PROVENANCE_FORMAT_INVALID',
            f'{entity_kind} {entity_id} provenance should be a mapping',
            entity_id,
        )
    for field_name in RANGE_PROVENANCE_FIELDS:
        value = provenance.get(field_name)
        if isinstance(value, (list, tuple)) and len(value) != 2:
            report.add(
                'WARNING',
                'PROVENANCE_RANGE_INVALID',
                f'{entity_kind} {entity_id} {field_name} should have two entries when given as a list',
                entity_id,
            )
    return provenance


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
        _validate_provenance(report, getattr(cs, 'metadata', {}) or {}, 'Cross section', cs_id)
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


@dataclass
class ReactionValidator:
    mechanism: MechanismBundle
    report: ValidationReport
    duplicate_signatures: dict[tuple, str] = field(default_factory=dict)

    def validate_all(self) -> None:
        for rxn in self.mechanism.gas_reactions + self.mechanism.surface_reactions:
            self.validate_reaction(rxn)

    def validate_reaction(self, rxn: Any) -> None:
        self._check_duplicate(rxn)
        self._check_species_and_models(rxn)
        rxn_provenance = _validate_provenance(self.report, getattr(rxn, 'provenance', {}) or {}, 'Reaction', rxn.reaction_id)
        self._check_phase_filters(rxn)
        self._check_rate_model(rxn, rxn_provenance)
        self._check_charge_balance(rxn)
        self._check_element_balance(rxn)
        self._check_site_balance(rxn)

    def _check_duplicate(self, rxn: Any) -> None:
        signature = self._signature(rxn)
        prev = self.duplicate_signatures.get(signature)
        if prev is not None:
            self.report.add('WARNING', 'DUPLICATE_REACTION', f'Reaction {rxn.reaction_id} duplicates {prev} under the same filters', rxn.reaction_id)
        else:
            self.duplicate_signatures[signature] = rxn.reaction_id

    def _signature(self, rxn: Any) -> tuple:
        return (
            rxn.phase,
            _normalized_side(rxn.reactants),
            _normalized_side(rxn.products),
            tuple(sorted(rxn.zone_filter)),
            tuple(sorted(rxn.surface_filter)),
            rxn.rate_model_key,
        )

    def _check_species_and_models(self, rxn: Any) -> None:
        for sp in set(rxn.reactants) | set(rxn.products):
            if sp not in self.mechanism.species_by_id:
                self.report.add('ERROR', 'UNKNOWN_SPECIES', f'Unknown species {sp} in reaction {rxn.equation}', rxn.reaction_id)
        if rxn.rate_model_key not in self.mechanism.rate_models:
            self.report.add('ERROR', 'UNKNOWN_RATE_MODEL', f'Unknown rate model {rxn.rate_model_key}', rxn.reaction_id)
        if rxn.energy_model_key and rxn.energy_model_key not in self.mechanism.rate_models:
            self.report.add('ERROR', 'UNKNOWN_ENERGY_MODEL', f'Unknown energy model {rxn.energy_model_key}', rxn.reaction_id)

    def _check_phase_filters(self, rxn: Any) -> None:
        if rxn.phase == 'surface' and not rxn.surface_filter:
            self.report.add('ERROR', 'SURFACE_REACTION_NO_SURFACE_FILTER', f'Surface reaction {rxn.reaction_id} has no surface_filter', rxn.reaction_id)
        if rxn.phase == 'gas' and rxn.surface_filter:
            self.report.add('WARNING', 'GAS_REACTION_SURFACE_FILTER', f'Gas reaction {rxn.reaction_id} has a surface_filter that will be ignored', rxn.reaction_id)

    def _check_rate_model(self, rxn: Any, rxn_provenance: dict[str, Any]) -> None:
        if rxn.rate_model_key not in self.mechanism.rate_models:
            return
        model = self.mechanism.rate_models[rxn.rate_model_key]
        model_provenance = _validate_provenance(
            self.report,
            model.get('provenance') if 'provenance' in model else model,
            'Rate model',
            rxn.rate_model_key,
        )
        backend = str(model.get('backend', '')).lower()
        if backend == 'electron_impact_xsec':
            self._check_electron_impact_model(rxn, model, rxn_provenance, model_provenance)
        if backend == 'first_order_loss':
            self._check_first_order_loss_model(rxn, model)
        self._check_coverage_model(rxn, model)

    def _check_electron_impact_model(self, rxn: Any, model: dict[str, Any], rxn_provenance: dict[str, Any], model_provenance: dict[str, Any]) -> None:
        cs_id = model.get('cross_section_id')
        if cs_id not in self.mechanism.cross_sections:
            self.report.add('ERROR', 'UNKNOWN_CROSS_SECTION', f'Unknown cross section {cs_id}', rxn.reaction_id)
        self._check_cross_section_provenance_match(rxn, cs_id, rxn_provenance.get('cross_section_id'), rxn.reaction_id, 'Reaction')
        self._check_cross_section_provenance_match(rxn, cs_id, model_provenance.get('cross_section_id'), rxn.rate_model_key, 'Rate model')

    def _check_cross_section_provenance_match(self, rxn: Any, cs_id: Any, declared_cs: Any, entity_id: str, entity_kind: str) -> None:
        if declared_cs and cs_id and str(declared_cs) != str(cs_id):
            subject = f'Reaction {rxn.reaction_id}' if entity_kind == 'Reaction' else f'Rate model {rxn.rate_model_key}'
            target = f'rate model {cs_id}' if entity_kind == 'Reaction' else f'model {cs_id}'
            self.report.add(
                'WARNING',
                'PROVENANCE_CROSS_SECTION_MISMATCH',
                f'{subject} provenance cross_section_id {declared_cs} differs from {target}',
                entity_id,
            )

    def _check_first_order_loss_model(self, rxn: Any, model: dict[str, Any]) -> None:
        if self._first_order_loss_rate_is_invalid(model):
            self.report.add(
                'ERROR',
                'FIRST_ORDER_LOSS_RATE_INVALID',
                'first_order_loss requires a non-negative rate_s_inv value',
                rxn.reaction_id,
            )
        if not self._first_order_loss_reaction_form_is_valid(rxn):
            self.report.add(
                'ERROR',
                'FIRST_ORDER_LOSS_REACTION_FORM',
                'first_order_loss is only for gas reactions with one non-electron reactant of stoichiometry 1',
                rxn.reaction_id,
            )

    @staticmethod
    def _first_order_loss_rate_is_invalid(model: dict[str, Any]) -> bool:
        rate = model.get('rate_s_inv', model.get('value'))
        try:
            return rate is None or float(rate) < 0.0
        except (TypeError, ValueError):
            return True

    def _first_order_loss_reaction_form_is_valid(self, rxn: Any) -> bool:
        reactants = [(sp, float(nu)) for sp, nu in rxn.reactants.items() if abs(float(nu)) > 1.0e-12]
        return (
            rxn.phase == 'gas'
            and len(reactants) == 1
            and abs(reactants[0][1] - 1.0) <= 1.0e-12
            and reactants[0][0] != self.mechanism.electron_species_id
        )

    def _check_coverage_model(self, rxn: Any, model: dict[str, Any]) -> None:
        coverage_factor = model.get('coverage_factor') or {}
        coverage_species = coverage_factor.get('species') or coverage_factor.get('site_species')
        if coverage_species and coverage_species not in self.mechanism.species_by_id:
            self.report.add('ERROR', 'UNKNOWN_COVERAGE_SPECIES', f'Coverage model references unknown species {coverage_species}', rxn.reaction_id)

    def _check_charge_balance(self, rxn: Any) -> None:
        charge_l = self._charge_sum(rxn.reactants)
        charge_r = self._charge_sum(rxn.products)
        if abs(charge_l - charge_r) > 1e-12:
            self.report.add('ERROR', 'CHARGE_NOT_CONSERVED', f'Charge balance failed in {rxn.equation}', rxn.reaction_id)

    def _charge_sum(self, side: dict[str, float]) -> float:
        return sum(self.mechanism.species_by_id[s].charge * nu for s, nu in side.items() if s in self.mechanism.species_by_id)

    def _check_element_balance(self, rxn: Any) -> None:
        left = self._element_sum(rxn.reactants)
        right = self._element_sum(rxn.products)
        for el in set(left) | set(right):
            if abs(left.get(el, 0.0) - right.get(el, 0.0)) > 1e-9:
                self.report.add('ERROR', 'ELEMENT_NOT_CONSERVED', f'Element {el} not conserved in {rxn.equation}', rxn.reaction_id)

    def _element_sum(self, side: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for sp, nu in side.items():
            if sp not in self.mechanism.species_by_id:
                continue
            for element, count in self.mechanism.species_by_id[sp].elements.items():
                out[element] = out.get(element, 0.0) + count * nu
        return out

    def _check_site_balance(self, rxn: Any) -> None:
        if rxn.phase == 'surface':
            left_site = self._site_sum(rxn.reactants)
            right_site = self._site_sum(rxn.products)
            if abs(left_site - right_site) > 1e-9:
                self.report.add('ERROR', 'SITE_NOT_CONSERVED', f'Surface site balance failed in {rxn.equation}', rxn.reaction_id)

    def _site_sum(self, side: dict[str, float]) -> float:
        return sum(float(self.mechanism.species_by_id[s].elements.get('site', 0.0)) * nu for s, nu in side.items() if s in self.mechanism.species_by_id)


def _validate_global_checks(mechanism: MechanismBundle, report: ValidationReport) -> None:
    if not mechanism.momentum_transfer_cross_sections:
        report.add(
            'WARNING',
            'NO_MOMENTUM_XS',
            'No momentum-transfer cross sections were provided; boltzmann_2term requires one for each neutral target species.',
        )


def validate_mechanism(mechanism: MechanismBundle) -> ValidationReport:
    report = ValidationReport()
    _validate_species(mechanism, report)
    _validate_cross_sections(mechanism, report)
    ReactionValidator(mechanism, report).validate_all()
    _validate_global_checks(mechanism, report)

    return report
