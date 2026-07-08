from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from plasma_global.chemistry.models import MechanismBundle
from plasma_global.chemistry.provenance import validate_provenance
from plasma_global.chemistry.rate_model_schema import COVERAGE_FACTOR_KINDS
from plasma_global.chemistry.rate_tables import BOUNDS_POLICIES, INTERPOLATIONS, TABULATED_1D_AXES


def _normalized_side(side: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((sp, float(nu)) for sp, nu in side.items() if abs(float(nu)) > 0.0))


def validate_reactions(mechanism: MechanismBundle, report: Any) -> None:
    ReactionValidator(mechanism, report).validate_all()


@dataclass
class ReactionValidator:
    mechanism: MechanismBundle
    report: Any
    duplicate_signatures: dict[tuple, str] = field(default_factory=dict)

    def validate_all(self) -> None:
        for rxn in self.mechanism.gas_reactions + self.mechanism.surface_reactions:
            self.validate_reaction(rxn)

    def validate_reaction(self, rxn: Any) -> None:
        self._check_duplicate(rxn)
        self._check_species_and_models(rxn)
        rxn_provenance = validate_provenance(self.report, getattr(rxn, 'provenance', {}) or {}, 'Reaction', rxn.reaction_id)
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
        model_provenance = validate_provenance(
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
        if backend in {'tabulated_1d', 'ion_yield_table'}:
            self._check_tabulated_rate_model(rxn, model)
        self._check_coverage_model(rxn, model)

    def _check_electron_impact_model(
        self,
        rxn: Any,
        model: dict[str, Any],
        rxn_provenance: dict[str, Any],
        model_provenance: dict[str, Any],
    ) -> None:
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

    def _check_tabulated_rate_model(self, rxn: Any, model: dict[str, Any]) -> None:
        backend = str(model.get('backend', '')).lower()
        if '_table_1d' not in model:
            self.report.add('ERROR', 'TABULATED_RATE_TABLE_MISSING', f'Rate model {rxn.rate_model_key} requires a loaded 1D table', rxn.reaction_id)
        if str(model.get('bounds_policy', 'clip')).lower() not in BOUNDS_POLICIES:
            self.report.add('ERROR', 'TABULATED_BOUNDS_POLICY_INVALID', f'Rate model {rxn.rate_model_key} bounds_policy must be one of {sorted(BOUNDS_POLICIES)}', rxn.reaction_id)
        if str(model.get('interpolation', 'linear')).lower() not in INTERPOLATIONS:
            self.report.add('ERROR', 'TABULATED_INTERPOLATION_INVALID', f'Rate model {rxn.rate_model_key} interpolation must be one of {sorted(INTERPOLATIONS)}', rxn.reaction_id)
        if backend == 'tabulated_1d' and str(model.get('x', '')).strip() not in TABULATED_1D_AXES:
            self.report.add('ERROR', 'TABULATED_AXIS_UNSUPPORTED', f'Rate model {rxn.rate_model_key} x must be one of {sorted(TABULATED_1D_AXES)}', rxn.reaction_id)

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
        kind = str(coverage_factor.get('kind', 'constant')).lower()
        if kind not in COVERAGE_FACTOR_KINDS:
            self.report.add('ERROR', 'COVERAGE_FACTOR_KIND_UNSUPPORTED', f'Coverage factor kind must be one of {sorted(COVERAGE_FACTOR_KINDS)}', rxn.reaction_id)
        coverage_species = coverage_factor.get('species') or coverage_factor.get('site_species')
        if kind in {'site_blocking', 'species_power'} and not coverage_species:
            self.report.add('ERROR', 'COVERAGE_SPECIES_MISSING', f'Coverage model for {rxn.reaction_id} requires species or site_species', rxn.reaction_id)
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
