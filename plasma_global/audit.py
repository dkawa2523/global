"""Pure quality checks for canonical simulation results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from plasma_global.core.result import SimulationResult, to_plain_mapping


@dataclass(frozen=True, slots=True)
class CaseAuditReport:
    """Detailed, read-only audit of one compiled schema-v3 case."""

    issues: tuple[AuditIssue, ...]
    model_ids: Mapping[str, Any]
    provenance: Mapping[str, Any]
    inventory: Mapping[str, int]
    conservation_max_abs_residual: Mapping[str, float]
    simulation: Mapping[str, Any] = field(default_factory=dict)
    classification: str = "standard"
    production_qualified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "issues", tuple(self.issues))
        for name in (
            "model_ids",
            "provenance",
            "inventory",
            "conservation_max_abs_residual",
            "simulation",
        ):
            value = dict(getattr(self, name))
            object.__setattr__(self, name, MappingProxyType(value))
        if self.classification not in {"standard", "experimental"}:
            raise ValueError("classification must be standard or experimental")
        if self.production_qualified and (
            self.classification != "standard" or not self.passed
        ):
            raise ValueError("production_qualified requires a passing standard audit")

    @property
    def passed(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "classification": self.classification,
            "production_qualified": self.production_qualified,
            "model_ids": to_plain_mapping(self.model_ids),
            "provenance": to_plain_mapping(self.provenance),
            "inventory": to_plain_mapping(self.inventory),
            "conservation_max_abs_residual": to_plain_mapping(
                self.conservation_max_abs_residual
            ),
            "simulation": to_plain_mapping(self.simulation),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class AuditIssue:
    level: str
    code: str
    message: str
    quantity: str = ""

    def __post_init__(self) -> None:
        level = str(self.level).upper()
        if level not in {"ERROR", "WARNING"}:
            raise ValueError(f"Unsupported audit issue level {self.level!r}")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "code", str(self.code).strip())
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "quantity", str(self.quantity))


@dataclass(frozen=True, slots=True)
class AuditReport:
    issues: tuple[AuditIssue, ...] = ()
    conservation_max_abs_residual: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        maxima = {
            str(name): float(value)
            for name, value in dict(self.conservation_max_abs_residual or {}).items()
        }
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(
            self, "conservation_max_abs_residual", MappingProxyType(maxima)
        )

    @property
    def passed(self) -> bool:
        return not any(issue.level == "ERROR" for issue in self.issues)

    @property
    def max_conservation_residual(self) -> float | None:
        if not self.conservation_max_abs_residual:
            return None
        return max(self.conservation_max_abs_residual.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_conservation_residual": self.max_conservation_residual,
            "conservation_max_abs_residual": dict(self.conservation_max_abs_residual),
            "issues": [asdict(issue) for issue in self.issues],
        }


def _nonfinite_count(values: np.ndarray) -> int:
    return int(values.size - np.count_nonzero(np.isfinite(values)))


def _series_or_issue(
    result: SimulationResult,
    name: str,
    issues: list[AuditIssue],
    *,
    code: str,
) -> np.ndarray | None:
    try:
        return result.series(name)
    except KeyError:
        issues.append(
            AuditIssue(
                "ERROR", code, f"Required result series {name!r} is missing.", name
            )
        )
        return None


def _validate_tolerances(
    conservation_tolerances: Mapping[str, float] | None,
    default_conservation_tolerance: float | None,
) -> tuple[dict[str, float], float | None]:
    tolerances = {
        str(name): float(value)
        for name, value in dict(conservation_tolerances or {}).items()
    }
    if any(value < 0.0 or not np.isfinite(value) for value in tolerances.values()):
        raise ValueError("conservation tolerances must be finite and non-negative")
    default = (
        None
        if default_conservation_tolerance is None
        else float(default_conservation_tolerance)
    )
    if default is not None and (default < 0.0 or not np.isfinite(default)):
        raise ValueError(
            "default conservation tolerance must be finite and non-negative"
        )
    return tolerances, default


def audit_result(
    result: SimulationResult,
    *,
    conservation_residuals: Mapping[str, Any] | None = None,
    conservation_tolerances: Mapping[str, float] | None = None,
    default_conservation_tolerance: float | None = None,
    required_series: Iterable[str] = (),
    nonnegative_series: Iterable[str] = (),
    negative_tolerance: float = 0.0,
) -> AuditReport:
    """Audit a result without modifying it or performing I/O.

    Conservation residuals are explicit inputs because their definitions are
    model-specific.  They may be scalars or arrays with one value per result
    time.  Tolerances are applied to their maximum absolute value.
    """

    if not isinstance(result, SimulationResult):
        raise TypeError("result must be a SimulationResult")
    negative_tolerance = float(negative_tolerance)
    if negative_tolerance < 0.0 or not np.isfinite(negative_tolerance):
        raise ValueError("negative_tolerance must be finite and non-negative")
    tolerances, default_tolerance = _validate_tolerances(
        conservation_tolerances,
        default_conservation_tolerance,
    )

    issues: list[AuditIssue] = []
    if not result.status.success:
        issues.append(
            AuditIssue(
                "ERROR",
                "SIMULATION_NOT_SUCCESSFUL",
                f"Simulation status is {result.status.code!r}: {result.status.message}",
                "status",
            )
        )
    if result.n_times == 0:
        issues.append(
            AuditIssue(
                "WARNING",
                "EMPTY_RESULT",
                "Simulation result contains no time points.",
                "time_s",
            )
        )

    state_nonfinite = _nonfinite_count(result.state)
    if state_nonfinite:
        issues.append(
            AuditIssue(
                "ERROR",
                "STATE_NONFINITE",
                f"State array contains {state_nonfinite} non-finite value(s).",
                "state",
            )
        )
    for name, values in result.observables.items():
        count = _nonfinite_count(values)
        if count:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "OBSERVABLE_NONFINITE",
                    f"Observable {name!r} contains {count} non-finite value(s).",
                    name,
                )
            )

    for name in dict.fromkeys(str(item) for item in required_series):
        _series_or_issue(result, name, issues, code="REQUIRED_SERIES_MISSING")

    for name in dict.fromkeys(str(item) for item in nonnegative_series):
        values = _series_or_issue(
            result, name, issues, code="NONNEGATIVE_SERIES_MISSING"
        )
        if values is None or values.size == 0:
            continue
        finite = values[np.isfinite(values)]
        if finite.size and float(np.min(finite)) < -negative_tolerance:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "NEGATIVE_SERIES_VALUE",
                    f"Series {name!r} has minimum {float(np.min(finite))}, below allowed {-negative_tolerance}.",
                    name,
                )
            )

    maxima: dict[str, float] = {}
    for raw_name, raw_values in dict(conservation_residuals or {}).items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("conservation residual names must not be empty")
        try:
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"conservation residual {name!r} must be numeric") from exc
        if values.ndim == 1 and values.shape != (result.n_times,):
            issues.append(
                AuditIssue(
                    "ERROR",
                    "CONSERVATION_RESIDUAL_SHAPE",
                    f"Conservation residual {name!r} must be scalar or have shape ({result.n_times},), got {values.shape}.",
                    name,
                )
            )
            continue
        if values.ndim not in {0, 1}:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "CONSERVATION_RESIDUAL_SHAPE",
                    f"Conservation residual {name!r} must be scalar or one-dimensional.",
                    name,
                )
            )
            continue
        if not np.all(np.isfinite(values)):
            issues.append(
                AuditIssue(
                    "ERROR",
                    "CONSERVATION_RESIDUAL_NONFINITE",
                    f"Conservation residual {name!r} contains non-finite values.",
                    name,
                )
            )
            continue
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        maxima[name] = max_abs
        tolerance = tolerances.get(name, default_tolerance)
        if tolerance is not None and max_abs > tolerance:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "CONSERVATION_RESIDUAL_EXCEEDED",
                    f"Conservation residual {name!r} maximum {max_abs} exceeds tolerance {tolerance}.",
                    name,
                )
            )

    return AuditReport(issues=tuple(issues), conservation_max_abs_residual=maxima)


def _normalized_balance(delta: float, before: float) -> float:
    return abs(float(delta)) / max(abs(float(before)), 1.0)


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def collect_file_provenance(case: Any, chemistry_data: Any) -> Mapping[str, Any]:
    """Return deterministic SHA-256 provenance for every runtime input file.

    Paths declared by included YAML files have already been resolved relative to
    their declaring file by the v3 loader.  Chemistry input files are recorded by
    the strict chemistry reader while it follows its manifest, so rate tables and
    cross-section curves are included without reparsing configuration documents.
    """

    from plasma_global.input.schema import CaseSpec

    if not isinstance(case, CaseSpec):
        raise TypeError("case must be a schema-v3 CaseSpec")

    roles: dict[Path, set[str]] = {}

    def add(path: Path | str | None, role: str) -> None:
        if path is None:
            return
        resolved = Path(path).resolve()
        roles.setdefault(resolved, set()).add(role)

    add(case.source_path, "case")
    for path in case.included_files:
        add(path, "case_include")
    for path in chemistry_data.source_files:
        add(path, "chemistry_input")

    electron_model = case.models.electrons
    add(getattr(electron_model, "file", None), "electron_kinetics_table")
    electron_density = case.models.electron_density
    add(getattr(electron_density, "file", None), "electron_density_profile")
    for port in case.reactor.power_ports:
        add(getattr(port.model, "file", None), f"power_port:{port.port_id}")
    for step in case.recipe.steps:
        for port_id, command in step.commands.power_ports.items():
            add(
                getattr(command, "file", None),
                f"recipe_power_override:{step.step_id}:{port_id}",
            )

    files: dict[str, Any] = {}
    for path in sorted(roles, key=lambda item: str(item).casefold()):
        digest, size = _file_sha256(path)
        files[str(path)] = {
            "sha256": digest,
            "size_bytes": size,
            "roles": sorted(roles[path]),
        }
    return MappingProxyType(
        {
            "algorithm": "sha256",
            "files": MappingProxyType(files),
        }
    )


def _case_classification(case: Any, chemistry_data: Any) -> str:
    """Classify the deliberately small, validated standard-model envelope."""

    heavy_ions = tuple(
        item
        for item in chemistry_data.species
        if item.phase == "gas" and item.id != "e" and item.charge != 0
    )
    positive_ions = tuple(item for item in heavy_ions if item.charge > 0)
    has_negative_ion = any(item.charge < 0 for item in heavy_ions)

    def deterministic_standard_wall(surface: Any) -> bool:
        if surface.wall_transport.kind == "off":
            return True
        if (
            surface.wall_transport.kind != "bohm"
            or len(positive_ions) != 1
            or positive_ions[0].charge != 1
            or has_negative_ion
            or surface.ion_impact_energy_eV != 0.0
        ):
            return False
        ion_id = positive_ions[0].id
        branches = [
            reaction
            for reaction in chemistry_data.boundary_reactions
            if reaction.reactants == {ion_id: 1.0}
            and bool(reaction.products)
            and (not reaction.zones or surface.zone_id in reaction.zones)
            and (not reaction.surfaces or surface.surface_id in reaction.surfaces)
        ]
        return len(branches) == 1

    deterministic_standard_walls = all(
        deterministic_standard_wall(surface) for surface in case.reactor.surfaces
    )
    electron_reactions_use_cross_sections = all(
        "e" not in reaction.reactants
        or (
            reaction.rate_model is not None
            and chemistry_data.rate_models[str(reaction.rate_model)].kind
            == "electron_impact"
        )
        for reaction in chemistry_data.gas_reactions
    )
    standard = (
        case.models.electrons.kind == "maxwellian"
        and case.models.electron_closure.kind == "electron_energy"
        and case.models.electron_density.kind == "quasineutral"
        and case.models.gas_energy.kind == "fixed"
        and all(
            port.model.kind == "prescribed_power" for port in case.reactor.power_ports
        )
        and case.models.surface_kinetics is None
        and not chemistry_data.surface_reactions
        and deterministic_standard_walls
        and electron_reactions_use_cross_sections
        and not has_negative_ion
        and len(positive_ions) == 1
        and positive_ions[0].charge == 1
        and not any(
            str(model.kind).startswith("experimental.")
            for model in chemistry_data.rate_models.values()
        )
        and not chemistry_data.experimental
        and case.experimental is None
    )
    return "standard" if standard else "experimental"


def runtime_diagnostic_maxima(
    compiled: Any,
    result: SimulationResult,
    *,
    include_detailed_ledgers: bool = True,
) -> dict[str, float]:
    """Evaluate charge closure and optional particle/energy ledger closure."""

    model = compiled.model
    maxima = {"charge_closure_normalized": 0.0}
    # Quasineutrality is the standard electron-density closure.  Its charge
    # residual is zero by construction, so a normal simulation must not run a
    # second full chemistry/wall/power traversal merely to rediscover that
    # identity at every saved point.  Prescribed electrons remain experimental
    # and take the explicit evaluation path below.
    uses_prescribed_electrons = model.electron_density_provider is not None or any(
        segment.prescribed_electron_density_m3_by_zone for segment in model.segments
    )
    if not include_detailed_ledgers and not uses_prescribed_electrons:
        return maxima
    if include_detailed_ledgers:
        maxima["particle_ledger_normalized"] = 0.0
    if include_detailed_ledgers and model.layout.evolves_electron_energy:
        maxima["electron_energy_ledger_normalized"] = 0.0
    if include_detailed_ledgers and model.layout.evolves_heavy_energy:
        maxima["heavy_energy_ledger_normalized"] = 0.0

    species_index = {name: index for index, name in enumerate(model.species_ids)}
    boundary_products = {
        reaction.id: reaction.products
        for reaction in compiled.chemistry_data.boundary_reactions
    }
    surface_sources: dict[str, np.ndarray] = {}
    if include_detailed_ledgers and model.surface_model is not None:
        surface_model = model.surface_model
        for kernel in surface_model._kernels:
            surface = surface_model.surfaces[kernel.surface_index]
            source = np.zeros(len(model.species_ids), dtype=float)
            area_over_volume = (
                surface.area_m2
                / surface_model.zone_volumes_m3[
                    surface_model._zone_index[surface.zone_id]
                ]
            )
            for index, delta in kernel.gas_delta:
                source[index] += area_over_volume * delta
            surface_sources[f"{kernel.reaction.id}@{surface.surface_id}"] = source

    segment_index = 0
    for time_s, state in zip(result.time_s, result.state):
        while (
            segment_index + 1 < len(model.segments)
            and time_s > model.segments[segment_index].end_s
        ):
            segment_index += 1
        evaluation = model.evaluate(
            float(time_s),
            state,
            model.segments[segment_index],
            collect_ledger=include_detailed_ledgers,
        )
        for zone in model.zones:
            zone_id = zone.zone_id
            density = state[model.layout.density_slices[zone_id]]
            electrons = evaluation.electron_states[zone_id]
            charge_scale = max(
                electrons.density_m3
                + float(np.dot(np.abs(model.charges), np.maximum(density, 0.0))),
                1.0,
            )
            charge_residual = abs(evaluation.charge_residual_m3_by_zone[zone_id])
            maxima["charge_closure_normalized"] = max(
                maxima["charge_closure_normalized"], charge_residual / charge_scale
            )

            if not include_detailed_ledgers:
                continue
            ledger = evaluation.ledger_by_zone[zone_id]
            rates = np.asarray(
                [ledger.reaction_rates_m3_s[name] for name in model.reaction_ids]
            )
            reaction_source = model.stoichiometry.T @ rates
            transport_source = np.asarray(
                [
                    ledger.transport_species_source_m3_s[name]
                    for name in model.species_ids
                ]
            )
            wall_source = np.zeros(len(model.species_ids), dtype=float)
            for record in ledger.wall_fluxes:
                wall_source[species_index[record.incident_species]] -= (
                    record.incident_rate_m3_s
                )
                for reaction_id, branch_rate in record.branch_rates_m3_s.items():
                    for product, yield_per_ion in boundary_products[
                        reaction_id
                    ].items():
                        wall_source[species_index[product]] += (
                            yield_per_ion * branch_rate
                        )
            surface_source = np.zeros(len(model.species_ids), dtype=float)
            for rate_id, rate_m2_s in ledger.surface_rates_m2_s.items():
                surface_source += surface_sources[rate_id] * rate_m2_s
            density_rhs = evaluation.derivative[model.layout.density_slices[zone_id]]
            reconstructed = (
                reaction_source + transport_source + wall_source + surface_source
            )
            particle_scale = np.maximum.reduce(
                (
                    np.abs(density_rhs),
                    np.abs(reaction_source)
                    + np.abs(transport_source)
                    + np.abs(wall_source)
                    + np.abs(surface_source),
                    np.full(len(model.species_ids), 1.0e-300),
                )
            )
            maxima["particle_ledger_normalized"] = max(
                maxima["particle_ledger_normalized"],
                float(np.max(np.abs(density_rhs - reconstructed) / particle_scale)),
            )
            if model.layout.evolves_electron_energy:
                terms = (
                    ledger.absorbed_power_J_m3_s,
                    -ledger.reaction_energy_loss_J_m3_s,
                    -ledger.wall_energy_loss_J_m3_s,
                    -ledger.elastic_heating_J_m3_s,
                    ledger.transport_electron_energy_J_m3_s,
                )
                rhs = float(
                    evaluation.derivative[model.layout.electron_energy_indices[zone_id]]
                )
                scale = max(abs(rhs), sum(abs(value) for value in terms), 1.0e-300)
                maxima["electron_energy_ledger_normalized"] = max(
                    maxima["electron_energy_ledger_normalized"],
                    abs(rhs - sum(terms)) / scale,
                )
            if model.layout.evolves_heavy_energy:
                terms = (
                    ledger.gas_power_J_m3_s,
                    ledger.gas_reaction_heating_J_m3_s,
                    ledger.surface_reaction_heating_J_m3_s,
                    ledger.wall_heavy_energy_exchange_J_m3_s,
                    ledger.elastic_heating_J_m3_s,
                    ledger.transport_heavy_energy_J_m3_s,
                )
                rhs = float(
                    evaluation.derivative[model.layout.heavy_energy_indices[zone_id]]
                )
                scale = max(abs(rhs), sum(abs(value) for value in terms), 1.0e-300)
                maxima["heavy_energy_ledger_normalized"] = max(
                    maxima["heavy_energy_ledger_normalized"],
                    abs(rhs - sum(terms)) / scale,
                )
    return maxima


def audit_case(case: Any) -> CaseAuditReport:
    """Compile, integrate, and audit one case without writing output files."""

    from plasma_global.build import _simulate_compiled_case, compile_case
    from plasma_global.input.schema import CaseSpec

    if not isinstance(case, CaseSpec):
        raise TypeError("case must be a schema-v3 CaseSpec")
    compiled = compile_case(case)
    result = _simulate_compiled_case(compiled, detailed_audit=True)
    data = compiled.chemistry_data
    species = {item.id: item for item in data.species}

    element_max = 0.0
    site_max = 0.0
    charge_max = 0.0
    for family, reactions in (
        ("gas", data.gas_reactions),
        ("boundary", data.boundary_reactions),
        ("surface", data.surface_reactions),
    ):
        for reaction in reactions:
            elements = {
                element
                for species_id in (*reaction.reactants, *reaction.products)
                for element in species[species_id].elements
            }
            for element in elements:
                before = sum(
                    order * species[species_id].elements.get(element, 0.0)
                    for species_id, order in reaction.reactants.items()
                )
                after = sum(
                    order * species[species_id].elements.get(element, 0.0)
                    for species_id, order in reaction.products.items()
                )
                residual = _normalized_balance(after - before, before)
                if element == "site":
                    site_max = max(site_max, residual)
                else:
                    element_max = max(element_max, residual)
            if family != "boundary":
                before_charge = sum(
                    order * species[species_id].charge
                    for species_id, order in reaction.reactants.items()
                )
                after_charge = sum(
                    order * species[species_id].charge
                    for species_id, order in reaction.products.items()
                )
                charge_max = max(
                    charge_max,
                    _normalized_balance(after_charge - before_charge, before_charge),
                )

    model_ids: dict[str, Any] = {
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
    experimental_ids: list[str] = []
    for value in (
        case.models.electrons.kind,
        case.models.electron_density.kind,
        *(item.model.kind for item in case.reactor.power_ports),
        *(item.kind for item in data.rate_models.values()),
    ):
        if str(value).startswith("experimental."):
            experimental_ids.append(str(value))
    if case.experimental is not None:
        for field_name in ("film", "wall_inventory", "extensions"):
            if getattr(case.experimental, field_name) is not None:
                experimental_ids.append(f"experimental.{field_name}")

    classification = _case_classification(case, data)

    issues = [
        AuditIssue(
            "WARNING",
            "EXPERIMENTAL_MODEL",
            (
                f"{identifier} is an explicit experimental model; only "
                "finite behavior, conservation, and qualitative trends are claimed."
            ),
            identifier,
        )
        for identifier in dict.fromkeys(experimental_ids)
    ]
    result_report = audit_result(result)
    issues.extend(result_report.issues)
    compiled_provenance = to_plain_mapping(result.metadata)["provenance"]
    missing_momentum_targets = tuple(
        str(target)
        for target in compiled_provenance.get(
            "missing_momentum_cross_section_targets", ()
        )
    )
    issues.extend(
        AuditIssue(
            "WARNING",
            "MISSING_MOMENTUM_CROSS_SECTION",
            (
                f"Neutral target {target!r} has no momentum-transfer cross section; "
                "electron elastic gas heating is omitted for this target."
            ),
            target,
        )
        for target in dict.fromkeys(missing_momentum_targets)
    )
    provenance = dict(compiled_provenance)
    provenance["input_files"] = collect_file_provenance(case, data)
    dynamic = dict(provenance["runtime_diagnostics"]["conservation_max_abs_residual"])
    static = {
        "normalized_element": element_max,
        "normalized_charge": charge_max,
        "normalized_site": site_max,
    }
    for name, value in static.items():
        if value > 1.0e-12:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "STATIC_REACTION_BALANCE_EXCEEDED",
                    f"Static reaction balance {name!r} is {value}, above 1e-12.",
                    name,
                )
            )
    for name in (
        "particle_ledger_normalized",
        "electron_energy_ledger_normalized",
        "heavy_energy_ledger_normalized",
    ):
        if dynamic.get(name, 0.0) > 1.0e-12:
            issues.append(
                AuditIssue(
                    "ERROR",
                    "DYNAMIC_LEDGER_RESIDUAL_EXCEEDED",
                    f"Dynamic ledger residual {name!r} is {dynamic[name]}, above 1e-12.",
                    name,
                )
            )
    if (
        case.models.electron_density.kind == "quasineutral"
        and dynamic["charge_closure_normalized"] > 1.0e-12
    ):
        issues.append(
            AuditIssue(
                "ERROR",
                "CHARGE_CLOSURE_RESIDUAL_EXCEEDED",
                "Quasineutral charge-closure residual exceeds 1e-12.",
                "charge_closure_normalized",
            )
        )
    inventory = {
        "zones": len(case.reactor.zones),
        "species": len(data.species),
        "gas_reactions": len(data.gas_reactions),
        "boundary_reactions": len(data.boundary_reactions),
        "surface_reactions": len(data.surface_reactions),
        "recipe_segments": len(compiled.segments),
        "state_variables": compiled.model.layout.size,
    }
    return CaseAuditReport(
        issues=tuple(issues),
        model_ids=model_ids,
        provenance=provenance,
        inventory=inventory,
        conservation_max_abs_residual={**static, **dynamic},
        classification=classification,
        production_qualified=(classification == "standard" and not issues),
        simulation={
            "status": {
                "success": result.status.success,
                "code": result.status.code,
                "message": result.status.message,
            },
            "time_count": result.n_times,
            "start_s": float(result.time_s[0]) if result.n_times else None,
            "end_s": float(result.time_s[-1]) if result.n_times else None,
            "solver": to_plain_mapping(result.solver_stats),
        },
    )


__all__ = [
    "AuditIssue",
    "AuditReport",
    "CaseAuditReport",
    "audit_case",
    "audit_result",
    "collect_file_provenance",
    "runtime_diagnostic_maxima",
]
