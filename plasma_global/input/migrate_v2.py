"""One-way migration from split schema-v2 inputs to one schema-v3 case."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_common import float_or_none
from plasma_global.input._migrate_v2_electrons import electron_models
from plasma_global.input._migrate_v2_experimental import migrate_experimental
from plasma_global.input._migrate_v2_reactor import migrate_reactor
from plasma_global.input._migrate_v2_recipe import migrate_output, migrate_recipe_steps
from plasma_global.input.schema import CaseSpec


@dataclass(frozen=True, slots=True)
class MigrationReport:
    source_path: Path
    unused_keys: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.unused_keys and not self.warnings


@dataclass(frozen=True, slots=True)
class MigrationResult:
    case: CaseSpec
    report: MigrationReport

    @property
    def data(self) -> dict[str, Any]:
        """Return the migrated v3 mapping suitable for YAML serialization."""

        return self.case.model_dump(mode="python", exclude_none=True)


def _solver(
    run: Any,
    *,
    sample_interval_s: float,
    unused: set[str],
    warnings: list[str],
) -> dict[str, Any]:
    integrator = str(run.physics.integrator).strip().lower()
    if integrator != "scipy_bdf":
        raise MigrationError(f"unsupported v2 integrator {run.physics.integrator!r}")
    positivity = dict(run.numerics.positivity or {})
    unused.update(f"numerics.positivity.{key}" for key in positivity)
    unused.add("numerics.atol")
    warnings.append(
        "v2 scalar atol used physical state units and cannot be transferred to the "
        "dimensionless v3 state; solver.atol was set explicitly to 1e-14"
    )
    result: dict[str, Any] = {
        "method": "BDF",
        "rtol": float(run.numerics.rtol),
        "atol": 1.0e-14,
        "first_step_s": float_or_none(run.numerics.first_step),
        "max_step_s": float_or_none(run.numerics.max_step),
        "sample_interval_s": sample_interval_s,
    }
    return result


def migrate_v2(path: str | Path) -> MigrationResult:
    """Read a schema-v2 case at the migration boundary and convert it to v3."""

    from plasma_global.input._legacy_v2 import load_legacy_v2_case

    source = Path(path).resolve(strict=False)
    loaded = load_legacy_v2_case(source)
    run = loaded.run_config
    chamber = loaded.chamber
    recipe = loaded.recipe
    resolved = loaded.resolved_paths
    unused: set[str] = {
        "case.kind",
        "case.tags",
        "physics.mode",
        "physics.gas_model",
        "physics.wall_relaxation_s_inv",
        "runtime.export_effective_config",
        "runtime.export_resolved_paths",
        "outputs.formats.solution_h5",
        "outputs.formats.summary_yaml",
    }
    warnings: list[str] = []
    used_external: set[str] = set()
    gas_fraction = float(run.physics.gas_heating_fraction)
    if not 0.0 <= gas_fraction <= 1.0:
        raise MigrationError("v2 physics.gas_heating_fraction must be between 0 and 1")
    models = electron_models(loaded, unused, warnings)

    reactor, port_model_by_id = migrate_reactor(
        loaded=loaded,
        models=models,
        gas_fraction=gas_fraction,
        used_external=used_external,
        unused=unused,
        warnings=warnings,
    )

    recipe_steps = migrate_recipe_steps(
        recipe=recipe,
        resolved=resolved,
        port_model_by_id=port_model_by_id,
        used_external=used_external,
        unused=unused,
        warnings=warnings,
    )

    for key in resolved.external_inputs:
        if key not in used_external:
            unused.add(f"files.external_inputs.{key}")

    if not any(
        port["model"]["kind"] == "prescribed_power" for port in reactor["power_ports"]
    ):
        unused.add("physics.gas_heating_fraction")

    output, sample_interval_s = migrate_output(run, recipe_steps, unused, warnings)
    experimental = migrate_experimental(
        loaded=loaded,
        run=run,
        chamber=chamber,
        unused=unused,
        warnings=warnings,
    )
    data = {
        "schema_version": 3,
        "case": {
            "name": str(run.case.name),
            "description": str(run.case.description),
        },
        "chemistry": {"manifest": Path(resolved.chemistry_manifest)},
        "reactor": reactor,
        "recipe": {
            "recipe_id": str(recipe.recipe_id),
            "description": str(recipe.description),
            "start_time_s": float(recipe.steps[0].t_start_s),
            "steps": recipe_steps,
        },
        "models": models,
        "solver": _solver(
            run,
            sample_interval_s=sample_interval_s,
            unused=unused,
            warnings=warnings,
        ),
        "output": output,
    }
    if experimental:
        data["experimental"] = experimental
    try:
        case = CaseSpec.model_validate(data)
    except Exception as exc:
        raise MigrationError(f"generated v3 case failed validation: {exc}") from exc
    report = MigrationReport(
        source_path=source,
        unused_keys=tuple(sorted(unused)),
        warnings=tuple(warnings),
    )
    return MigrationResult(case=case, report=report)


def write_v3_case(result: MigrationResult, destination: str | Path) -> Path:
    """Write migrated v3 YAML without copying referenced assets."""

    from plasma_global.input._migrate_v2_bundle import write_v3_case as publish

    return publish(result, destination)


def write_migration_report(result: MigrationResult, destination: str | Path) -> Path:
    """Persist the intentionally dropped keys and migration warnings."""

    from plasma_global.input._migrate_v2_bundle import (
        write_migration_report as publish_report,
    )

    return publish_report(result, destination)


def migrate_v2_to_yaml(
    source: str | Path,
    destination: str | Path,
    *,
    boundary_products: Mapping[str, str] | None = None,
    cross_section_segments: Mapping[str, int] | None = None,
) -> MigrationResult:
    """Publish a runnable v3 case, canonical chemistry, assets, and report."""

    from plasma_global.input._migrate_v2_bundle import migrate_v2_to_yaml as publish

    return publish(
        source,
        destination,
        boundary_products=boundary_products,
        cross_section_segments=cross_section_segments,
    )


__all__ = [
    "MigrationError",
    "MigrationReport",
    "MigrationResult",
    "migrate_v2",
    "migrate_v2_to_yaml",
    "write_migration_report",
    "write_v3_case",
]
