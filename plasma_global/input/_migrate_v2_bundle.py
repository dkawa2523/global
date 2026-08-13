"""Publication helpers for schema-v2 migration bundles.

The translation itself lives in :mod:`plasma_global.input.migrate_v2`.  This
module owns filesystem publication: rendering YAML, staging runtime assets,
converting chemistry, and committing the completed bundle atomically.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from plasma_global.errors import MigrationError
from plasma_global.input.schema import CaseSpec

if TYPE_CHECKING:
    from plasma_global.input.migrate_v2 import MigrationResult


def _yaml_value(value: Any, *, relative_to: Path | None = None) -> Any:
    if isinstance(value, Path):
        if relative_to is not None and value.is_absolute():
            try:
                return Path(os.path.relpath(value, relative_to)).as_posix()
            except ValueError:
                pass
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _yaml_value(item, relative_to=relative_to)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_yaml_value(item, relative_to=relative_to) for item in value]
    return value


def write_v3_case(result: MigrationResult, destination: str | Path) -> Path:
    """Write migrated v3 YAML without copying referenced assets."""

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        _yaml_value(result.data, relative_to=target.parent),
        sort_keys=False,
        allow_unicode=True,
    )
    target.write_text(text, encoding="utf-8")
    return target


def _migration_report_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}.migration.yaml")


def write_migration_report(result: MigrationResult, destination: str | Path) -> Path:
    """Persist the intentionally dropped keys and migration warnings."""

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(result.report.source_path),
        "unused_keys": list(result.report.unused_keys),
        "warnings": list(result.report.warnings),
    }
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


class _AssetStager:
    """Copy or convert each unique runtime asset into a publication bundle."""

    def __init__(self, staging_directory: Path, published_directory: Path) -> None:
        self.staging_directory = staging_directory
        self.published_directory = published_directory
        self.staged: dict[tuple[Path, str], Path] = {}
        self.warnings: list[str] = []

    def stage(self, value: Any, label: str, *, electron_table: bool = False) -> Path:
        source = Path(value).resolve()
        if not source.is_file():
            raise MigrationError(f"{label} does not exist: {source}")
        conversion = "electron_table" if electron_table else "copy"
        key = (source, conversion)
        if key in self.staged:
            return self.staged[key]
        suffix = source.suffix or ".dat"
        filename = f"{len(self.staged):02d}_{label.replace('.', '_')}{suffix}"
        temporary_path = self.staging_directory / filename
        published_path = (self.published_directory / filename).resolve()
        self.staging_directory.mkdir(parents=True, exist_ok=True)
        try:
            if electron_table:
                from tools.importers.rate_table import convert_v2_rate_table_h5

                _, dropped = convert_v2_rate_table_h5(source, temporary_path)
                if dropped:
                    self.warnings.append(
                        f"{label}: dropped unused legacy HDF5 datasets: "
                        f"{', '.join(dropped)}"
                    )
            else:
                shutil.copy2(source, temporary_path)
        except (OSError, ValueError) as exc:
            raise MigrationError(
                f"cannot migrate {label} from {source}: {exc}"
            ) from exc
        self.staged[key] = published_path
        return published_path


def _stage_electron_assets(data: dict[str, Any], stager: _AssetStager) -> None:
    models = data["models"]
    electrons = models["electrons"]
    if electrons["kind"] == "table":
        electrons["file"] = stager.stage(
            electrons["file"], "models_electrons", electron_table=True
        )
    electron_density = models["electron_density"]
    if electron_density["kind"] == "experimental.prescribed_profile":
        electron_density["file"] = stager.stage(
            electron_density["file"], "models_electron_density"
        )


def _stage_external_power_assets(data: dict[str, Any], stager: _AssetStager) -> None:
    for port_index, port in enumerate(data["reactor"]["power_ports"]):
        model = port["model"]
        if model["kind"] == "external_table":
            model["file"] = stager.stage(
                model["file"], f"power_model_{port_index}_{port['port_id']}"
            )

    for step_index, step in enumerate(data["recipe"]["steps"]):
        commands = step["commands"]["power_ports"]
        for port_id, command in commands.items():
            if command["kind"] == "external_table" and "file" in command:
                command["file"] = stager.stage(
                    command["file"],
                    f"power_command_{step_index}_{port_id}",
                )


def stage_case_assets(
    case: CaseSpec,
    staging_directory: Path,
    published_directory: Path,
) -> tuple[CaseSpec, tuple[str, ...]]:
    """Bundle runtime file dependencies and replace their paths in a case."""

    data = case.model_dump(mode="python", exclude_none=True)
    stager = _AssetStager(staging_directory, published_directory)
    _stage_electron_assets(data, stager)
    _stage_external_power_assets(data, stager)
    try:
        migrated = CaseSpec.model_validate(data)
    except Exception as exc:
        raise MigrationError(f"bundled v3 case failed validation: {exc}") from exc
    return migrated, tuple(stager.warnings)


def migrate_v2_to_yaml(
    source: str | Path,
    destination: str | Path,
    *,
    boundary_products: Mapping[str, str] | None = None,
    cross_section_segments: Mapping[str, int] | None = None,
) -> MigrationResult:
    """Write a runnable v3 case, canonical chemistry, and migration report.

    Chemistry conversion is intentionally part of this one-way boundary. A
    wall product or concatenated cross-section axis that cannot be selected
    uniquely is an error; publication never guesses scientific data.
    """

    from plasma_global.chemistry.compile import compile_chemistry
    from plasma_global.chemistry.data import load_chemistry
    from plasma_global.input.migrate_v2 import (
        MigrationReport,
        MigrationResult,
        migrate_v2,
    )
    from tools.importers.chemistry_v2 import convert_v2_chemistry

    target = Path(destination).resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    chemistry_target = target.parent / f"{target.stem}_chemistry"
    assets_target = target.parent / f"{target.stem}_assets"
    _require_available_targets(target, chemistry_target, assets_target)

    result = migrate_v2(source)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{target.stem}-migration-", dir=target.parent
        ) as temporary:
            temporary_root = Path(temporary)
            temporary_chemistry = temporary_root / "chemistry"
            temporary_assets = temporary_root / "assets"
            bundled_case, asset_warnings = stage_case_assets(
                result.case,
                temporary_assets,
                assets_target,
            )
            chemistry_warnings: list[str] = []
            canonical_manifest = convert_v2_chemistry(
                result.case.chemistry.manifest,
                temporary_chemistry,
                boundary_products=boundary_products,
                cross_section_segments=cross_section_segments,
                migration_warnings=chemistry_warnings,
            )
            # Validate the complete scientific bundle before publishing it.
            compile_chemistry(load_chemistry(canonical_manifest))
            shutil.move(str(temporary_chemistry), str(chemistry_target))
            if temporary_assets.exists():
                shutil.move(str(temporary_assets), str(assets_target))

        published_manifest = (chemistry_target / "chemistry.yaml").resolve()
        chemistry = bundled_case.chemistry.model_copy(
            update={"manifest": published_manifest}
        )
        report = MigrationReport(
            source_path=result.report.source_path,
            unused_keys=result.report.unused_keys,
            warnings=(
                *result.report.warnings,
                *chemistry_warnings,
                *asset_warnings,
            ),
        )
        migrated = MigrationResult(
            case=bundled_case.model_copy(update={"chemistry": chemistry}),
            report=report,
        )
        write_v3_case(migrated, target)
        write_migration_report(migrated, _migration_report_path(target))
    except Exception:
        _remove_created_targets(target, chemistry_target, assets_target)
        raise
    return migrated


def _require_available_targets(
    target: Path, chemistry_target: Path, assets_target: Path
) -> None:
    if target.exists():
        raise MigrationError(f"migration output already exists: {target}")
    if chemistry_target.exists():
        raise MigrationError(
            f"migration chemistry output already exists: {chemistry_target}"
        )
    if assets_target.exists():
        raise MigrationError(f"migration asset output already exists: {assets_target}")


def _remove_created_targets(
    target: Path, chemistry_target: Path, assets_target: Path
) -> None:
    if chemistry_target.exists():
        shutil.rmtree(chemistry_target)
    if assets_target.exists():
        shutil.rmtree(assets_target)
    target.unlink(missing_ok=True)
    _migration_report_path(target).unlink(missing_ok=True)
