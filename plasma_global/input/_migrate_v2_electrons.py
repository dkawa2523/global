"""Translate schema-v2 electron and gas-energy model selections to schema v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_common import resolved_path

_LEGACY_APPROXIMATE_FIELD_GRID = (0.2, 2500.0, 48)


def _table_electron_model(
    run: Any,
    resolved: Any,
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    table = run.swarm.table
    if not table.file:
        raise MigrationError("v2 swarm table model has no table file")
    requested_lookup = str(table.lookup or run.swarm.closure or "auto").lower()
    closure_kind = (
        "local_field" if requested_lookup == "local_field" else "electron_energy"
    )
    if table.electron_energy_mode:
        unused.add("swarm.table.electron_energy_mode")
        unused.add("swarm.table.energy_relaxation_time_s")
    if str(table.bounds_policy).lower() != "error":
        warnings.append(
            "swarm.table.bounds_policy was changed to 'error' to prevent "
            "silent extrapolation/clipping"
        )
    return (
        {
            "kind": "table",
            "file": resolved_path(table.file, Path(resolved.chemistry_dir)),
            "lookup": (
                "local_field" if closure_kind == "local_field" else "mean_energy"
            ),
            "bounds_policy": "error",
        },
        closure_kind,
    )


def _warn_approximate_closure_migration(run: Any, warnings: list[str]) -> None:
    """Record when v2 requested a closure unavailable to the prepared model."""

    requested_closure = str(run.swarm.closure or "auto").lower()
    if requested_closure not in {"auto", "local_field"}:
        warnings.append(
            "boltzmann_2term closure was changed to local_field because the "
            "v3 approximate model is compile-time prepared and has no "
            "electron-energy RHS coupling"
        )


def _approximate_reduced_field_grid(
    cfg: Any,
    warnings: list[str],
) -> dict[str, float | int] | None:
    """Preserve an explicit v2 field grid and safely replace legacy defaults."""

    requested_min = float(cfg.reduced_field_grid_Td.min)
    requested_max = float(cfg.reduced_field_grid_Td.max)
    requested_count = int(cfg.reduced_field_grid_Td.n)
    uses_legacy_defaults = (
        not cfg.reduced_field_grid_was_explicit
        and (
            requested_min,
            requested_max,
            requested_count,
        )
        == _LEGACY_APPROXIMATE_FIELD_GRID
    )
    if uses_legacy_defaults:
        warnings.append(
            "boltzmann_2term used the legacy default reduced-field grid; migration "
            "selected the current v3 defaults because the strict approximate "
            "closure validates a power-balance root at every requested point"
        )
        return None
    return {
        "min_Td": requested_min,
        "max_Td": requested_max,
        "n": requested_count,
    }


def _approximate_two_term_electron_model(
    run: Any, warnings: list[str]
) -> tuple[dict[str, Any], str]:
    cfg = run.swarm.boltzmann_2term
    _warn_approximate_closure_migration(run, warnings)
    reduced_field_grid = _approximate_reduced_field_grid(cfg, warnings)
    electron_model: dict[str, Any] = {
        "kind": "experimental.approximate_two_term",
        "cache": {"max_entries": int(run.swarm.cache.max_entries)},
        "energy_grid": {
            "min_eV": float(cfg.energy_grid.min_eV),
            "max_eV": float(cfg.energy_grid.max_eV),
            "n": int(cfg.energy_grid.n),
        },
        "max_shape_iterations": int(cfg.max_shape_iterations),
    }
    if run.swarm.mixture_key_species:
        warnings.append(
            "swarm.mixture_key_species was removed; approximate tables are keyed "
            "by the exact fractions of every collision target"
        )
    if int(run.swarm.cache.fraction_decimals) != 3:
        warnings.append(
            "swarm.cache.fraction_decimals was removed because approximate table "
            "identity no longer uses rounded fractions"
        )
    if reduced_field_grid is not None:
        electron_model["reduced_field_grid"] = reduced_field_grid
    return (
        electron_model,
        "local_field",
    )


def _electron_kinetics_model(
    run: Any,
    resolved: Any,
    unused: set[str],
    warnings: list[str],
) -> tuple[dict[str, Any], str]:
    backend = str(run.physics.eedf_backend).strip().lower()
    if backend == "maxwell":
        return {"kind": "maxwellian"}, "electron_energy"
    model_name = str(run.swarm.model_name).lower()
    if backend == "swarm" and model_name == "table":
        return _table_electron_model(run, resolved, unused, warnings)
    if backend == "swarm" and model_name == "boltzmann_2term":
        return _approximate_two_term_electron_model(run, warnings)
    raise MigrationError(
        f"unsupported v2 EEDF selection {backend!r}/{run.swarm.model_name!r}"
    )


def _electron_density_model(
    run: Any, resolved: Any, warnings: list[str]
) -> dict[str, Any]:
    closure = str(run.physics.electron_density_closure).strip().lower()
    if closure == "quasi_neutral":
        return {"kind": "quasineutral"}
    if closure != "prescribed_profile":
        raise MigrationError(f"unsupported v2 electron density closure {closure!r}")

    cfg = run.swarm.prescribed_electron_profile
    if cfg.file:
        profile_file = resolved_path(cfg.file, Path(resolved.base_dir))
    elif cfg.file_key and resolved.external_inputs.get(str(cfg.file_key)):
        profile_file = Path(str(resolved.external_inputs[str(cfg.file_key)])).resolve(
            strict=False
        )
    else:
        raise MigrationError("v2 prescribed electron profile has no input file")
    if str(cfg.hold).lower() != "error":
        warnings.append(
            "prescribed electron profile hold was changed to 'error' to "
            "prevent silent endpoint extension"
        )
    return {
        "kind": "experimental.prescribed_profile",
        "file": profile_file,
        "zone_columns": dict(cfg.zone_columns),
        "interpolation": str(cfg.interpolation).lower(),
        "hold": "error",
    }


def electron_models(
    loaded: Any, unused: set[str], warnings: list[str]
) -> dict[str, Any]:
    """Translate the complete electron, gas-energy, and surface model selection."""

    run = loaded.run_config
    electrons, closure_kind = _electron_kinetics_model(
        run,
        loaded.resolved_paths,
        unused,
        warnings,
    )
    density = _electron_density_model(run, loaded.resolved_paths, warnings)
    return {
        "electrons": electrons,
        "electron_closure": {"kind": closure_kind},
        "electron_density": density,
        "gas_energy": {
            "kind": "evolved" if run.physics.enable_gas_temperature else "fixed"
        },
        "surface_kinetics": (
            {}
            if run.physics.enable_surface_coverages
            and loaded.chamber.surfaces
            and loaded.mechanism.surface_species
            else None
        ),
    }
