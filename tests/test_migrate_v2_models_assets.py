from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plasma_global.errors import MigrationError
from plasma_global.input._legacy_v2 import load_legacy_v2_case
from plasma_global.input.load import load_case
from plasma_global.input.migrate_v2 import (
    MigrationReport,
    MigrationResult,
    _electron_models,
    _stage_case_assets,
    migrate_v2,
    write_v3_case,
)
from plasma_global.input.schema import CaseSpec

ROOT = Path(__file__).resolve().parents[1]
V2_CONFIGS = ROOT / "tests" / "fixtures" / "v2" / "configs"
V3_FIXTURE = ROOT / "tests" / "fixtures" / "v3_minimal" / "case.yaml"


def test_table_electron_migration_preserves_mapping_and_diagnostic_order() -> None:
    loaded = load_legacy_v2_case(V2_CONFIGS / "case_zdplaskin_example2.yaml")
    unused: set[str] = set()
    warnings: list[str] = []

    models = _electron_models(loaded, unused, warnings)

    assert models["electrons"] == {
        "kind": "table",
        "file": (
            ROOT
            / "tests"
            / "fixtures"
            / "v2"
            / "chemistry_zdplaskin_example2"
            / "tables"
            / "zdplaskin_example2_eovern_rates.h5"
        ).resolve(),
        "lookup": "local_field",
        "bounds_policy": "error",
    }
    assert models["electron_closure"] == {"kind": "local_field"}
    assert models["electron_density"] == {"kind": "quasineutral"}
    assert models["gas_energy"] == {"kind": "fixed"}
    assert models["surface_kinetics"] is None
    assert unused == {
        "swarm.table.electron_energy_mode",
        "swarm.table.energy_relaxation_time_s",
    }
    assert warnings == [
        "swarm.table.bounds_policy was changed to 'error' to prevent "
        "silent extrapolation/clipping"
    ]


def test_electron_migration_preserves_selection_errors() -> None:
    missing_table = load_legacy_v2_case(V2_CONFIGS / "case_zdplaskin_example2.yaml")
    missing_table.run_config.swarm.table.file = None
    with pytest.raises(MigrationError, match="swarm table model has no table file"):
        _electron_models(missing_table, set(), [])

    unsupported = load_legacy_v2_case(V2_CONFIGS / "case_smoke.yaml")
    unsupported.run_config.physics.eedf_backend = "swarm"
    unsupported.run_config.swarm.model_name = "future_solver"
    with pytest.raises(
        MigrationError,
        match=r"unsupported v2 EEDF selection 'swarm'/'future_solver'",
    ):
        _electron_models(unsupported, set(), [])

    unsupported.run_config.physics.eedf_backend = "maxwell"
    unsupported.run_config.physics.electron_density_closure = "future_density"
    with pytest.raises(
        MigrationError,
        match="unsupported v2 electron density closure 'future_density'",
    ):
        _electron_models(unsupported, set(), [])


def test_prescribed_density_external_key_preserves_mapping_and_warning(
    tmp_path: Path,
) -> None:
    loaded = load_legacy_v2_case(V2_CONFIGS / "case_smoke.yaml")
    profile_path = tmp_path / "density.csv"
    loaded.run_config.physics.electron_density_closure = "prescribed_profile"
    profile = loaded.run_config.swarm.prescribed_electron_profile
    profile.file = None
    profile.file_key = "density_profile"
    profile.zone_columns = {"plasma": "ne"}
    profile.interpolation = "previous"
    profile.hold = "edge"
    loaded.resolved_paths.external_inputs["density_profile"] = str(profile_path)
    warnings: list[str] = []

    models = _electron_models(loaded, set(), warnings)

    assert models["electron_density"] == {
        "kind": "experimental.prescribed_profile",
        "file": profile_path.resolve(),
        "zone_columns": {"plasma": "ne"},
        "interpolation": "previous",
        "hold": "error",
    }
    assert warnings == [
        "prescribed electron profile hold was changed to 'error' to "
        "prevent silent endpoint extension"
    ]

    profile.file_key = "missing"
    with pytest.raises(
        MigrationError,
        match="v2 prescribed electron profile has no input file",
    ):
        _electron_models(loaded, set(), [])


def test_external_assets_are_deduplicated_copied_and_written_relative(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-power.csv"
    source_bytes = b"time_s,electron_power_W\r\n0.0,1.25\r\n1.0,2.5\r\n"
    source.write_bytes(source_bytes)
    data = load_case(V3_FIXTURE).model_dump(mode="python")
    data["reactor"]["power_ports"][0]["model"] = {
        "kind": "external_table",
        "file": source,
    }
    data["recipe"]["steps"][0]["commands"]["power_ports"]["source"] = {
        "kind": "external_table",
        "file": source,
    }
    case = CaseSpec.model_validate(data)
    staging = tmp_path / "staging"
    bundle = tmp_path / "bundle"
    published_assets = bundle / "case_assets"

    staged_case, warnings = _stage_case_assets(case, staging, published_assets)

    staged_files = tuple(staging.iterdir())
    assert [path.name for path in staged_files] == ["00_power_model_0_source.csv"]
    assert staged_files[0].read_bytes() == source_bytes
    model_file = staged_case.reactor.power_ports[0].model.file
    command = staged_case.recipe.steps[0].commands.power_ports["source"]
    assert (
        command.file
        == model_file
        == (published_assets / "00_power_model_0_source.csv").resolve()
    )
    assert warnings == ()

    result = MigrationResult(
        case=staged_case,
        report=MigrationReport(source_path=source, unused_keys=(), warnings=()),
    )
    destination = write_v3_case(result, bundle / "case.yaml")
    written = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert written["reactor"]["power_ports"][0]["model"]["file"] == (
        "case_assets/00_power_model_0_source.csv"
    )
    assert (
        written["recipe"]["steps"][0]["commands"]["power_ports"]["source"]["file"]
        == "case_assets/00_power_model_0_source.csv"
    )


def test_electron_table_staging_preserves_conversion_warning_order(
    tmp_path: Path,
) -> None:
    case = migrate_v2(V2_CONFIGS / "case_zdplaskin_example2.yaml").case

    staged_case, warnings = _stage_case_assets(
        case,
        tmp_path / "staging",
        tmp_path / "published",
    )

    assert staged_case.models.electrons.kind == "table"
    assert (
        staged_case.models.electrons.file
        == (tmp_path / "published" / "00_models_electrons.h5").resolve()
    )
    assert warnings == (
        "models_electrons: dropped unused legacy HDF5 datasets: diffusion_m2_s",
    )
    assert (tmp_path / "staging" / "00_models_electrons.h5").is_file()


def test_prescribed_density_asset_is_copied_with_stable_label(tmp_path: Path) -> None:
    profile_path = tmp_path / "density.csv"
    profile_bytes = b"time_s,ne\n0.0,1e12\n1.0,2e12\n"
    profile_path.write_bytes(profile_bytes)
    data = load_case(V3_FIXTURE).model_dump(mode="python")
    data["models"]["electron_density"] = {
        "kind": "experimental.prescribed_profile",
        "file": profile_path,
        "zone_columns": {"plasma": "ne"},
        "interpolation": "previous",
        "hold": "error",
    }
    case = CaseSpec.model_validate(data)
    staging = tmp_path / "staging"
    published = tmp_path / "published"

    staged_case, warnings = _stage_case_assets(case, staging, published)

    staged_path = staging / "00_models_electron_density.csv"
    assert staged_path.read_bytes() == profile_bytes
    assert (
        staged_case.models.electron_density.file
        == (published / staged_path.name).resolve()
    )
    assert warnings == ()

    missing_data = case.model_dump(mode="python")
    missing_data["models"]["electron_density"]["file"] = tmp_path / "missing.csv"
    with pytest.raises(
        MigrationError,
        match=r"models_electron_density does not exist: .*missing\.csv",
    ):
        _stage_case_assets(
            CaseSpec.model_validate(missing_data),
            tmp_path / "unused-staging",
            tmp_path / "unused-published",
        )
