from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from plasma_global.errors import MigrationError
from plasma_global.input.migrate_v2 import _port_model, _power_command


def _legacy_port(
    *, port_id: str = "source", kind: str = "source", parameters: dict | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        port_id=port_id,
        kind=kind,
        parameters={} if parameters is None else parameters,
    )


def _legacy_recipe(*commands: dict) -> SimpleNamespace:
    return SimpleNamespace(
        steps=[SimpleNamespace(power_ports={"source": command}) for command in commands]
    )


def _model(
    tmp_path: Path,
    *,
    backend: str,
    port: SimpleNamespace,
    recipe: SimpleNamespace | None = None,
    external_inputs: dict[str, str | None] | None = None,
) -> tuple[dict, set[str], set[str]]:
    used_external: set[str] = set()
    unused: set[str] = set()
    result = _port_model(
        port=port,
        backend=backend,
        recipe=_legacy_recipe() if recipe is None else recipe,
        legacy_base_dir=tmp_path,
        external_inputs={} if external_inputs is None else external_inputs,
        used_external=used_external,
        unused=unused,
        prefix="reactor.power_ports[0]",
        gas_fraction=0.2,
    )
    return result, used_external, unused


def _command(
    tmp_path: Path,
    *,
    kind: str,
    values: dict,
) -> tuple[dict, set[str], set[str], list[str]]:
    used_external: set[str] = set()
    unused: set[str] = set()
    warnings: list[str] = []
    result = _power_command(
        values=values,
        kind=kind,
        base_dir=tmp_path,
        external_inputs={},
        used_external=used_external,
        prefix="recipe.steps[0].power_ports.source",
        unused=unused,
        warnings=warnings,
    )
    return result, used_external, unused, warnings


def test_power_model_migration_keeps_static_setpoints_and_partitions(
    tmp_path: Path,
) -> None:
    prescribed, _, prescribed_unused = _model(
        tmp_path,
        backend="direct_power",
        port=_legacy_port(parameters={"power_W": 80.0}),
    )
    delivered, _, delivered_unused = _model(
        tmp_path,
        backend="icp",
        port=_legacy_port(parameters={"delivered_power_W": 250.0}),
    )

    assert prescribed == {
        "kind": "prescribed_power",
        "electron_fraction": 0.8,
        "gas_fraction": 0.2,
        "default_absorbed_power_W": 80.0,
    }
    assert delivered == {
        "kind": "experimental.icp",
        "frequency_Hz": 13.56e6,
        "default_delivered_power_W": 250.0,
    }
    assert prescribed_unused == set()
    assert delivered_unused == set()


def test_external_table_model_resolves_static_recipe_and_fallback_files(
    tmp_path: Path,
) -> None:
    static, _, static_unused = _model(
        tmp_path,
        backend="external_circuit_table",
        port=_legacy_port(
            parameters={
                "file": "static.csv",
                "interpolation": "PREVIOUS",
                "hold": "clip",
                "power_scale": 2.0,
                "voltage_scale": 3.0,
                "current_scale": 4.0,
                "gap_m": 0.01,
                "total_density_m3": 1.0e20,
                "plasma_potential_V": 7.0,
                "time_offset_s": 2.0,
            }
        ),
    )
    recipe_file, _, _ = _model(
        tmp_path,
        backend="external_circuit_table",
        port=_legacy_port(),
        recipe=_legacy_recipe({"file": "recipe.csv"}),
    )
    fallback_path = tmp_path / "fallback.csv"
    fallback, used_external, _ = _model(
        tmp_path,
        backend="external_circuit_table",
        port=_legacy_port(),
        external_inputs={"circuit_result_csv": str(fallback_path)},
    )

    assert static == {
        "kind": "external_table",
        "file": (tmp_path / "static.csv").resolve(),
        "interpolation": "previous",
        "bounds_policy": "hold",
        "power_scale": 2.0,
        "voltage_scale": 3.0,
        "current_scale": 4.0,
        "plasma_potential_V": 7.0,
        "gap_m": 0.01,
        "total_density_m3": 1.0e20,
    }
    assert static_unused == {"reactor.power_ports[0].parameters.time_offset_s"}
    assert recipe_file["file"] == (tmp_path / "recipe.csv").resolve()
    assert fallback["file"] == fallback_path.resolve()
    assert used_external == {"circuit_result_csv"}


def test_external_table_model_requires_one_resolvable_file(tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="external_table has no input file"):
        _model(
            tmp_path,
            backend="external_circuit_table",
            port=_legacy_port(),
        )


def test_power_command_migration_handles_prescribed_and_nested_dc(
    tmp_path: Path,
) -> None:
    prescribed, _, prescribed_unused, _ = _command(
        tmp_path,
        kind="prescribed_power",
        values={
            "power_W": 12.0,
            "waveform": "square",
            "duty_cycle": 0.25,
            "repetition_Hz": 10.0,
        },
    )
    dc, _, dc_unused, _ = _command(
        tmp_path,
        kind="dc_series",
        values={
            "voltage": {
                "high_V": 100.0,
                "low_V": 5.0,
                "waveform": "square",
                "duty_cycle": 0.4,
                "frequency_Hz": 50.0,
            }
        },
    )
    scalar_dc, _, scalar_unused, _ = _command(
        tmp_path,
        kind="dc_series",
        values={"voltage": 5.0},
    )

    assert prescribed == {
        "kind": "prescribed_power",
        "absorbed_power_W": 12.0,
        "waveform": {
            "kind": "square_pulse",
            "duty_cycle": 0.25,
            "repetition_Hz": 10.0,
        },
    }
    assert dc == {
        "kind": "dc_series",
        "source_voltage_V": 100.0,
        "off_voltage_V": 5.0,
        "waveform": {
            "kind": "square_pulse",
            "duty_cycle": 0.4,
            "repetition_Hz": 50.0,
        },
    }
    assert dc_unused == set()
    assert "source_voltage_V" not in scalar_dc
    assert scalar_unused == {"recipe.steps[0].power_ports.source.voltage"}
    assert prescribed_unused == set()


def test_external_and_experimental_step_commands_keep_distinct_contracts(
    tmp_path: Path,
) -> None:
    external, _, external_unused, _ = _command(
        tmp_path,
        kind="external_table",
        values={
            "file": "power.csv",
            "power_scale": 2.0,
            "voltage_scale": 3.0,
            "current_scale": 4.0,
            "time_offset_s": 0.5,
            "gap_m": 0.01,
            "total_density_m3": 1.0e20,
            "plasma_potential_V": 8.0,
            "interpolation": "PREVIOUS",
            "bounds_policy": "edge",
            "waveform": "off",
        },
    )
    voltage, _, _, _ = _command(
        tmp_path,
        kind="experimental.ccp",
        values={"control_mode": "voltage", "value": 30.0},
    )
    power, _, _, _ = _command(
        tmp_path,
        kind="experimental.rf_envelope",
        values={"value": 40.0},
    )
    delivered, _, _, _ = _command(
        tmp_path,
        kind="experimental.icp",
        values={"value": 50.0},
    )

    assert external == {
        "kind": "external_table",
        "file": (tmp_path / "power.csv").resolve(),
        "power_scale": 2.0,
        "voltage_scale": 3.0,
        "current_scale": 4.0,
        "time_offset_s": 0.5,
        "gap_m": 0.01,
        "total_density_m3": 1.0e20,
        "plasma_potential_V": 8.0,
        "interpolation": "previous",
        "bounds_policy": "hold",
    }
    assert external_unused == {"recipe.steps[0].power_ports.source.waveform"}
    assert voltage["voltage_rms_V"] == 30.0
    assert power["absorbed_power_W"] == 40.0
    assert delivered["delivered_power_W"] == 50.0
