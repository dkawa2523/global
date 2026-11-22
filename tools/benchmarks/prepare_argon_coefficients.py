"""Generate the Ar example coefficients with an independent Swarm checkout.

Run this script with an environment containing Swarm's dependencies. The
checkout is imported read-only; native results are written below --work-dir.
Global consumes only the resulting coefficient CSV and recorded provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tools._process import run_process

ROOT = Path(__file__).resolve().parents[2]
CHEMISTRY = ROOT / "plasma_global/examples/chemistry/argon_lxcat"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_swarm_input(
    work: Path, fields: list[float]
) -> tuple[Path, Path, dict[str, Any]]:
    """Convert the canonical cross sections and write the independent DC case."""
    chemistry = yaml.safe_load(
        (CHEMISTRY / "chemistry.yaml").read_text(encoding="utf-8")
    )
    channels = {item["id"]: item for item in chemistry["cross_sections"]}
    converted = work / "argon_cross_sections.csv"
    with converted.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "species",
                "process",
                "type",
                "threshold_eV",
                "mass_amu",
                "energy_eV",
                "cross_section_m2",
            ]
        )
        with (CHEMISTRY / "cross_sections.csv").open(encoding="utf-8") as source:
            for row in csv.DictReader(source):
                channel = channels[row["channel_id"]]
                writer.writerow(
                    [
                        channel["target"],
                        channel["id"],
                        "momentum"
                        if channel["kind"] == "momentum_transfer"
                        else channel["kind"],
                        channel["threshold_eV"],
                        39.948,
                        row["energy_eV"],
                        row["sigma_m2"],
                    ]
                )

    config = {
        "schema_version": 2,
        "run": {
            "solvers": [{"id": "two_term"}],
            "e_over_n_Td": fields,
            "case_prefix": "global_argon",
        },
        "conditions": {
            "gas_temperature_K": 350.0,
            "pressure_Pa": 8.0,
            "gas_mixture": [{"species": "Ar", "fraction": 1.0, "mass_amu": 39.948}],
        },
        "cross_sections": {"files": [{"path": str(converted), "species": "Ar"}]},
        "output": {
            "directory": str(work / "native"),
            "base_name": "argon",
            "export": {
                "solver": "two_term",
                "quantities": ["reaction_rates", "transport"],
            },
        },
    }
    source_case = work / "case.yaml"
    source_case.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return source_case, converted, channels


def _read_swarm_results(
    native: Path, field_count: int
) -> tuple[list[dict[str, str]], dict[str, dict[str, float]]]:
    """Require a converged result at each field and index per-target rates."""
    with (native / "argon_summary.csv").open(encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    invalid = [
        row["E_over_N_Td"]
        for row in summary
        if row["meta_numerical_status"] != "numerically_converged"
    ]
    if invalid or len(summary) != field_count:
        raise ValueError(f"Swarm did not converge at every requested field: {invalid}")
    rates: dict[str, dict[str, float]] = {}
    with (native / "argon_rates.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rates.setdefault(row["case_id"], {})[row["process"]] = float(
                row["rate_coefficient_m3_s"]
            )
    return summary, rates


def _export_coefficients(
    summary: list[dict[str, str]],
    rates: dict[str, dict[str, float]],
    channel_ids: list[str],
    target: Path,
) -> None:
    """Write Global's existing coefficient format without changing native values."""
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "reduced_field_Td",
                "mean_energy_eV",
                "reduced_mobility_1_V_m_s",
                *(f"k:{name}" for name in channel_ids),
            ]
        )
        for row in summary:
            coefficients = rates[row["case_id"]]
            if set(coefficients) != set(channel_ids):
                raise ValueError(
                    "Swarm rate channels differ from the chemistry cross sections"
                )
            writer.writerow(
                [
                    float(row["E_over_N_Td"]),
                    float(row["mean_energy_eV"]),
                    float(row["reduced_mobility_m2_V_s_m3"]),
                    *(coefficients[name] for name in channel_ids),
                ]
            )


def _swarm_source_hash(swarm: Path) -> str:
    """Identify the actual external Python source, including local modifications."""
    source_hash = hashlib.sha256()
    for directory in ("electron_swarm", "swarm_workflow"):
        for path in sorted((swarm / directory).rglob("*.py")):
            source_hash.update(path.relative_to(swarm).as_posix().encode())
            source_hash.update(path.read_bytes())
    return source_hash.hexdigest()


def _write_provenance(
    swarm: Path,
    native: Path,
    converted: Path,
    fields: list[float],
    channels: dict[str, Any],
    target: Path,
) -> None:
    """Record the calculation conditions and exact native and exported artifacts."""
    channel_ids = sorted(channels)
    identity = run_process(["git", "rev-parse", "HEAD"], cwd=swarm, capture_output=True)
    if identity.returncode != 0:
        raise RuntimeError(f"Cannot identify Swarm checkout: {identity.stderr.strip()}")
    revision = identity.stdout.strip()
    metadata = {
        "source_repository": "https://github.com/dkawa2523/swarm",
        "source_revision": revision,
        "source_working_tree_python_sha256": _swarm_source_hash(swarm),
        "source_execution_id": json.loads(
            (native / "argon_manifest.json").read_text(encoding="utf-8")
        )["execution"]["id"],
        "source_manifest_sha256": _sha(native / "argon_manifest.json"),
        "generation_script": "tools/benchmarks/prepare_argon_coefficients.py",
        "solver": "two_term",
        "backend": "native_sg",
        "numerical_status": "numerically_converged",
        "fields_Td": fields,
        "gas_temperature_K": 350.0,
        "pressure_Pa": 8.0,
        "mole_fractions": {"Ar": 1.0},
        "magnetic_field_T": 0.0,
        "field_waveform": "DC",
        "ionization_source": "equal",
        "growth_model": "temporal_growth",
        "cross_sections": (CHEMISTRY / "cross_sections.csv")
        .relative_to(ROOT)
        .as_posix(),
        "cross_sections_sha256": _sha(CHEMISTRY / "cross_sections.csv"),
        "chemistry_definition_sha256": _sha(CHEMISTRY / "chemistry.yaml"),
        "elastic_threshold_eV": 0.0,
        "source_elastic_mass_ratio": channels["xs_ar_momentum_zenodo"]["metadata"][
            "original_electron_target_mass_ratio"
        ],
        "cross_section_source_doi": "10.5281/zenodo.8192503",
        "cross_section_license": "CC-BY-4.0",
        "converted_swarm_cross_sections_sha256": _sha(converted),
        "channel_ids": channel_ids,
        "rate_basis": "per target particle, no mixture weighting",
        "mobility": "flux drift over field, exported as mu*N",
        "fixed_state_scope": (
            "Pure Ar at 350 K. Shared by both example zones; runtime gas "
            "temperature does not regenerate the coefficients."
        ),
        "purpose": (
            "Independent coefficient input for illustrative RF/ICP/CCP coupling; "
            "DC coefficients are a local-field approximation, not an RF swarm "
            "solution or device validation."
        ),
        "electrons_csv_sha256": _sha(target),
    }
    (target.parent / "provenance.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swarm-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "plasma_global/examples/data/swarm_argon_rf",
    )
    args = parser.parse_args()
    swarm = args.swarm_root.resolve()
    work = args.work_dir.resolve()
    output = args.output.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    fields = sorted({*np.geomspace(1.0, 200.0, 80).tolist(), 25.0, 100.0})
    source_case, converted, channels = _prepare_swarm_input(work, fields)
    sys.path.insert(0, str(swarm))
    importlib.import_module("electron_swarm").run_from_config(source_case)
    native = work / "native"
    summary, rates = _read_swarm_results(native, len(fields))
    channel_ids = sorted(channels)
    target = output / "electrons.csv"
    _export_coefficients(summary, rates, channel_ids, target)
    _write_provenance(swarm, native, converted, fields, channels, target)
    print(
        f"Wrote {len(summary)} converged fields and "
        f"{len(channel_ids)} channels to {target}"
    )


if __name__ == "__main__":
    main()
