"""Recompute the manual's examples, figures, and inspectable data excerpts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

from plasma_global import load_case, simulate, write_result
from plasma_global.core.result import SimulationResult, to_plain_mapping
from plasma_global.examples import initialize_example
from plasma_global.input._schema_output import NetworkConfig, PlotConfig
from plasma_global.output import network_result_h5, read_result_h5

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"


def _rows(
    result: SimulationResult, times: tuple[float, ...], names: tuple[str, ...]
) -> list[dict[str, float]]:
    rows = []
    for time in times:
        matches = np.flatnonzero(
            np.abs(result.time_s - time) <= 8 * abs(np.spacing(time))
        )
        if matches.size != 1:
            raise ValueError(f"requested excerpt time {time:g} is not a saved point")
        index = int(matches[0])
        rows.append(
            {
                "time_s": float(result.time_s[index]),
                **{name: float(result.series(name)[index]) for name in names},
            }
        )
    return rows


def _layout(path: Path) -> dict[str, object]:
    datasets: dict[str, object] = {}
    with h5py.File(path, "r") as source:

        def record(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset):
                datasets[f"/{name}"] = {
                    "shape": item.shape,
                    "dtype": str(item.dtype),
                    "compression": item.compression,
                }

        source.visititems(record)
    return datasets


def _run(
    name: str, output: Path, plots: dict[str, PlotConfig] | None = None
) -> tuple[SimulationResult, Path]:
    case = initialize_example(name, output / "inputs" / name)
    if plots:
        document = yaml.safe_load(case.read_text(encoding="utf-8"))
        selected = document.setdefault("output", {})
        selected["plots"] = {
            identity: config.model_dump(mode="json")
            for identity, config in plots.items()
        }
        case.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    result = simulate(load_case(case))
    if result.status.success is not True:
        raise RuntimeError(f"{name}: {result.status.message}")
    paths = write_result(result, output / "runs" / name)
    restored = read_result_h5(paths.result_h5)
    if not np.array_equal(result.state, restored.state):
        raise ValueError(f"{name}: saved states differ from the calculation")
    for series, values in result.observables.items():
        if not np.array_equal(values, restored.series(series)):
            raise ValueError(f"{name}: saved observable differs: {series}")
    print(f"{name}: {result.n_times} saved times, HDF5 round trip verified", flush=True)
    return restored, paths.result_h5


def render(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    argon, argon_h5 = _run("argon_analysis", output)
    configured = {
        "dc-circuit": PlotConfig(
            series=[
                "port_plasma_voltage_V[dc_series_drive]",
                "port_current_A[dc_series_drive]",
                "reduced_field_Td[plasma]",
            ],
            time_unit="ms",
        )
    }
    dc, dc_h5 = _run("zdplaskin_example2", output, configured)
    network_result_h5(
        argon_h5,
        output / "ion-mechanism",
        config=NetworkConfig(
            mode="mechanism", species=["Ar_plus"], domains=["gas", "boundary"]
        ),
    )
    copies = {
        output / "runs/argon_analysis/plots/plasma.png": "argon-plasma.png",
        output / "runs/argon_analysis/plots/densities.png": "argon-densities.png",
        output / "runs/argon_analysis/plots/fractions.png": "argon-fractions.png",
        output / "runs/argon_analysis/networks/events.svg": "argon-events.svg",
        output / "ion-mechanism/diagrams/network.svg": "argon-mechanism.svg",
        output / "runs/zdplaskin_example2/plots/dc-circuit.png": "dc-circuit.png",
    }
    for source, name in copies.items():
        shutil.copy2(source, ASSETS / name)
    excerpts = {
        "argon": {
            "state_labels": argon.state_labels,
            "state_shape": argon.state.shape,
            "observable_count": len(argon.observables),
            "metadata_keys": list(argon.metadata),
            "status": argon.status.code,
            "solver": to_plain_mapping(argon.solver_stats),
            "rows": _rows(
                argon,
                (0.0, 0.0005, 0.001, 0.00101, 0.002),
                (
                    "n[plasma,Ar_plus]",
                    "electron_temperature_eV[plasma]",
                    "absorbed_power_W[plasma]",
                    "probe[wall,incident_flux_m2_s,Ar_plus]",
                    "gas_event_count[plasma,G_AR_ION]",
                ),
            ),
            "hdf5": _layout(argon_h5),
        },
        "dc": {
            "state_labels": dc.state_labels,
            "state_shape": dc.state.shape,
            "final": {
                name: dc.final_value(name)
                for name in (
                    *configured["dc-circuit"].series,
                    "electron_density_m3[plasma]",
                )
            },
            "hdf5": _layout(dc_h5),
        },
    }
    (output / "values.json").write_text(
        json.dumps(excerpts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(copies)} manual figures updated; excerpts: {output / 'values.json'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    render(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
