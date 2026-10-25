"""Optional plotting adapter for canonical HDF5 results."""

from __future__ import annotations

import importlib
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global._result_common import new_output_directory
from plasma_global.core.result import SimulationResult, series_unit, to_plain_mapping
from plasma_global.input._schema_output import OutputConfig, PlotConfig


def _validate_plot_request(
    result: SimulationResult,
    names: Sequence[str],
    image_format: str,
    dpi: int,
) -> None:
    if not names:
        raise ValueError("the result contains no series to plot")
    if result.n_times == 0:
        raise ValueError("an empty result cannot be plotted")
    if not image_format.isalnum() or dpi <= 0:
        raise ValueError("plot format and dpi are invalid")
    if len(set(names)) != len(names):
        raise ValueError("plot series must not contain duplicates")
    for name in names:
        result.series(name)


def _load_pyplot():
    try:
        importlib.import_module("matplotlib")
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires the optional dependency: pip install .[plot]"
        ) from exc


def _series_filename(name: str, index: int, image_format: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.")
    return f"{index:04d}_{stem[:80] or 'series'}.{image_format}"


def _plot_series(
    plt: Any,
    result: SimulationResult,
    name: str,
    path: Path,
    dpi: int,
) -> None:
    values = result.series(name)
    if values.ndim == 0:
        values = np.full(result.n_times, float(values))
    figure, axes = plt.subplots()
    axes.plot(result.time_s, values)
    axes.set_xlabel("time [s]")
    unit = series_unit(name, result.metadata.get("series_units", {}))
    axes.set_ylabel(f"{name} [{unit}]")
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=dpi)
    finally:
        plt.close(figure)


def _configured_requests(
    configured: Mapping[str, PlotConfig], image_format: str | None, dpi: int | None
) -> dict[str, PlotConfig]:
    requests = dict(configured)
    overrides: dict[str, Any] = {}
    if image_format is not None:
        overrides["format"] = image_format
    if dpi is not None:
        overrides["dpi"] = dpi
    return {
        name: PlotConfig.model_validate({**config.model_dump(), **overrides})
        for name, config in requests.items()
    }


def plot_result_h5(
    result_h5: str | Path,
    output_dir: str | Path,
    *,
    series: Sequence[str] = (),
    image_format: str | None = None,
    dpi: int | None = None,
    result_reader: Callable[[str | Path], SimulationResult],
) -> tuple[Path, ...]:
    """Plot selected series; matplotlib is imported only by this operation."""

    result = result_reader(result_h5)
    configured = OutputConfig.model_validate(
        to_plain_mapping(result.metadata.get("output_requests", {}))
    ).plots
    if not series and configured:
        requests = _configured_requests(configured, image_format, dpi)
        directory = Path(output_dir).resolve()
        with new_output_directory(directory) as stage:
            configured_paths = write_configured_plots(
                result, stage / "figures", requests
            )
            relative = tuple(path.relative_to(stage) for path in configured_paths)
        return tuple(directory / path for path in relative)
    image_format = image_format or "png"
    dpi = 150 if dpi is None else dpi
    names = tuple(series) or tuple(result.observables) or result.state_labels
    _validate_plot_request(result, names, image_format, dpi)
    plt = _load_pyplot()
    directory = Path(output_dir).resolve()
    paths: list[Path] = []
    with new_output_directory(directory) as stage:
        for index, name in enumerate(names):
            filename = _series_filename(name, index, image_format)
            _plot_series(plt, result, name, stage / filename, dpi)
            paths.append(directory / filename)
    return tuple(paths)


def _composition_selection(
    result: SimulationResult, config: PlotConfig
) -> tuple[dict[str, str], tuple[str, ...]]:
    prefix = f"n[{config.zone},"
    all_names = [name for name in result.state_labels if name.startswith(prefix)]
    if not all_names:
        raise ValueError(f"no saved composition for zone {config.zone!r}")
    available = {name[len(prefix) : -1]: name for name in all_names}
    species = tuple(available) if config.species == "all" else tuple(config.species)
    if not species or len(set(species)) != len(species):
        raise ValueError("composition species must be nonempty and unique")
    if unknown := set(species) - set(available):
        raise ValueError(f"unknown composition species: {sorted(unknown)}")
    return available, species


def composition_data(
    result: SimulationResult, config: PlotConfig
) -> dict[str, np.ndarray]:
    available, species = _composition_selection(result, config)
    all_names = tuple(available.values())
    denominator = sum(
        (result.series(name) for name in all_names), np.zeros(result.n_times)
    )
    if config.quantity == "mole_fraction" and np.any(denominator <= 0):
        raise ValueError(
            "mole fraction is undefined when total heavy-species density is zero"
        )
    return {
        name: result.series(available[name]) / denominator
        if config.quantity == "mole_fraction"
        else result.series(available[name])
        for name in species
    }


def _group_data(result: SimulationResult, config: PlotConfig):
    if config.kind == "composition":
        unit = "1" if config.quantity == "mole_fraction" else "m^-3"
        return {unit: composition_data(result, config)}
    # Load selection expansion only when grouped series are requested.
    from plasma_global._series_selection import expand_series

    names = expand_series(config.series, (*result.state_labels, *result.observables))
    groups: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for name in names:
        unit = series_unit(name, result.metadata.get("series_units", {}))
        values = result.series(name)
        groups[unit][name] = (
            np.full(result.n_times, float(values)) if values.ndim == 0 else values
        )
    return groups


def _plot_axis(
    axes: Any,
    times: np.ndarray,
    values: Mapping[str, np.ndarray],
    config: PlotConfig,
    unit: str,
) -> None:
    omitted = 0
    for label, raw in values.items():
        if config.yscale == "log":
            if np.any(raw < 0):
                raise ValueError(
                    f"log plot contains negative values in {label!r}; "
                    "choose linear or symlog"
                )
            omitted += int(np.count_nonzero(raw == 0))
            raw = np.where(raw > 0, raw, np.nan)
        axes.plot(times, raw, label=label)
    axes.set_yscale(config.yscale)
    axes.set_ylabel(
        f"{config.quantity if config.kind == 'composition' else 'value'} [{unit}]"
    )
    axes.grid(True, alpha=0.2)
    axes.legend(loc="best", fontsize="small")
    if omitted:
        axes.text(
            0.01,
            0.02,
            f"{omitted} zero values omitted on log scale",
            transform=axes.transAxes,
            fontsize="small",
        )


def _plot_title(name: str, config: PlotConfig) -> str:
    title = config.title or name
    if config.kind == "composition" and config.quantity == "mole_fraction":
        title += (
            "\nBasis: all heavy gas species; electrons excluded; "
            "selected species not renormalized"
        )
    return title


def _plot_group(
    plt: Any, result: SimulationResult, name: str, config: PlotConfig, path: Path
) -> None:
    if not result.n_times:
        raise ValueError("an empty result cannot be plotted")
    groups = _group_data(result, config)
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}[config.time_unit]
    figure, panels = plt.subplots(
        len(groups), 1, squeeze=False, figsize=(10, 4 * len(groups)), sharex=True
    )
    try:
        for axes, (unit, values) in zip(panels[:, 0], groups.items(), strict=True):
            _plot_axis(axes, result.time_s * scale, values, config, unit)
            for boundary in result.metadata.get("run_step_boundaries_s", ())[1:-1]:
                axes.axvline(
                    boundary * scale,
                    color="grey",
                    linestyle=":",
                    linewidth=0.7,
                )
            if config.time_range_s is not None:
                if (
                    config.time_range_s[0] < result.time_s[0]
                    or config.time_range_s[1] > result.time_s[-1]
                ):
                    raise ValueError("plot time_range_s is outside saved data")
                axes.set_xlim(*(value * scale for value in config.time_range_s))
        panels[-1, 0].set_xlabel(f"time [{config.time_unit}]")
        figure.suptitle(_plot_title(name, config))
        figure.tight_layout()
        figure.savefig(path, dpi=config.dpi)
    finally:
        plt.close(figure)


def write_configured_plots(
    result: SimulationResult, directory: Path, requests: Mapping[str, PlotConfig]
) -> tuple[Path, ...]:
    plt = _load_pyplot()
    directory.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for name, config in requests.items():
        path = directory / f"{name}.{config.format}"
        _plot_group(plt, result, name, config, path)
        paths.append(path)
    return tuple(paths)


__all__ = ["plot_result_h5"]
