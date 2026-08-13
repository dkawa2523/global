"""Optional plotting adapter for canonical HDF5 results."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.core.result import SimulationResult


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


def _load_pyplot():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires the optional dependency: pip install .[plot]"
        ) from exc


def _series_filename(name: str, index: int, image_format: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.")
    return f"{stem or f'series_{index}'}.{image_format}"


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
    axes.set_ylabel(name)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def plot_result_h5(
    result_h5: str | Path,
    output_dir: str | Path,
    *,
    series: Sequence[str] = (),
    image_format: str = "png",
    dpi: int = 150,
    result_reader: Callable[[str | Path], SimulationResult],
) -> tuple[Path, ...]:
    """Plot selected series; matplotlib is imported only by this operation."""

    result = result_reader(result_h5)
    names = tuple(series) or tuple(result.observables) or result.state_labels
    _validate_plot_request(result, names, image_format, dpi)
    plt = _load_pyplot()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, name in enumerate(names):
        path = directory / _series_filename(name, index, image_format)
        _plot_series(plt, result, name, path, dpi)
        paths.append(path)
    return tuple(paths)


__all__ = ["plot_result_h5"]
