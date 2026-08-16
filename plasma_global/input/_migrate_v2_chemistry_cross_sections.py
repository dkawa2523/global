"""Validate and publish legacy cross sections without scientific repair."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from plasma_global.errors import MigrationError
from plasma_global.input._migrate_v2_chemistry_models import (
    read_rows,
    read_yaml,
    write_rows,
    write_yaml,
)


def publish_cross_sections(
    manifest_filename: Any,
    base: Path,
    target: Path,
    selected_segments: Mapping[str, int] | None,
) -> str | None:
    """Publish selected legacy axes and their canonical manifest."""

    if not manifest_filename:
        return None
    cross_source = (base / str(manifest_filename)).resolve()
    entries = read_yaml(cross_source).get("cross_sections") or []
    converted: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.get("file"):
            raise MigrationError(
                f"cross section {entry.get('cross_section_id')} has no canonical "
                "source file"
            )
        original = (cross_source.parent / str(entry["file"])).resolve()
        destination = target / "cross_sections" / original.name
        cross_section_id = str(entry["cross_section_id"])
        segment_index = (selected_segments or {}).get(cross_section_id)
        kind = str(entry.get("kind", "inelastic"))
        threshold = float(entry.get("threshold_eV", 0.0))
        _write_canonical_cross_section(
            original,
            destination,
            segment_index=segment_index,
            require_zero_energy=kind == "momentum_transfer",
        )
        converted.append(
            {
                "id": entry["cross_section_id"],
                "kind": kind,
                "target": entry.get("target_species", ""),
                "threshold_eV": threshold,
                "energy_loss_eV": float(
                    entry.get("energy_loss_eV", entry.get("threshold_eV", 0.0))
                ),
                "file": f"cross_sections/{original.name}",
                "metadata": entry.get("metadata", {}),
            }
        )
    output_name = "cross_sections.yaml"
    write_yaml(target / output_name, {"cross_sections": converted})
    return output_name


def _write_canonical_cross_section(
    source: Path,
    destination: Path,
    *,
    segment_index: int | None = None,
    require_zero_energy: bool = False,
) -> None:
    """Convert a legacy SI curve to the runtime's strict two-column form.

    A repeated energy node is removed only when its cross section is exactly
    identical. Conflicting duplicates, descending axes, negative data, and
    non-finite values are rejected rather than repaired.
    """

    segments = _read_cross_section_segments(source)
    converted = _canonical_cross_section_rows(source, segments, segment_index)
    _validate_zero_energy_support(source, converted, require_zero_energy)
    write_rows(destination, ["energy_eV", "sigma_m2"], converted)


def _read_cross_section_segments(
    source: Path,
) -> list[list[tuple[float, float]]]:
    rows = read_rows(source)
    if not rows or set(rows[0]) != {"energy_eV", "sigma_m2"}:
        raise MigrationError(
            f"cross section {source} must have exactly energy_eV,sigma_m2 columns"
        )
    segments: list[list[tuple[float, float]]] = [[]]
    previous_energy: float | None = None
    for index, row in enumerate(rows, start=2):
        energy, sigma = _parse_cross_section_point(source, index, row)
        if previous_energy is not None and energy < previous_energy:
            segments.append([])
        segments[-1].append((energy, sigma))
        previous_energy = energy
    return segments


def _canonical_cross_section_rows(
    source: Path,
    segments: list[list[tuple[float, float]]],
    segment_index: int | None,
) -> list[dict[str, float]]:
    selected = _select_cross_section_segment(source, segments, segment_index)
    converted: list[dict[str, float]] = []
    previous_energy: float | None = None
    previous_sigma: float | None = None
    for energy, sigma in selected:
        if previous_energy is not None and energy == previous_energy:
            if sigma != previous_sigma:
                raise MigrationError(
                    f"cross section {source} has conflicting values at {energy:g} eV"
                )
            continue
        converted.append({"energy_eV": energy, "sigma_m2": sigma})
        previous_energy, previous_sigma = energy, sigma
    if len(converted) < 2:
        raise MigrationError(f"cross section {source} needs at least two unique nodes")
    return converted


def _parse_cross_section_point(
    source: Path, index: int, row: Mapping[str, str]
) -> tuple[float, float]:
    try:
        energy = float(row["energy_eV"])
        sigma = float(row["sigma_m2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MigrationError(
            f"cross section {source}:{index} must contain numeric SI values"
        ) from exc
    if not math.isfinite(energy) or not math.isfinite(sigma):
        raise MigrationError(f"cross section {source}:{index} is non-finite")
    if energy < 0.0 or sigma < 0.0:
        raise MigrationError(f"cross section {source}:{index} is negative")
    return energy, sigma


def _select_cross_section_segment(
    source: Path,
    segments: list[list[tuple[float, float]]],
    segment_index: int | None,
) -> list[tuple[float, float]]:
    if len(segments) > 1 and segment_index is None:
        raise MigrationError(
            f"cross section {source} contains {len(segments)} concatenated axes; "
            "select one explicitly with --cross-section-segments"
        )
    selected_index = 0 if segment_index is None else segment_index
    if selected_index < 0 or selected_index >= len(segments):
        raise MigrationError(
            f"cross section {source} segment {selected_index} is outside "
            f"0..{len(segments) - 1}"
        )
    return segments[selected_index]


def _validate_zero_energy_support(
    source: Path,
    converted: list[dict[str, float]],
    required: bool,
) -> None:
    if required and converted[0]["energy_eV"] != 0.0:
        raise MigrationError(
            f"momentum cross section {source} must explicitly cover 0 eV; "
            "migration does not extrapolate missing low-energy data"
        )
