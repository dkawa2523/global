"""One-way v2 chemistry conversion to the canonical schema-v3 format."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from plasma_global.input import (
    _migrate_v2_chemistry_cross_sections as _cross_sections,
)
from plasma_global.input import _migrate_v2_chemistry_models as _models


def convert_v2_chemistry(
    source_manifest: str | Path,
    target_directory: str | Path,
    *,
    boundary_products: Mapping[str, str] | None = None,
    cross_section_segments: Mapping[str, int] | None = None,
    migration_warnings: list[str] | None = None,
) -> Path:
    """Convert one legacy manifest without adding legacy behavior to runtime."""

    source = Path(source_manifest).resolve()
    target = Path(target_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    manifest = _models.read_yaml(source)
    base = source.parent

    species_rows = _models.read_rows(base / str(manifest["species_file"]))
    species_out, cv_warning = _models.convert_species_rows(species_rows)
    _models.write_rows(
        target / "species.csv",
        [
            "id",
            "phase",
            "charge",
            "mass_amu",
            "elements",
            "state_tags",
            "surfaces",
            "display_name",
            "cv_over_kb",
        ],
        species_out,
    )
    if cv_warning is not None and migration_warnings is not None:
        migration_warnings.append(cv_warning)

    rate_models, energy_models = _models.convert_rate_models(manifest, base, target)
    _models.write_yaml(target / "rate_models.yaml", {"rate_models": rate_models})

    reaction_columns = [
        "id",
        "equation",
        "rate_model",
        "energy_loss_eV",
        "zones",
        "surfaces",
        "notes",
    ]
    gas_reactions = _models.convert_reaction_rows(
        base / str(manifest["gas_reactions_file"]),
        energy_models,
        surface=False,
    )
    _models.write_rows(target / "gas_reactions.csv", reaction_columns, gas_reactions)
    surface_file = manifest.get("surface_reactions_file")
    if surface_file:
        surface_reactions = _models.convert_reaction_rows(
            base / str(surface_file), energy_models, surface=True
        )
        _models.write_rows(
            target / "surface_reactions.csv",
            reaction_columns,
            surface_reactions,
        )

    boundary_rows = _models.convert_boundary_rows(species_rows, boundary_products)
    _models.write_rows(
        target / "boundary_reactions.csv",
        ["id", "equation", "zones", "surfaces"],
        boundary_rows,
    )

    cross_section_output = _cross_sections.publish_cross_sections(
        manifest.get("cross_sections_manifest"),
        base,
        target,
        cross_section_segments,
    )

    output_manifest = {
        "schema_version": 3,
        "species": "species.csv",
        "gas_reactions": "gas_reactions.csv",
        "boundary_reactions": "boundary_reactions.csv",
        **({"surface_reactions": "surface_reactions.csv"} if surface_file else {}),
        "rate_models": "rate_models.yaml",
        **({"cross_sections": cross_section_output} if cross_section_output else {}),
        "provenance": {"migrated_from": str(source)},
    }
    output = target / "chemistry.yaml"
    _models.write_yaml(output, output_manifest)
    return output


__all__ = ["convert_v2_chemistry"]
