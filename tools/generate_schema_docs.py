"""Generate deterministic documentation artifacts from the public input schema."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from plasma_global.input.schema import CaseSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "docs" / "generated"


def minimal_case_document() -> dict[str, object]:
    """Return the smallest self-contained schema-v4 case document.

    No external files or implicit species seeds are required.
    """

    return {
        "schema_version": 4,
        "case": {"name": "minimal_argon"},
        "chemistry": {
            "species": [
                {
                    "id": "e",
                    "charge": -1,
                    "mass_amu": 0.00054858,
                    "state_tags": ["electron"],
                },
                {"id": "Ar", "mass_amu": 39.948, "elements": {"Ar": 1.0}},
                {
                    "id": "Ar_plus",
                    "charge": 1,
                    "mass_amu": 39.948,
                    "elements": {"Ar": 1.0},
                },
            ],
        },
        "reactor": {
            "zones": {
                "plasma": {
                    "volume_m3": 0.01,
                    "initial": {
                        "pressure_Pa": 1.0,
                        "gas_temperature_K": 300.0,
                        "mole_fractions": {"Ar": 1.0},
                        "seed_densities_m3": {"Ar_plus": 1e12},
                        "mean_energy_eV": 3.0,
                    },
                }
            }
        },
        "models": {"electrons": {"kind": "energy_balance"}},
        "run": {"duration_s": 0.001},
    }


def render_schema_documents() -> dict[str, str]:
    """Render generated files without touching the filesystem."""

    minimal = minimal_case_document()
    CaseSpec.model_validate(minimal)
    schema_text = (
        json.dumps(
            CaseSpec.model_json_schema(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    example_text = yaml.safe_dump(
        minimal,
        sort_keys=False,
        allow_unicode=True,
    )
    return {
        "case.schema.json": schema_text,
        "minimal_case.yaml": example_text,
    }


def generate(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, ...]:
    """Write the generated schema and example, returning their paths."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, content in render_schema_documents().items():
        path = destination / name
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)


def main() -> int:
    for path in generate():
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
