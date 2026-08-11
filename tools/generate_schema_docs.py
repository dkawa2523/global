"""Generate deterministic documentation artifacts from the v3 input schema."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from plasma_global.input.schema import CaseSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = ROOT / "docs" / "generated"


def minimal_case_document() -> dict[str, object]:
    """Return the smallest useful schema-v3 case document.

    The referenced chemistry file is intentionally only a path placeholder;
    schema validation does not perform chemistry I/O.
    """

    return {
        "schema_version": 3,
        "case": {"name": "minimal_argon"},
        "chemistry": {"manifest": "chemistry/chemistry.yaml"},
        "reactor": {
            "chamber_id": "reactor",
            "zones": [
                {
                    "zone_id": "plasma",
                    "volume_m3": 0.01,
                    "pressure_Pa": 9.94067280414195,
                    "gas_temperature_K": 300.0,
                    "initial_densities_m3": {
                        "Ar": 2.4e21,
                        "Ar_plus": 1.0e12,
                    },
                    "initial_mean_energy_eV": 3.0,
                }
            ],
        },
        "recipe": {
            "recipe_id": "one_step",
            "steps": [{"step_id": "hold", "duration_s": 0.001}],
        },
        "models": {
            "electrons": {"kind": "maxwellian"},
            "electron_closure": {"kind": "electron_energy"},
            "electron_density": {"kind": "quasineutral"},
            "gas_energy": {"kind": "fixed"},
        },
        "solver": {"method": "BDF"},
        "output": {},
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
