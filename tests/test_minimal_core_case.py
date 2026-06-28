from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from plasma_global.workflows.context import build_case, load_case_from_yaml
from plasma_global.workflows import outputs as workflow_outputs
from plasma_global.workflows.runner import run_from_yaml
from plasma_global.workflows.solve import solve_built_case


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding='utf-8')
    return path


def _write_minimal_case(tmp_path: Path) -> Path:
    chemistry = tmp_path / 'chemistry'
    _write(
        chemistry / 'species.csv',
        """
        canonical_id,display_name,phase,charge,mass_amu,elements,aliases,state_tags,zones,surfaces
        e,e,gas,-1,0.00054858,,electron,electron,plasma,
        Ar,Ar,gas,0,39.948,Ar:1,argon,stable|parent,plasma,
        Ar_plus,Ar+,gas,1,39.948,Ar:1,Ar+|Ar(+),ion,plasma,
        """,
    )
    _write(
        chemistry / 'gas_reactions.csv',
        """
        reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes
        AR_ION,gas,e + Ar -> e + e + Ar_plus,RM_AR_ION,,plasma,,true,minimal constant ionization
        """,
    )
    _write(
        chemistry / 'surface_reactions.csv',
        """
        reaction_id,phase,equation,rate_model_key,energy_model_key,zone_filter,surface_filter,enabled,notes
        """,
    )
    _write(
        chemistry / 'gas_rate_models.yaml',
        """
        rate_models:
          RM_AR_ION:
            backend: constant
            value: 1.0e-20
        """,
    )
    _write(
        chemistry / 'chemistry_manifest.yaml',
        """
        species_file: species.csv
        gas_reactions_file: gas_reactions.csv
        surface_reactions_file: surface_reactions.csv
        model_files:
          gas_rate: gas_rate_models.yaml
        """,
    )

    _write(
        tmp_path / 'chamber.yaml',
        """
        chamber_id: minimal_single_zone
        zones:
          - zone_id: plasma
            volume_m3: 1.0
            pressure_Pa: 10.0
            gas_temperature_K: 300.0
            initial_densities_m3:
              Ar: 2.0e20
              Ar_plus: 1.0e12
        edges: []
        surfaces: []
        gas_inlets: []
        pumps: []
        power_ports: []
        """,
    )
    _write(
        tmp_path / 'recipe.yaml',
        """
        recipe_id: minimal_single_zone
        steps:
          - step_id: transient
            t_start_s: 0.0
            t_end_s: 1.0e-7
            gas_inlets: {}
            power_ports: {}
        """,
    )
    return _write(
        tmp_path / 'case.yaml',
        """
        case:
          name: minimal_single_zone_kinetics
          schema_version: 2
          kind: plasma_global_case

        files:
          chamber: chamber.yaml
          recipe: recipe.yaml
          chemistry:
            manifest: chemistry/chemistry_manifest.yaml
          output_dir: outputs

        runtime:
          export_effective_config: false
          export_resolved_paths: false

        physics:
          eedf_backend: maxwell
          electrical_backend: direct_power
          integrator: scipy_bdf
          enable_gas_temperature: false
          enable_surface_coverages: false
          enable_wall_inventory: false

        numerics:
          rtol: 1.0e-6
          atol: 1.0e-3
          first_step: 1.0e-10
          max_step: 1.0e-8
          positivity:
            clip_negative: true
            floor_density_m3: 1.0
            floor_energy_J_m3: 1.0e-30

        outputs:
          formats:
            solution_h5: false
            observables_csv: false
            summary_yaml: false
          plots:
            enabled: false
        """,
    )


def test_minimal_single_zone_kinetics_case_loads_builds_and_solves(tmp_path: Path) -> None:
    case_path = _write_minimal_case(tmp_path)

    loaded = load_case_from_yaml(case_path)
    built = build_case(loaded)
    solution = solve_built_case(built)

    assert loaded.run_config.case.name == 'minimal_single_zone_kinetics'
    assert built.system.surface_core.enabled is False
    assert set(built.state_layout.slices) == {'gas_densities', 'electron_energy'}
    assert solution.success is True
    assert solution.t.ndim == 1
    assert solution.t[-1] == pytest.approx(1.0e-7)
    assert solution.y.shape == (built.state_layout.size, solution.t.size)

    gas = solution.y[built.state_layout.slice('gas_densities'), :]
    assert np.all(np.isfinite(gas))
    assert np.all(gas >= 0.0)


def test_run_with_all_file_outputs_disabled_skips_observable_postprocessing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_path = _write_minimal_case(tmp_path)

    def _unexpected_observables(*_args, **_kwargs):
        raise AssertionError('observables should not be computed when no output needs them')

    monkeypatch.setattr(workflow_outputs, 'compute_observables', _unexpected_observables)

    result = run_from_yaml(case_path)

    assert result['summary']['success'] is True
    assert result['observables'] == []
    assert not (tmp_path / 'outputs').exists()
