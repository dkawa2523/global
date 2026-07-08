from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from plasma_global.observables.adapter import compute_observable_record
from plasma_global import build_case, load_case_from_yaml


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding='utf-8')
    return path


def _write_case(tmp_path: Path, *, extensions_block: str = '') -> Path:
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
        f"""
        species_file: species.csv
        gas_reactions_file: gas_reactions.csv
        surface_reactions_file: surface_reactions.csv
        model_files:
          gas_rate: gas_rate_models.yaml
        {extensions_block}
        """,
    )
    _write(
        tmp_path / 'chamber.yaml',
        """
        chamber_id: extension_test
        zones:
          - zone_id: plasma
            volume_m3: 1.0
            pressure_Pa: 10.0
            gas_temperature_K: 300.0
            initial_densities_m3:
              Ar: 2.0e20
              Ar_plus: 1.0e12
        edges: []
        surfaces:
          - surface_id: wafer
            zone_id: plasma
            kind: wafer
            area_m2: 0.01
            material: Si
            temperature_K: 300.0
            site_density_m2: 1.0e18
            models: {}
        gas_inlets: []
        pumps: []
        power_ports: []
        """,
    )
    _write(
        tmp_path / 'recipe.yaml',
        """
        recipe_id: extension_test
        steps:
          - step_id: transient
            t_start_s: 0.0
            t_end_s: 1.0e-8
            gas_inlets: {}
            power_ports: {}
        """,
    )
    return _write(
        tmp_path / 'case.yaml',
        """
        case:
          name: extension_test
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


def test_manifest_without_extensions_keeps_empty_extension_lists(tmp_path: Path) -> None:
    loaded = load_case_from_yaml(_write_case(tmp_path))

    assert loaded.mechanism.state_variables == []
    assert loaded.mechanism.processes == []


def test_extra_states_processes_labels_and_observables(tmp_path: Path) -> None:
    chemistry = tmp_path / 'chemistry'
    _write(
        chemistry / 'state_variables.yaml',
        """
        state_variables:
          zone_marker:
            scope: zone
            unit: m-3
            initial: 2.0
            lower_bound: 0.0
            scale: 10.0
            output: true
          hidden_zone:
            scope: zone
            initial: 4.0
            output: false
          wall_charge:
            scope: surface
            unit: C_m2
            initial: -1.0
            lower_bound: null
            output: true
            surfaces: [wafer]
        """,
    )
    _write(
        chemistry / 'processes.yaml',
        """
        processes:
          zone_source:
            kind: source
            target: zone_marker
            value: 3.0
            zones: [plasma]
          zone_relaxation:
            kind: relaxation
            target: hidden_zone
            tau_s: 2.0
            equilibrium: 10.0
          wall_charge_from_flux:
            kind: ion_flux_source
            target: wall_charge
            coefficient: 1.0
            surfaces: [wafer]
        """,
    )
    case_path = _write_case(
        tmp_path,
        extensions_block="""
        extensions:
          state_variables: state_variables.yaml
          processes: processes.yaml
        """,
    )

    built = build_case(load_case_from_yaml(case_path))
    system = built.system
    y0 = system.initial_state()
    dydt = system.rhs(0.0, y0)

    assert set(system.state_layout.extra_state_index) == {'zone_marker', 'hidden_zone', 'wall_charge'}
    assert system.state_labels()[-3:] == ['extra[zone_marker,plasma]', 'extra[hidden_zone,plasma]', 'extra[wall_charge,wafer]']
    assert y0[system.state_layout.extra_state_index['zone_marker']['plasma']] == pytest.approx(2.0)
    assert y0[system.state_layout.extra_state_index['wall_charge']['wafer']] == pytest.approx(-1.0)
    assert dydt[system.state_layout.extra_state_index['zone_marker']['plasma']] == pytest.approx(3.0)
    assert dydt[system.state_layout.extra_state_index['hidden_zone']['plasma']] == pytest.approx(3.0)
    assert dydt[system.state_layout.extra_state_index['wall_charge']['wafer']] > 0.0

    record = compute_observable_record(system, 0.0, y0)

    assert record['extra_zone_marker_plasma'] == pytest.approx(2.0)
    assert record['extra_wall_charge_wafer'] == pytest.approx(-1.0)
    assert 'extra_hidden_zone_plasma' not in record
    assert np.isfinite(dydt).all()


def test_extension_validation_reports_bad_process_and_owner(tmp_path: Path) -> None:
    chemistry = tmp_path / 'chemistry'
    _write(
        chemistry / 'state_variables.yaml',
        """
        state_variables:
          wall_charge:
            scope: surface
            initial: 0.0
            surfaces: [missing_surface]
        """,
    )
    _write(
        chemistry / 'processes.yaml',
        """
        processes:
          bad_process:
            kind: python_expr
            target: wall_charge
        """,
    )
    case_path = _write_case(
        tmp_path,
        extensions_block="""
        extensions:
          state_variables: state_variables.yaml
          processes: processes.yaml
        """,
    )

    with pytest.raises(ValueError) as exc:
        load_case_from_yaml(case_path)

    message = str(exc.value)
    assert 'STATE_VARIABLE_UNKNOWN_SURFACE' in message
    assert 'PROCESS_KIND_UNSUPPORTED' in message
