from __future__ import annotations

import pytest
from types import SimpleNamespace

from plasma_global.eedf.base import EEDFRequest, EEDFTransport
from plasma_global.eedf.table import TabulatedSwarmModel
from scripts.build_rate_table_h5 import build_rate_table_h5


h5py = pytest.importorskip('h5py')


def _write_minimal_rate_table(path, rate_ids=('xs_ion',), n=2) -> None:
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('mean_energy_eV', data=[1.0, 2.0][:n])
        h5.create_dataset('effective_field_Td', data=[10.0, 20.0][:n])
        h5.create_dataset('mobility_m2_V_s', data=[0.1, 0.2][:n])
        h5.create_dataset('diffusion_m2_s', data=[0.2, 0.4][:n])
        group = h5.create_group('rate_coefficients')
        for rate_id in rate_ids:
            group.create_dataset(rate_id, data=[1.0e-16, 2.0e-16][:n])


def test_build_rate_table_h5_from_csv(tmp_path) -> None:
    table_dir = tmp_path / 'table'
    table_dir.mkdir()
    (table_dir / 'rates.csv').write_text(
        'mean_energy_eV,xs_ion,xs_exc\n'
        '1.0,1.0e-16,2.0e-16\n'
        '2.0,3.0e-16,4.0e-16\n',
        encoding='utf-8',
    )
    (table_dir / 'transport.csv').write_text(
        'mean_energy_eV,effective_field_Td,mobility_m2_V_s,diffusion_m2_s\n'
        '1.0,10.0,0.1,0.2\n'
        '2.0,20.0,0.2,0.4\n',
        encoding='utf-8',
    )

    out = build_rate_table_h5(table_dir, tmp_path / 'rates.h5')

    with h5py.File(out, 'r') as h5:
        assert list(h5['mean_energy_eV'][:]) == [1.0, 2.0]
        assert list(h5['rate_coefficients']['xs_ion'][:]) == [1.0e-16, 3.0e-16]
        assert h5.attrs['grid_column'] == 'mean_energy_eV'


def test_rate_table_backend_can_lookup_by_reduced_field(tmp_path) -> None:
    table_dir = tmp_path / 'field_table'
    table_dir.mkdir()
    (table_dir / 'rates.csv').write_text(
        'EoverN_Td,xs_ion,xs_exc\n'
        '10.0,1.0e-16,2.0e-16\n'
        '20.0,3.0e-16,6.0e-16\n',
        encoding='utf-8',
    )
    (table_dir / 'transport.csv').write_text(
        'EoverN_Td,mean_energy_eV,mobility_m2_V_s,diffusion_m2_s\n'
        '10.0,1.0,0.1,0.2\n'
        '20.0,2.0,0.2,0.4\n',
        encoding='utf-8',
    )
    out = build_rate_table_h5(table_dir, tmp_path / 'field_rates.h5')

    model = TabulatedSwarmModel()
    swarm = SimpleNamespace(closure='local_field', table=SimpleNamespace(file=str(out)))
    resolved_paths = SimpleNamespace(chemistry_dir=str(tmp_path))
    model.prepare(
        mechanism=SimpleNamespace(cross_sections={}),
        chamber=SimpleNamespace(),
        run_config=SimpleNamespace(),
        resolved_paths=resolved_paths,
        swarm_config=swarm,
    )
    result = model.evaluate(
        EEDFRequest(
            time_s=0.0,
            zone_id='plasma',
            composition={},
            electron_density_m3=1.0e16,
            mean_energy_eV=99.0,
            reduced_field_Td=15.0,
            gas_temperature_K=300.0,
            pressure_Pa=100.0,
        )
    )

    assert result.rate_coefficients['xs_ion'] == pytest.approx(2.0e-16)
    assert isinstance(result.transport, EEDFTransport)
    assert result.transport.mean_energy_eV == pytest.approx(1.5)
    assert result.transport.effective_field_Td == pytest.approx(15.0)
    assert result.transport.lookup_mode == 'field'


def test_field_rate_table_builder_keeps_sorted_field_axis_consistent(tmp_path) -> None:
    table_dir = tmp_path / 'descending_field_table'
    table_dir.mkdir()
    (table_dir / 'rates.csv').write_text(
        'EoverN_Td,xs_ion\n'
        '20.0,3.0e-16\n'
        '10.0,1.0e-16\n',
        encoding='utf-8',
    )
    (table_dir / 'transport.csv').write_text(
        'EoverN_Td,mean_energy_eV,mobility_m2_V_s,diffusion_m2_s\n'
        '20.0,2.0,0.2,0.4\n'
        '10.0,1.0,0.1,0.2\n',
        encoding='utf-8',
    )
    out = build_rate_table_h5(table_dir, tmp_path / 'field_rates.h5')

    with h5py.File(out, 'r') as h5:
        assert list(h5['effective_field_Td'][:]) == [10.0, 20.0]
        assert list(h5['mean_energy_eV'][:]) == [1.0, 2.0]
        assert list(h5['rate_coefficients']['xs_ion'][:]) == [1.0e-16, 3.0e-16]


def test_rate_table_fails_fast_when_required_cross_section_rate_is_missing(tmp_path) -> None:
    table_path = tmp_path / 'missing_required.h5'
    _write_minimal_rate_table(table_path, rate_ids=('xs_other',))
    model = TabulatedSwarmModel()
    mechanism = SimpleNamespace(
        gas_reactions=[SimpleNamespace(enabled=True, rate_model_key='rm_ion')],
        rate_models={'rm_ion': {'backend': 'electron_impact_xsec', 'cross_section_id': 'xs_ion'}},
    )
    swarm = SimpleNamespace(closure='mean_energy', table=SimpleNamespace(file=str(table_path)))
    resolved_paths = SimpleNamespace(chemistry_dir=str(tmp_path))

    with pytest.raises(ValueError, match='missing rate_coefficients.*xs_ion'):
        model.prepare(
            mechanism=mechanism,
            chamber=SimpleNamespace(),
            run_config=SimpleNamespace(),
            resolved_paths=resolved_paths,
            swarm_config=swarm,
        )


def test_rate_table_rejects_invalid_dataset_shape(tmp_path) -> None:
    table_path = tmp_path / 'bad_shape.h5'
    with h5py.File(table_path, 'w') as h5:
        h5.create_dataset('mean_energy_eV', data=[1.0, 2.0])
        h5.create_dataset('effective_field_Td', data=[10.0, 20.0])
        h5.create_dataset('mobility_m2_V_s', data=[0.1])
        h5.create_dataset('diffusion_m2_s', data=[0.2, 0.4])
        group = h5.create_group('rate_coefficients')
        group.create_dataset('xs_ion', data=[1.0e-16, 2.0e-16])
    model = TabulatedSwarmModel()
    swarm = SimpleNamespace(closure='mean_energy', table=SimpleNamespace(file=str(table_path)))
    resolved_paths = SimpleNamespace(chemistry_dir=str(tmp_path))

    with pytest.raises(ValueError, match='mobility_m2_V_s.*does not match'):
        model.prepare(
            mechanism=SimpleNamespace(rate_models={}, gas_reactions=[]),
            chamber=SimpleNamespace(),
            run_config=SimpleNamespace(),
            resolved_paths=resolved_paths,
            swarm_config=swarm,
        )
