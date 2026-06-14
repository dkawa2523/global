from __future__ import annotations

from pathlib import Path

import numpy as np

from plasma_global.eedf.base import EEDFRequest, EEDFResult
from plasma_global.eedf.swarm_backend import SWARM_MODEL_REGISTRY, SwarmEEDFBackend
from plasma_global.eedf.swarm_base import SwarmModel


class TabulatedSwarmModel(SwarmModel):
    """Tabulated swarm properties backend.

    Expected HDF5 structure (minimal):
      /mean_energy_eV
      /rate_coefficients/<cross_section_id>
      /mobility_m2_V_s
      /diffusion_m2_s
      /effective_field_Td

    The table can be gridded by either `mean_energy_eV` or `EoverN_Td` /
    `effective_field_Td`. Field-gridded tables should be used with
    `swarm.closure: local_field`.

    If no table is supplied, this backend falls back to the analytic Maxwell
    closure so that the interface remains usable during development.
    """

    def prepare(self, mechanism, chamber, run_config, swarm_config=None) -> None:
        super().prepare(mechanism, chamber, run_config, swarm_config)
        self.table_path = None
        self.lookup_mode = str(getattr(swarm_config, 'closure', 'auto') or 'auto').lower() if swarm_config else 'auto'
        if swarm_config is not None:
            table_cfg = getattr(swarm_config, 'table', None)
            path = getattr(table_cfg, 'file', None)
            table_lookup = getattr(table_cfg, 'lookup', None)
            if table_lookup:
                self.lookup_mode = str(table_lookup).lower()
            if path:
                self.table_path = Path(path)
                if not self.table_path.is_absolute():
                    self.table_path = Path(run_config.paths.chemistry_dir).joinpath(path).resolve()
        if self.table_path is None:
            raise ValueError('rate_table EEDF backend requires swarm.table.file.')
        if not self.table_path.exists():
            raise FileNotFoundError(f'Rate table file not found: {self.table_path}')
        self._load_table(self.table_path)

    def _load_table(self, path: Path) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError('The rate_table EEDF backend requires the optional h5py dependency. Install plasma-global-model[io].') from exc

        with h5py.File(path, 'r') as h5:
            self.mean_energy = np.asarray(h5['mean_energy_eV'][:], dtype=float)
            self.mobility = np.asarray(h5['mobility_m2_V_s'][:], dtype=float)
            self.diffusion = np.asarray(h5['diffusion_m2_s'][:], dtype=float)
            self.eff_field = np.asarray(h5['effective_field_Td'][:], dtype=float)
            self.k_tables = {name: np.asarray(ds[:], dtype=float) for name, ds in h5['rate_coefficients'].items()}
            self.grid_column = str(h5.attrs.get('grid_column', 'mean_energy_eV'))

    @staticmethod
    def _interp_slope(x: np.ndarray, y: np.ndarray, x0: float) -> float:
        order = np.argsort(x)
        x = np.asarray(x[order], dtype=float)
        y = np.asarray(y[order], dtype=float)
        if x.size < 2:
            return 0.0
        if x0 <= x[0]:
            i = 0
        elif x0 >= x[-1]:
            i = x.size - 2
        else:
            i = max(int(np.searchsorted(x, x0)) - 1, 0)
        dx = max(float(x[i + 1] - x[i]), 1.0e-30)
        return float((y[i + 1] - y[i]) / dx)

    @staticmethod
    def _interp(axis: np.ndarray, values: np.ndarray, x0: float) -> float:
        order = np.argsort(axis)
        x = np.asarray(axis[order], dtype=float)
        y = np.asarray(values[order], dtype=float)
        x_clip = float(np.clip(x0, x[0], x[-1]))
        return float(np.interp(x_clip, x, y))

    def _use_field_lookup(self, request: EEDFRequest) -> bool:
        if self.lookup_mode in {'local_field', 'field', 'eovern', 'eovern_td'}:
            return request.reduced_field_Td is not None
        if self.lookup_mode in {'mean_energy', 'energy'}:
            return False
        return self.grid_column in {'EoverN_Td', 'effective_field_Td'} and request.reduced_field_Td is not None

    def evaluate(self, request: EEDFRequest) -> EEDFResult:
        eps = max(float(request.mean_energy_eV), 1.0e-3)
        if self._use_field_lookup(request):
            axis = self.eff_field
            x0 = max(float(request.reduced_field_Td or 0.0), 0.0)
            lookup = 'field'
            k_map = {cs_id: self._interp(axis, table, x0) for cs_id, table in self.k_tables.items()}
            dk_map = {cs_id: 0.0 for cs_id in self.k_tables}
        else:
            axis = self.mean_energy
            x0 = eps
            lookup = 'mean_energy'
            k_map = {cs_id: self._interp(axis, table, x0) for cs_id, table in self.k_tables.items()}
            dk_map = {cs_id: self._interp_slope(axis, table, x0) for cs_id, table in self.k_tables.items()}
        return EEDFResult(
            rate_coefficients=k_map,
            d_rate_d_mean_energy_eV=dk_map,
            transport={
                'mean_energy_eV': self._interp(axis, self.mean_energy, x0),
                'mobility_m2_V_s': self._interp(axis, self.mobility, x0),
                'diffusion_m2_s': self._interp(axis, self.diffusion, x0),
                'effective_field_Td': self._interp(axis, self.eff_field, x0),
                'swarm_model': 'table',
                'lookup_mode': lookup,
                'grid_column': self.grid_column,
            },
        )


class TableEEDFBackend(SwarmEEDFBackend):
    def __init__(self) -> None:
        super().__init__(forced_model_name='table')


SWARM_MODEL_REGISTRY.register('table', lambda **kwargs: TabulatedSwarmModel())
