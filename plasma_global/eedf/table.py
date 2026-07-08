from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.eedf.base import EEDFRequest, EEDFResult, EEDFTransport
from plasma_global.eedf.swarm_base import SwarmModel


@dataclass(frozen=True)
class RateTableLookupInfo:
    grid_column: str
    lookup_mode: str
    lookup_value: float
    lookup_clipped_value: float
    axis_min: float
    axis_max: float
    lookup_clipped: bool
    lookup_clipped_low: bool
    lookup_clipped_high: bool


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

    If no table is supplied, this backend fails fast. Use the explicit
    `maxwell` backend for cheap smoke or development runs.
    """

    def prepare(self, mechanism, chamber, run_config, resolved_paths, swarm_config=None) -> None:
        super().prepare(mechanism, chamber, run_config, resolved_paths, swarm_config)
        self.table_path = None
        self.lookup_mode = str(swarm_config.closure or 'auto').lower() if swarm_config else 'auto'
        self.bounds_policy = 'clip'
        if swarm_config is not None:
            table_cfg = swarm_config.table
            path = getattr(table_cfg, 'file', None)
            table_lookup = getattr(table_cfg, 'lookup', None)
            self.bounds_policy = str(getattr(table_cfg, 'bounds_policy', 'clip') or 'clip').strip().lower()
            if self.bounds_policy not in {'clip', 'error'}:
                raise ValueError("swarm.table.bounds_policy must be 'clip' or 'error'")
            if table_lookup:
                self.lookup_mode = str(table_lookup).lower()
            if path:
                self.table_path = Path(path)
                if not self.table_path.is_absolute():
                    self.table_path = Path(resolved_paths.chemistry_dir).joinpath(path).resolve()
        if self.table_path is None:
            raise ValueError('table swarm model requires swarm.table.file.')
        if not self.table_path.exists():
            raise FileNotFoundError(f'Rate table file not found: {self.table_path}')
        self._load_table(self.table_path)
        self._validate_required_rate_coefficients(mechanism)

    def _load_table(self, path: Path) -> None:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError('The table swarm model requires the optional h5py dependency. Install plasma-global-model[io].') from exc

        with h5py.File(path, 'r') as h5:
            self.mean_energy = self._read_1d_dataset(h5, 'mean_energy_eV')
            self.mobility = self._read_1d_dataset(h5, 'mobility_m2_V_s')
            self.diffusion = self._read_1d_dataset(h5, 'diffusion_m2_s')
            self.eff_field = self._read_1d_dataset(h5, 'effective_field_Td')
            n_grid = self.mean_energy.size
            for name, arr in {
                'mobility_m2_V_s': self.mobility,
                'diffusion_m2_s': self.diffusion,
                'effective_field_Td': self.eff_field,
            }.items():
                if arr.size != n_grid:
                    raise ValueError(f'table HDF5 dataset {name!r} length {arr.size} does not match mean_energy_eV length {n_grid}.')
            if 'rate_coefficients' not in h5:
                raise ValueError('table HDF5 file requires a rate_coefficients group.')
            self.k_tables = {}
            for name, ds in h5['rate_coefficients'].items():
                arr = np.asarray(ds[:], dtype=float)
                if arr.ndim != 1 or arr.size != n_grid:
                    raise ValueError(
                        f'table HDF5 rate_coefficients/{name} must be a 1D dataset with length {n_grid}.'
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f'table HDF5 rate_coefficients/{name} contains non-finite values.')
                self.k_tables[name] = arr
            self.grid_column = str(h5.attrs.get('grid_column', 'mean_energy_eV'))

    @staticmethod
    def _read_1d_dataset(h5, name: str) -> np.ndarray:
        if name not in h5:
            raise ValueError(f'table HDF5 file requires dataset {name!r}.')
        arr = np.asarray(h5[name][:], dtype=float)
        if arr.ndim != 1 or arr.size < 1:
            raise ValueError(f'table HDF5 dataset {name!r} must be a non-empty 1D array.')
        if not np.all(np.isfinite(arr)):
            raise ValueError(f'table HDF5 dataset {name!r} contains non-finite values.')
        return arr

    def _validate_required_rate_coefficients(self, mechanism) -> None:
        required: set[str] = set()
        rate_models = getattr(mechanism, 'rate_models', {}) or {}
        for reaction in getattr(mechanism, 'gas_reactions', []) or []:
            if not getattr(reaction, 'enabled', True):
                continue
            model = rate_models.get(getattr(reaction, 'rate_model_key', ''), {})
            if str(model.get('backend', '')).lower() != 'electron_impact_xsec':
                continue
            cs_id = model.get('cross_section_id')
            if cs_id:
                required.add(str(cs_id))
        missing = sorted(required.difference(self.k_tables))
        if missing:
            have = ', '.join(sorted(self.k_tables)) or '<none>'
            need = ', '.join(missing)
            raise ValueError(
                f'table HDF5 file {self.table_path} is missing rate_coefficients for required cross_section_id(s): {need}. '
                f'Available rate_coefficients: {have}'
            )

    @staticmethod
    def _interp(axis: np.ndarray, values: np.ndarray, x0: float) -> float:
        order = np.argsort(axis)
        x = np.asarray(axis[order], dtype=float)
        y = np.asarray(values[order], dtype=float)
        x_clip = float(np.clip(x0, x[0], x[-1]))
        return float(np.interp(x_clip, x, y))

    def _lookup_info(self, *, lookup: str, axis: np.ndarray, x0: float) -> RateTableLookupInfo:
        axis_min = float(np.nanmin(axis))
        axis_max = float(np.nanmax(axis))
        clipped_value = float(np.clip(x0, axis_min, axis_max))
        clipped_low = bool(x0 < axis_min)
        clipped_high = bool(x0 > axis_max)
        return RateTableLookupInfo(
            grid_column=str(self.grid_column),
            lookup_mode=lookup,
            lookup_value=float(x0),
            lookup_clipped_value=clipped_value,
            axis_min=axis_min,
            axis_max=axis_max,
            lookup_clipped=bool(clipped_low or clipped_high),
            lookup_clipped_low=clipped_low,
            lookup_clipped_high=clipped_high,
        )

    def _check_bounds_policy(self, *, request: EEDFRequest, lookup_info: RateTableLookupInfo) -> None:
        if self.bounds_policy != 'error' or not lookup_info.lookup_clipped:
            return
        raise ValueError(
            'table lookup outside table bounds: '
            f'zone_id={request.zone_id}, '
            f'lookup_mode={lookup_info.lookup_mode}, '
            f'value={lookup_info.lookup_value}, '
            f'axis_min={lookup_info.axis_min}, '
            f'axis_max={lookup_info.axis_max}, '
            f'table_path={self.table_path}'
        )

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
            lookup_info = self._lookup_info(lookup=lookup, axis=axis, x0=x0)
            self._check_bounds_policy(request=request, lookup_info=lookup_info)
            k_map = {cs_id: self._interp(axis, table, x0) for cs_id, table in self.k_tables.items()}
        else:
            axis = self.mean_energy
            x0 = eps
            lookup = 'mean_energy'
            lookup_info = self._lookup_info(lookup=lookup, axis=axis, x0=x0)
            self._check_bounds_policy(request=request, lookup_info=lookup_info)
            k_map = {cs_id: self._interp(axis, table, x0) for cs_id, table in self.k_tables.items()}
        return EEDFResult(
            rate_coefficients=k_map,
            transport=EEDFTransport(
                mean_energy_eV=self._interp(axis, self.mean_energy, x0),
                mobility_m2_V_s=self._interp(axis, self.mobility, x0),
                diffusion_m2_s=self._interp(axis, self.diffusion, x0),
                effective_field_Td=self._interp(axis, self.eff_field, x0),
                lookup_mode=lookup,
            ),
            metadata={'table_lookup': lookup_info},
        )

    def provenance(self) -> dict[str, Any]:
        return {
            'backend': 'table',
            'table_path': str(self.table_path),
            'grid_column': str(self.grid_column),
            'rate_ids': sorted(self.k_tables),
        }
