from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

from plasma_global.chemistry.models import CrossSectionSpec

_ENERGY_SCALE = {
    'ev': 1.0,
}

_SIGMA_SCALE = {
    'm2': 1.0,
    'cm2': 1.0e-4,
}

_NUM_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?$')


def _as_float(token: str) -> float | None:
    token = token.strip().replace('D', 'E').replace('d', 'e')
    if not _NUM_RE.match(token):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_numeric_pair(line: str) -> tuple[float, float] | None:
    cleaned = line.strip()
    if not cleaned:
        return None
    cleaned = cleaned.split('#', 1)[0].split('//', 1)[0].strip()
    if not cleaned:
        return None
    for delim in [',', ';', '	']:
        cleaned = cleaned.replace(delim, ' ')
    toks = [tok for tok in cleaned.split() if tok]
    if len(toks) < 2:
        return None
    a = _as_float(toks[0])
    b = _as_float(toks[1])
    if a is None or b is None:
        return None
    return a, b


def _load_simple_xy(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            pair = _parse_numeric_pair(line)
            if pair is None:
                continue
            xs.append(pair[0])
            ys.append(pair[1])
    if len(xs) < 2:
        raise ValueError(f'No numeric cross-section data found in {path}')
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), {'reader': 'simple_xy'}


def _finalize_block(meta: dict[str, Any], pairs: list[tuple[float, float]], blocks: list[tuple[dict[str, Any], np.ndarray, np.ndarray]]) -> None:
    if len(pairs) < 2:
        return
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    blocks.append((dict(meta), x, y))


def _load_lxcat_text(path: Path, selector: str | None = None, dataset_index: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    blocks: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    meta: dict[str, Any] = {}
    pairs: list[tuple[float, float]] = []
    with path.open('r', encoding='utf-8') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                if pairs:
                    _finalize_block(meta, pairs, blocks)
                    pairs = []
                    meta = {}
                continue
            if set(line) <= {'-', '='}:
                if pairs:
                    _finalize_block(meta, pairs, blocks)
                    pairs = []
                    meta = {}
                continue
            pair = _parse_numeric_pair(line)
            if pair is not None:
                pairs.append(pair)
                continue
            if pairs:
                _finalize_block(meta, pairs, blocks)
                pairs = []
                meta = {}
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip().lower()] = v.strip()
            else:
                meta.setdefault('text', []).append(line)
    if pairs:
        _finalize_block(meta, pairs, blocks)
    if not blocks:
        raise ValueError(f'No cross-section blocks found in {path}')
    if dataset_index is not None:
        if dataset_index < 0 or dataset_index >= len(blocks):
            raise IndexError(f'dataset_index {dataset_index} out of range for {path}')
        meta_out, x, y = blocks[dataset_index]
        meta_out = dict(meta_out)
        meta_out['reader'] = 'lxcat_text'
        meta_out['block_index'] = dataset_index
        return x, y, meta_out
    if selector:
        selector_l = selector.lower()
        for idx, (meta_out, x, y) in enumerate(blocks):
            hay = ' '.join([f'{k}:{v}' for k, v in meta_out.items()]).lower()
            if selector_l in hay:
                meta_sel = dict(meta_out)
                meta_sel['reader'] = 'lxcat_text'
                meta_sel['block_index'] = idx
                return x, y, meta_sel
    meta_out, x, y = blocks[0]
    meta_out = dict(meta_out)
    meta_out['reader'] = 'lxcat_text'
    meta_out['block_index'] = 0
    return x, y, meta_out


def _convert_units(x: np.ndarray, y: np.ndarray, unit_energy: str, unit_sigma: str) -> tuple[np.ndarray, np.ndarray]:
    e_scale = _ENERGY_SCALE.get(str(unit_energy).lower(), None)
    s_scale = _SIGMA_SCALE.get(str(unit_sigma).lower(), None)
    if e_scale is None:
        raise ValueError(f'Unsupported energy unit: {unit_energy}')
    if s_scale is None:
        raise ValueError(f'Unsupported sigma unit: {unit_sigma}')
    return x * e_scale, y * s_scale


def _sanitize_curve(energy_eV: np.ndarray, sigma_m2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(energy_eV) & np.isfinite(sigma_m2)
    energy = np.asarray(energy_eV[mask], dtype=float)
    sigma = np.asarray(sigma_m2[mask], dtype=float)
    order = np.argsort(energy)
    energy = energy[order]
    sigma = np.clip(sigma[order], 0.0, None)
    uniq, idx = np.unique(energy, return_index=True)
    sigma = sigma[idx]
    energy = uniq
    if energy.size < 2:
        raise ValueError('Cross-section curve must contain at least two unique energies')
    return energy, sigma


def build_surrogate_cross_section(spec: CrossSectionSpec, n_energy: int = 256, energy_max_eV: float = 200.0) -> tuple[np.ndarray, np.ndarray]:
    eth = max(float(spec.threshold_eV), 0.05)
    p = max(float(spec.exponent), 0.05)
    energy = np.geomspace(1.0e-3, max(energy_max_eV, 5.0 * eth), n_energy)
    x = np.clip(energy / eth - 1.0, 0.0, None)
    sigma0 = float(spec.metadata.get('sigma_peak_m2', 3.0e-20))
    sigma = sigma0 * np.where(
        energy > eth,
        x ** p / np.maximum(1.0 + x, 1.5 + 0.25 * p) ** (1.5 + 0.3 * p),
        0.0,
    )
    if str(spec.kind).lower() in {'elastic', 'elastic_momentum', 'momentum_transfer', 'mt'}:
        sigma = sigma0 * (1.0 + energy / max(eth, 1.0)) ** -0.4
    return _sanitize_curve(energy, sigma)


def load_cross_section_curve(spec: CrossSectionSpec, base_dir: str | Path) -> CrossSectionSpec:
    base_dir = Path(base_dir)
    fmt = str(spec.format or '').lower()
    if spec.file:
        path = (base_dir / spec.file).resolve()
        if not path.exists():
            raise FileNotFoundError(f'Cross-section file not found: {path}')
        if fmt in {'csv_xy', 'simple_xy', 'simple_csv', 'tsv_xy', 'simple_tsv', ''}:
            energy, sigma, meta = _load_simple_xy(path)
        elif fmt in {'lxcat_text', 'lxcat', 'txt'}:
            energy, sigma, meta = _load_lxcat_text(path, selector=spec.selector, dataset_index=spec.dataset_index)
        else:
            raise ValueError(f'Unsupported cross-section format: {spec.format}')
        energy, sigma = _convert_units(energy, sigma, spec.unit_energy, spec.unit_sigma)
        energy, sigma = _sanitize_curve(energy, sigma)
        merged_meta = dict(spec.metadata)
        merged_meta.update(meta)
        return replace(spec, file=str(path), energy_eV=energy, sigma_m2=sigma, metadata=merged_meta)
    if str(spec.source or '').lower() == 'surrogate' or not spec.file:
        energy, sigma = build_surrogate_cross_section(spec)
        merged_meta = dict(spec.metadata)
        merged_meta.setdefault('reader', 'surrogate')
        return replace(spec, energy_eV=energy, sigma_m2=sigma, metadata=merged_meta)
    raise ValueError(f'Cross section {spec.cross_section_id} has neither file nor supported source')


def load_cross_sections_manifest(path: str | Path) -> dict[str, CrossSectionSpec]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as fh:
        manifest = yaml.safe_load(fh) or {}
    base_dir = path.parent
    out: dict[str, CrossSectionSpec] = {}
    for entry in manifest.get('cross_sections', []):
        data = dict(entry)
        metadata = data.pop('metadata', {}) or {}
        spec = CrossSectionSpec(metadata=metadata, **data)
        spec = load_cross_section_curve(spec, base_dir)
        if spec.energy_loss_eV is None:
            spec.energy_loss_eV = float(spec.threshold_eV)
        out[spec.cross_section_id] = spec
    return out
