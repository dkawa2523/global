from __future__ import annotations

from pathlib import Path
from typing import Any

import csv
import yaml


def write_solution_h5(path: str | Path, solution, state_labels: list[str] | None = None) -> None:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError('Writing solution.h5 requires the optional h5py dependency. Install plasma-global-model[io].') from exc

    path = Path(path)
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('t_s', data=solution.t)
        h5.create_dataset('y', data=solution.y)
        if state_labels is not None:
            h5.create_dataset('state_labels', data=[s.encode('utf-8') for s in state_labels])
        for k, v in solution.diagnostics.items():
            if v is None:
                continue
            try:
                h5.attrs[k] = v
            except TypeError:
                pass


def write_observables_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    if not records:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for rec in records:
        for key in rec.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def write_summary_yaml(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    with path.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(summary, fh, sort_keys=False, allow_unicode=True)
