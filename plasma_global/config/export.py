from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from plasma_global.config.models import ResolvedPaths, RunConfig


def write_effective_config(path: str | Path, run_config: RunConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(run_config.to_dict(), fh, sort_keys=False, allow_unicode=True)


def write_resolved_paths(path: str | Path, resolved_paths: ResolvedPaths) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(resolved_paths.to_dict(), fh, sort_keys=False, allow_unicode=True)
