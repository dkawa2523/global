from __future__ import annotations

import re
from typing import Any


def observable_id(value: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', str(value)).strip('_')


def electrical_port_observable_fields(snapshot: Any, port_id: str) -> dict[str, float]:
    port_key = observable_id(port_id)
    values = dict(snapshot)
    return {
        f'electrical_{port_key}_{observable_id(name)}': float(value)
        for name, value in values.items()
    }


def table_lookup_observable_fields(lookup_info: Any, zone_id: str) -> dict[str, float | str | int]:
    zone_key = observable_id(zone_id)
    return {
        f'table_lookup_value_{zone_key}': float(lookup_info.lookup_value),
        f'table_lookup_clipped_value_{zone_key}': float(lookup_info.lookup_clipped_value),
        f'table_axis_min_{zone_key}': float(lookup_info.axis_min),
        f'table_axis_max_{zone_key}': float(lookup_info.axis_max),
        f'table_lookup_clipped_{zone_key}': int(lookup_info.lookup_clipped),
        f'table_lookup_clipped_low_{zone_key}': int(lookup_info.lookup_clipped_low),
        f'table_lookup_clipped_high_{zone_key}': int(lookup_info.lookup_clipped_high),
        f'table_lookup_mode_{zone_key}': str(lookup_info.lookup_mode),
        f'table_grid_column_{zone_key}': str(lookup_info.grid_column),
    }


def extra_state_observable_fields(system: Any, state: Any) -> dict[str, float]:
    fields: dict[str, float] = {}
    for spec in system.mechanism.state_variables:
        if not spec.output:
            continue
        state_key = observable_id(spec.state_id)
        for owner_id, idx in system.state_layout.extra_state_index.get(spec.state_id, {}).items():
            fields[f'extra_{state_key}_{observable_id(owner_id)}'] = float(state[idx])
    return fields
