"""State-vector labels for solver outputs."""

from __future__ import annotations

from typing import Any


def state_labels(system: Any) -> list[str]:
    labels: list[str] = [''] * system.state_layout.size
    for zone_id, mapping in system.state_layout.gas_index.items():
        for sp_id, idx in mapping.items():
            labels[idx] = f'n[{zone_id},{sp_id}]'
    for zone_id, idx in system.state_layout.electron_energy_index.items():
        labels[idx] = f'We[{zone_id}]'
    for zone_id, idx in system.state_layout.gas_temperature_index.items():
        labels[idx] = f'Tg[{zone_id}]'
    for surface_id, mapping in system.state_layout.surface_index.items():
        for sp_id, idx in mapping.items():
            labels[idx] = f'theta[{surface_id},{sp_id}]'
    for surface_id, mapping in system.state_layout.inventory_index.items():
        for key, idx in mapping.items():
            labels[idx] = f'inventory[{surface_id},{key}]'
    for surface_id, idx in system.state_layout.film_index.items():
        labels[idx] = f'film[{surface_id}]'
    for state_id, mapping in system.state_layout.extra_state_index.items():
        for owner_id, idx in mapping.items():
            labels[idx] = f'extra[{state_id},{owner_id}]'
    return labels


__all__ = ['state_labels']
