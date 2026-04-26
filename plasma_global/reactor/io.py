from __future__ import annotations

from pathlib import Path

import yaml

from plasma_global.reactor.models import ChamberConfig, Edge, Inlet, PowerPort, Pump, RecipeConfig, RecipeStep, Surface, Zone


def _load_yaml(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as fh:
        return yaml.safe_load(fh) or {}


def _zone(d: dict) -> Zone:
    return Zone(
        zone_id=d['zone_id'],
        description=d.get('description', ''),
        volume_m3=float(d['volume_m3']),
        pressure_Pa=float(d['pressure_Pa']),
        gas_temperature_K=float(d['gas_temperature_K']),
        role=d.get('role', 'process'),
        initial_densities_m3={str(k): float(v) for k, v in (d.get('initial_densities_m3', {}) or {}).items()},
    )


def _edge(d: dict) -> Edge:
    return Edge(
        edge_id=d['edge_id'],
        from_zone=d['from_zone'],
        to_zone=d['to_zone'],
        conductance_m3_s=float(d['conductance_m3_s']),
        notes=d.get('notes', ''),
    )


def _surface(d: dict) -> Surface:
    return Surface(
        surface_id=d['surface_id'],
        zone_id=d['zone_id'],
        kind=d['kind'],
        area_m2=float(d['area_m2']),
        material=d.get('material', ''),
        temperature_K=float(d['temperature_K']),
        site_density_m2=float(d['site_density_m2']),
        initial_coverages={str(k): float(v) for k, v in (d.get('initial_coverages', {}) or {}).items()},
        initial_inventory={str(k): float(v) for k, v in (d.get('initial_inventory', {}) or {}).items()},
        models=d.get('models', {}) or {},
    )


def _inlet(d: dict) -> Inlet:
    return Inlet(
        inlet_id=d['inlet_id'],
        zone_id=d['zone_id'],
        flow_sccm={str(k): float(v) for k, v in (d.get('flow_sccm', {}) or {}).items()},
        temperature_K=float(d['temperature_K']),
    )


def _pump(d: dict) -> Pump:
    return Pump(
        pump_id=d['pump_id'],
        zone_id=d['zone_id'],
        speed_m3_s=float(d['speed_m3_s']),
        target_pressure_Pa=float(d['target_pressure_Pa']) if d.get('target_pressure_Pa') is not None else None,
    )


def _power_port(d: dict) -> PowerPort:
    return PowerPort(
        port_id=d['port_id'],
        kind=d['kind'],
        zone_id=d['zone_id'],
        coupling_target=d.get('coupling_target', ''),
        parameters=d.get('parameters', {}) or {},
    )


def load_chamber_config(path: str | Path) -> ChamberConfig:
    raw = _load_yaml(Path(path))
    return ChamberConfig(
        chamber_id=raw['chamber_id'],
        description=raw.get('description', ''),
        zones=[_zone(z) for z in raw.get('zones', [])],
        edges=[_edge(e) for e in raw.get('edges', [])],
        surfaces=[_surface(s) for s in raw.get('surfaces', [])],
        gas_inlets=[_inlet(i) for i in raw.get('gas_inlets', [])],
        pumps=[_pump(p) for p in raw.get('pumps', [])],
        power_ports=[_power_port(p) for p in raw.get('power_ports', [])],
        metadata=raw.get('metadata', {}),
    )


def _recipe_step(d: dict) -> RecipeStep:
    def _power_ports(obj: dict) -> dict:
        out = {}
        for k, v in (obj or {}).items():
            vv = dict(v)
            for kk in ['value_W', 'frequency_Hz', 'carrier_frequency_Hz', 'duty_cycle', 'repetition_Hz']:
                if kk in vv and vv[kk] is not None:
                    vv[kk] = float(vv[kk])
            out[str(k)] = vv
        return out

    def _gas(obj: dict) -> dict:
        return {str(k): {str(sk): float(sv) for sk, sv in (v or {}).items()} for k, v in (obj or {}).items()}

    return RecipeStep(
        step_id=d['step_id'],
        t_start_s=float(d['t_start_s']),
        t_end_s=float(d['t_end_s']),
        gas_inlets=_gas(d.get('gas_inlets', {})),
        power_ports=_power_ports(d.get('power_ports', {})),
        surface_overrides=d.get('surface_overrides', {}) or {},
        imported_inputs=d.get('imported_inputs', {}) or {},
    )


def load_recipe_config(path: str | Path) -> RecipeConfig:
    raw = _load_yaml(Path(path))
    return RecipeConfig(
        recipe_id=raw['recipe_id'],
        description=raw.get('description', ''),
        steps=[_recipe_step(s) for s in raw.get('steps', [])],
    )
