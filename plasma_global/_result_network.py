"""Graphviz incidence networks: species -> reaction event -> species."""

from __future__ import annotations

import importlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from plasma_global.core.result import SimulationResult, to_plain_mapping
from plasma_global.input._schema_output import NetworkConfig


def graphviz_module(*, render: bool = True):
    try:
        module = importlib.import_module("graphviz")
    except ImportError as exc:
        raise RuntimeError(
            "Networks require the optional dependency: pip install .[graph]"
        ) from exc
    if render and shutil.which("dot") is None:
        raise RuntimeError(
            "Rendering networks requires Graphviz's dot executable on PATH; "
            "DOT export only needs .[graph]"
        )
    return module


def _matches_selection(event: Mapping[str, Any], config: NetworkConfig) -> bool:
    region_matches = (
        event["domain"] in config.domains
        and (not config.zones or event["zone"] in config.zones)
        and (
            not config.surfaces
            or (event["domain"] != "gas" and event["region"] in config.surfaces)
        )
    )
    species_matches = not config.species or bool(
        set(config.species) & (set(event["reactants"]) | set(event["products"]))
    )
    return region_matches and species_matches


def _selected_events(
    result: SimulationResult, config: NetworkConfig
) -> list[dict[str, Any]]:
    definitions = result.metadata.get("reaction_network")
    if definitions is None:
        raise ValueError(
            "this result contains no reaction topology; rerun with the current package"
        )
    geometry = result.metadata.get("model_geometry", {})
    if unknown := set(config.species) - set(geometry.get("species", {})):
        raise ValueError(f"network references unknown species {sorted(unknown)}")
    if set(config.zones) - set(geometry.get("zones", {})) or set(config.surfaces) - set(
        geometry.get("surfaces", {})
    ):
        raise ValueError("network references unknown zones or surfaces")
    chosen = [
        to_plain_mapping(event)
        for event in definitions
        if _matches_selection(event, config)
    ]
    if not chosen:
        raise ValueError("network selection contains no modeled reaction events")
    return chosen


def _sample_index(result: SimulationResult, time_s: float | None) -> int:
    if not result.n_times:
        raise ValueError("an event network requires saved time points")
    if time_s is None:
        return result.n_times - 1
    matches = np.flatnonzero(
        np.abs(result.time_s - time_s) <= 8 * abs(np.spacing(time_s))
    )
    if matches.size != 1:
        raise ValueError(
            f"network time_s={time_s:g} must match a saved time; "
            "set solver.save_at_s or sample_interval_s"
        )
    return int(matches[0])


def _event_weights(
    result: SimulationResult, events: list[dict[str, Any]], field: str, index: int
) -> None:
    for event in events:
        name = event[field]
        if name not in result.observables:
            raise ValueError(
                f"event data {name!r} was not saved; "
                "enable output.reaction_rates or an event network and rerun"
            )
        event["value"] = float(result.series(name)[index])


def _filter_events(
    events: list[dict[str, Any]], fraction: float
) -> tuple[list[dict[str, Any]], list[str]]:
    maximum = max((event.get("value", 0.0) for event in events), default=0.0)
    selected = [
        event for event in events if event.get("value", maximum) >= fraction * maximum
    ]
    selected_ids = {event["id"] for event in selected}
    return selected, [
        event["id"] for event in events if event["id"] not in selected_ids
    ]


def network_data(result: SimulationResult, config: NetworkConfig) -> dict[str, Any]:
    events = _selected_events(result, config)
    time_s = None
    units = None
    if config.mode == "events":
        index = _sample_index(result, config.time_s)
        time_s = float(result.time_s[index])
        field = "count_series" if config.measure == "cumulative" else "frequency_series"
        units = "events" if config.measure == "cumulative" else "events s^-1"
        _event_weights(result, events, field, index)
    selected, omitted = _filter_events(events, config.min_fraction)
    return {
        "mode": config.mode,
        "selection": config.model_dump(mode="json"),
        "measure": config.measure if config.mode == "events" else None,
        "time_s": time_s,
        "unit": units,
        "min_fraction_of_maximum": config.min_fraction,
        "events": selected,
        "omitted_event_ids": omitted,
        "edge_meaning": (
            "stoichiometric participation, not inferred pairwise atom-transfer pathways"
        ),
        "event_integration": to_plain_mapping(
            result.metadata.get("event_integration", {})
        ),
        "observable_time_convention": result.metadata.get("observable_time_convention"),
        "spatial_interpretation": (
            "gas events are zone averages; "
            "surface and boundary events refer to whole modeled surfaces"
        ),
    }


def _species_location(
    result: SimulationResult, event: Mapping[str, Any], species: str
) -> str:
    definition = (
        result.metadata.get("model_geometry", {}).get("species", {}).get(species, {})
    )
    return event["region"] if definition.get("phase") == "surface" else event["zone"]


def network_graph(result: SimulationResult, data: Mapping[str, Any]):
    gv = graphviz_module(render=False)
    graph = gv.Digraph("reaction_events", engine="dot")
    description = (
        "Reaction mechanism"
        if data["mode"] == "mechanism"
        else (
            f"{data['measure']} reaction events "
            f"at t={data['time_s']:.6g} s [{data['unit']}]"
        )
    )
    if data["mode"] == "events" and data["measure"] == "cumulative":
        description = (
            "Estimated " + description + "\nSaved-grid quadrature; see JSON for method"
        )
    graph.attr(
        rankdir="LR", labelloc="t", label=gv.escape(description), fontname="sans-serif"
    )
    graph.attr("node", fontname="sans-serif", fontsize="11")
    graph.attr("edge", fontname="sans-serif", fontsize="9")
    species_nodes: dict[tuple[str, str], str] = {}
    maximum = max((event.get("value", 0.0) for event in data["events"]), default=0.0)
    for index, event in enumerate(data["events"]):
        node = f"reaction_{index}"
        label = f"{event['reaction_id']}\n{event['domain']} @ {event['region']}"
        if "value" in event:
            label += f"\n{event['value']:.4g} {data['unit']}"
        graph.node(
            node,
            label=gv.escape(label),
            shape="box",
            style="rounded,filled",
            fillcolor="#e8eff8",
        )
        width = 1.0 + (
            4.0 * math.sqrt(event.get("value", 0.0) / maximum) if maximum > 0 else 0.0
        )
        _event_edges(graph, gv, result, event, node, species_nodes, width)
    if data["omitted_event_ids"]:
        graph.node(
            "filter_note",
            label=(
                f"{len(data['omitted_event_ids'])} events omitted "
                "by explicit threshold; see JSON"
            ),
            shape="note",
        )
    return graph


def _event_edges(
    graph: Any,
    gv: Any,
    result: SimulationResult,
    event: Mapping[str, Any],
    node: str,
    species_nodes: dict[tuple[str, str], str],
    width: float,
) -> None:
    for side in ("reactants", "products"):
        for species, amount in event[side].items():
            region = _species_location(result, event, species)
            key = (region, species)
            if key not in species_nodes:
                identity = f"species_{len(species_nodes)}"
                species_nodes[key] = identity
                graph.node(
                    identity, label=gv.escape(f"{species}\n@ {region}"), shape="ellipse"
                )
            source, target = (
                (species_nodes[key], node)
                if side == "reactants"
                else (node, species_nodes[key])
            )
            graph.edge(source, target, label=f"{amount:g}", penwidth=f"{width:.3f}")


def write_networks(
    result: SimulationResult, directory: Path, requests: Mapping[str, NetworkConfig]
) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for name, config in requests.items():
        data = network_data(result, config)
        graph = network_graph(result, data)
        dot = directory / f"{name}.dot"
        dot.write_text(graph.source, encoding="utf-8")
        record = directory / f"{name}.json"
        record.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths.extend((dot, record))
        if config.format != "dot":
            gv = graphviz_module(render=True)
            image = directory / f"{name}.{config.format}"
            try:
                image.write_bytes(graph.pipe(format=config.format))
            except gv.CalledProcessError as exc:
                raise RuntimeError(f"Graphviz rendering failed: {exc}") from exc
            paths.append(image)
    return tuple(paths)
