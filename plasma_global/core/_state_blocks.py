"""Compiled ownership of state layout, initialization, domains and dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from plasma_global.core._compiled_jacobian import StateDependency, compile_dependencies
from plasma_global.core.domain import InitialState, RecipeSegment

if TYPE_CHECKING:
    from plasma_global.core.compiled import CompiledGlobalModel

StateKind = Literal[
    "density", "electron_energy", "heavy_energy", "coverage", "inventory"
]


@dataclass(frozen=True, slots=True)
class StateBlockLayout:
    kind: StateKind
    owner_id: str
    indices: tuple[int, ...]
    labels: tuple[str, ...]
    units: tuple[str, ...]


class StateDomain(Protocol):
    nonnegative: ClassVar[bool]

    def continue_trial(self, values: np.ndarray) -> None: ...

    def canonicalize(
        self,
        state: np.ndarray,
        tolerance: np.ndarray,
        segment: RecipeSegment,
        time_s: float,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class CompiledStateBlock:
    layout: StateBlockLayout
    initialize: Callable[[InitialState], np.ndarray]
    domain: StateDomain
    dependencies: tuple[StateDependency, ...]


def compile_state_blocks(model: CompiledGlobalModel) -> tuple[CompiledStateBlock, ...]:
    from plasma_global.core._initial_state import block_initializer
    from plasma_global.core._state_domain import block_domain

    dependencies = compile_dependencies(model)
    blocks: list[CompiledStateBlock] = []
    for layout in model.layout.blocks:
        rows = set(layout.indices)
        owned = tuple(
            StateDependency(
                tuple(row for row in item.rows if row in rows), item.columns
            )
            for item in dependencies
            if rows.intersection(item.rows)
        )
        blocks.append(
            CompiledStateBlock(
                layout,
                block_initializer(model, layout),
                block_domain(model, layout),
                owned,
            )
        )
    return tuple(blocks)


def jacobian_sparsity(size: int, blocks: tuple[CompiledStateBlock, ...]) -> csr_matrix:
    pattern = lil_matrix((size, size), dtype=bool)
    for block in blocks:
        for dependency in block.dependencies:
            pattern[np.ix_(dependency.rows, dependency.columns)] = True
    return csr_matrix(pattern)
