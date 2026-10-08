"""Recursive deterministic IDS decomposition with explicit stop reasons."""

from __future__ import annotations

from dataclasses import dataclass

from .atoms import assign_vector_atoms
from .candidate import compile_stroke_seeds
from .domain import ComponentProgram, DecompositionNodeEvidence
from .grammar import GlyphRepository
from .solver import solve_root_partition, solve_surround_partition

SUPPORTED_BINARY_OPERATORS = frozenset({"⿰", "⿱", "⿸"})


@dataclass(frozen=True)
class DecompositionNode:
    program: ComponentProgram
    geometry: object
    evidence: DecompositionNodeEvidence
    children: tuple[DecompositionNode, ...] = ()
    stroke_regions: tuple[object, ...] = ()
    stroke_labels: tuple[str, ...] = ()


def _local_search_bounds(geometry, operator: str) -> tuple[float, float, float]:
    bounds = geometry.bounds
    low, high = (bounds[0], bounds[2]) if operator == "⿰" else (bounds[1], bounds[3])
    span = high - low
    inset = max(1.0, span * 0.08)
    step = max(0.25, span / 192.0)
    return low + inset, high - inset, step


def decompose_recursive(
    geometry,
    program: ComponentProgram,
    repository: GlyphRepository | None = None,
) -> DecompositionNode:
    if program.kind == "component":
        evidence = DecompositionNodeEvidence(
            glyphId=program.glyph_id,
            status="leaf",
            area=geometry.area,
        )
        return DecompositionNode(program, geometry, evidence)

    if program.operator not in SUPPORTED_BINARY_OPERATORS or len(program.children) != 2:
        evidence = DecompositionNodeEvidence(
            glyphId=program.glyph_id,
            status="unsupported-operator",
            operator=program.operator,
            area=geometry.area,
        )
        return DecompositionNode(program, geometry, evidence)

    try:
        if program.operator == "⿸":
            try:
                partition = solve_surround_partition(geometry, program)
            except ValueError:
                if repository is None:
                    raise
                seeds = compile_stroke_seeds(repository, program.glyph_id)
                groups = tuple(
                    frozenset(child.leaf_ids()) for child in program.children
                )
                partition = assign_vector_atoms(geometry, seeds, groups)
        else:
            minimum, maximum, step = _local_search_bounds(geometry, program.operator)
            partition = solve_root_partition(
                geometry,
                program,
                minimum=minimum,
                maximum=maximum,
                step=step,
            )
            if repository is not None and partition.evidence.crossing_ratio > 0.001:
                try:
                    seeds = compile_stroke_seeds(repository, program.glyph_id)
                    groups = tuple(
                        frozenset(child.leaf_ids()) for child in program.children
                    )
                    atom_partition = assign_vector_atoms(
                        geometry,
                        seeds,
                        groups,
                        minimum_seed_coverage=0.70,
                    )
                    if (
                        atom_partition.evidence.stroke_continuity >= 0.95
                        and atom_partition.evidence.stroke_independence >= 0.999999
                    ):
                        partition = atom_partition
                except ValueError:
                    pass
    except ValueError as error:
        evidence = DecompositionNodeEvidence(
            glyphId=program.glyph_id,
            status="infeasible",
            operator=program.operator,
            reason=str(error),
            area=geometry.area,
        )
        return DecompositionNode(program, geometry, evidence)

    children = tuple(
        decompose_recursive(child_geometry, child_program, repository)
        for child_geometry, child_program in zip(partition.children, program.children)
    )
    evidence = DecompositionNodeEvidence(
        glyphId=program.glyph_id,
        status="partitioned",
        operator=program.operator,
        area=geometry.area,
        partition=partition.evidence,
        children=tuple(child.evidence for child in children),
    )
    return DecompositionNode(
        program,
        geometry,
        evidence,
        children,
        partition.stroke_regions,
        partition.stroke_labels,
    )


def terminal_nodes(node: DecompositionNode) -> tuple[DecompositionNode, ...]:
    if not node.children:
        return (node,)
    return tuple(
        terminal for child in node.children for terminal in terminal_nodes(child)
    )
