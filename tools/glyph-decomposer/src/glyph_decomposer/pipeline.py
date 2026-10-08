"""End-to-end deterministic prototype pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain import DecompositionRequest, DecompositionResult
from .evaluation import evaluate_root_partition, evaluate_terminal_partition
from .geometry import polygon_parts
from .grammar import GlyphRepository
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells
from .recursive import DecompositionNode, decompose_recursive, terminal_nodes


@dataclass(frozen=True)
class DecompositionArtifacts:
    result: DecompositionResult
    source_geometry: object
    predicted_children: tuple[object, object]
    decomposition: DecompositionNode


def parse_unicode(value: str) -> int:
    normalized = value.upper().removeprefix("U+")
    return int(normalized, 16)


def decompose_with_artifacts(request: DecompositionRequest) -> DecompositionArtifacts:
    unicode_value = parse_unicode(request.unicode)
    cells = parse_cells(Path(request.bbox_path))
    cell = find_cell(cells, unicode_value, request.source)
    page_svg = export_page_svg(Path(request.pdf_path), cell.page)
    source = extract_cell_geometry(page_svg, cell)
    repository = GlyphRepository.load(Path(request.glyph_data_path))
    program = repository.compile(request.candidate_glyph_id)
    decomposition = decompose_recursive(source, program)
    if decomposition.evidence.partition is None or len(decomposition.children) != 2:
        raise ValueError("candidate root could not be partitioned")
    partition = decomposition.evidence.partition
    root_children = tuple(child.geometry for child in decomposition.children)
    terminals = terminal_nodes(decomposition)
    evaluation = None
    recursive_evaluation = None
    if request.annotation_path:
        evaluation = evaluate_root_partition(
            source,
            root_children,
            program,
            partition.axis,
            Path(request.annotation_path),
        )
        recursive_evaluation = evaluate_terminal_partition(
            source,
            terminals,
            partition.axis,
            Path(request.annotation_path),
        )
    return DecompositionArtifacts(
        result=DecompositionResult(
            unicode=f"U+{unicode_value:04X}",
            source=request.source,
            candidateGlyphId=request.candidate_glyph_id,
            page=cell.page,
            sourceArea=source.area,
            sourceValid=source.is_valid,
            sourceParts=polygon_parts(source),
            program=program,
            partition=partition,
            evaluation=evaluation,
            decomposition=decomposition.evidence,
            recursiveEvaluation=recursive_evaluation,
        ),
        source_geometry=source,
        predicted_children=root_children,
        decomposition=decomposition,
    )


def decompose(request: DecompositionRequest) -> DecompositionResult:
    return decompose_with_artifacts(request).result
