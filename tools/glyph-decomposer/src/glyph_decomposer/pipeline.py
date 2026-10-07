"""End-to-end deterministic prototype pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .domain import DecompositionRequest, DecompositionResult
from .evaluation import evaluate_root_partition
from .geometry import polygon_parts
from .grammar import GlyphRepository
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells
from .solver import solve_root_partition


@dataclass(frozen=True)
class DecompositionArtifacts:
    result: DecompositionResult
    source_geometry: object
    predicted_children: tuple[object, object]


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
    partition = solve_root_partition(source, program)
    evaluation = None
    if request.annotation_path:
        evaluation = evaluate_root_partition(
            source,
            partition.children,
            program,
            partition.evidence.axis,
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
            partition=partition.evidence,
            evaluation=evaluation,
        ),
        source_geometry=source,
        predicted_children=partition.children,
    )


def decompose(request: DecompositionRequest) -> DecompositionResult:
    return decompose_with_artifacts(request).result
