"""Language-neutral domain records for decomposition requests and results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DecompositionRequest(StrictModel):
    schema_version: Literal[1] = Field(alias="schemaVersion")
    pdf_path: str = Field(alias="pdfPath")
    bbox_path: str = Field(alias="bboxPath")
    glyph_data_path: str = Field(alias="glyphDataPath")
    unicode: str
    source: str
    candidate_glyph_id: int = Field(alias="candidateGlyphId")
    annotation_path: str | None = Field(default=None, alias="annotationPath")


class ComponentProgram(StrictModel):
    glyph_id: int = Field(alias="glyphId")
    kind: Literal["component", "compound"]
    operator: str | None = None
    stroke_features: tuple[str, ...] = Field(default=(), alias="strokeFeatures")
    children: tuple[ComponentProgram, ...] = ()

    def leaf_ids(self) -> tuple[int, ...]:
        if self.kind == "component":
            return (self.glyph_id,)
        return tuple(leaf for child in self.children for leaf in child.leaf_ids())

    def stroke_count(self) -> int:
        if self.kind == "component":
            return len(self.stroke_features)
        return sum(child.stroke_count() for child in self.children)


class PartitionEvidence(StrictModel):
    axis: Literal["x", "y", "xy", "atoms"]
    cut: float | None
    secondary_cut: float | None = Field(default=None, alias="secondaryCut")
    score: float
    crossing_ratio: float = Field(alias="crossingRatio")
    balance_error: float = Field(alias="balanceError")
    reconstruction_error: float = Field(alias="reconstructionError")
    solver_status: str = Field(alias="solverStatus")
    seed_coverage: float | None = Field(default=None, alias="seedCoverage")
    alignment_iou: float | None = Field(default=None, alias="alignmentIoU")
    atom_count: int | None = Field(default=None, alias="atomCount")
    stroke_count: int | None = Field(default=None, alias="strokeCount")
    junction_count: int | None = Field(default=None, alias="junctionCount")
    stroke_continuity: float | None = Field(default=None, alias="strokeContinuity")
    stroke_independence: float | None = Field(default=None, alias="strokeIndependence")
    stroke_inertia: float | None = Field(default=None, alias="strokeInertia")
    orphan_ink_ratio: float | None = Field(default=None, alias="orphanInkRatio")


class EvaluationEvidence(StrictModel):
    truth_isolation: Literal[True] = Field(True, alias="truthIsolation")
    classified_ink_ratio: float = Field(alias="classifiedInkRatio")
    component_accuracy: float = Field(alias="componentAccuracy")
    child_ious: tuple[float, float] = Field(alias="childIoUs")
    mapping_mode: str = Field(alias="mappingMode")


class RecursiveEvaluationEvidence(StrictModel):
    truth_isolation: Literal[True] = Field(True, alias="truthIsolation")
    terminal_count: int = Field(alias="terminalCount")
    classified_ink_ratio: float = Field(alias="classifiedInkRatio")
    component_accuracy: float = Field(alias="componentAccuracy")
    terminal_ious: tuple[float, ...] = Field(alias="terminalIoUs")
    mean_iou: float = Field(alias="meanIoU")
    minimum_iou: float = Field(alias="minimumIoU")
    mapping_mode: str = Field(alias="mappingMode")


class DecompositionNodeEvidence(StrictModel):
    glyph_id: int = Field(alias="glyphId")
    status: Literal["leaf", "partitioned", "unsupported-operator", "infeasible"]
    operator: str | None = None
    reason: str | None = None
    area: float
    partition: PartitionEvidence | None = None
    children: tuple[DecompositionNodeEvidence, ...] = ()


class DecompositionResult(StrictModel):
    schema_version: Literal[1] = Field(1, alias="schemaVersion")
    unicode: str
    source: str
    candidate_glyph_id: int = Field(alias="candidateGlyphId")
    page: int
    source_area: float = Field(alias="sourceArea")
    source_valid: bool = Field(alias="sourceValid")
    source_parts: int = Field(alias="sourceParts")
    program: ComponentProgram
    partition: PartitionEvidence
    evaluation: EvaluationEvidence | None = None
    decomposition: DecompositionNodeEvidence | None = None
    recursive_evaluation: RecursiveEvaluationEvidence | None = Field(
        default=None, alias="recursiveEvaluation"
    )


ComponentProgram.model_rebuild()
DecompositionNodeEvidence.model_rebuild()
