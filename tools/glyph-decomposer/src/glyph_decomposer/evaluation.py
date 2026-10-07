"""Post-inference metrics. Human truth is never imported by solver modules."""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .domain import ComponentProgram, EvaluationEvidence


def _truth_polygons(path: Path) -> list[tuple[int | None, Polygon]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for annotation in payload.get("annotations", ()):
        if annotation.get("type") not in {"lasso", "polygon"}:
            continue
        points = annotation.get("fixedPoints") or annotation.get("points") or ()
        if len(points) < 3:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        try:
            label = int(annotation["label"])
        except (KeyError, TypeError, ValueError):
            label = None
        output.append((label, polygon))
    return output


def _ordered_truth_groups(
    truth: list[tuple[int | None, Polygon]],
    program: ComponentProgram,
    axis: str,
):
    child_leaves = [set(child.leaf_ids()) for child in program.children]
    labels = {label for label, _ in truth if label is not None}
    if all(leaves <= labels for leaves in child_leaves):
        groups = [
            unary_union([polygon for label, polygon in truth if label in leaves])
            for leaves in child_leaves
        ]
        return groups, "component-id"
    if len(truth) == 2:
        coordinate = (
            (lambda item: item[1].centroid.x)
            if axis == "x"
            else (lambda item: item[1].centroid.y)
        )
        ordered = sorted(truth, key=coordinate)
        return [ordered[0][1], ordered[1][1]], "spatial-two-component"
    raise ValueError("annotation polygons cannot be mapped to root children")


def evaluate_root_partition(
    source_geometry,
    predicted_children: tuple[object, object],
    program: ComponentProgram,
    axis: str,
    annotation_path: Path,
) -> EvaluationEvidence:
    truth = _truth_polygons(annotation_path)
    groups, mapping_mode = _ordered_truth_groups(truth, program, axis)
    truth_ink = [source_geometry.intersection(group) for group in groups]
    classified = unary_union(truth_ink)
    classified_area = classified.area
    if classified_area <= 0:
        raise ValueError("annotation does not overlap source ink")

    correct_area = sum(
        predicted.intersection(expected).area
        for predicted, expected in zip(predicted_children, truth_ink)
    )
    ious = []
    for predicted, expected in zip(predicted_children, truth_ink):
        union_area = predicted.union(expected).area
        ious.append(
            predicted.intersection(expected).area / union_area if union_area else 1.0
        )
    return EvaluationEvidence(
        truthIsolation=True,
        classifiedInkRatio=classified_area / source_geometry.area,
        componentAccuracy=correct_area / classified_area,
        childIoUs=(ious[0], ious[1]),
        mappingMode=mapping_mode,
    )
