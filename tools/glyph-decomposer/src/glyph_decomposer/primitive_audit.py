"""Blind primitive fitting audit with annotation truth revealed only afterward."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from .audit_cache import (
    PrimitiveAuditCache,
    array_identity,
    file_identity,
    stable_key,
)
from .candidate_graph import (
    CandidateComponent,
    CandidateGraph,
    compile_candidate_graph,
    compile_catalog_candidate_graph,
)
from .grammar import GlyphRepository
from .leaf_alignment import search_leaf_alignments, select_joint_leaf_alignments
from .leaf_route_selection import refine_suspicious_leaf_alignments
from .multiscale import certify_multiscale
from .pdf_svg import export_page_svg, extract_cell_geometry, find_cell, parse_cells
from .primitive_fit import fit_stroke_primitives, resample_polyline, symmetric_chamfer
from .primitive_route import enumerate_primitive_routes, select_joint_routes
from .skeleton import generate_skeleton, load_annotation_metadata
from .stroke_grammar import canonical_stroke_grammar
from .topology_certificate import certify_skeleton

_COLOURS = (
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#dc2626",
    "#0d9488",
    "#7c3aed",
)


def _compile_audit_candidate(
    repository: GlyphRepository,
    catalog: dict,
    codepoint: int,
    candidate_id: int,
) -> CandidateGraph:
    """Use reviewed geometry only when generic ternary intervals lose layout.

    Repository geometry is the stable default. Its ternary IDS intervals are
    deliberately generic, however, and can distort a narrow middle child of
    ``⿲``/``⿳``. In that one case the reviewed catalog preserves the actual
    candidate geometry and leaf hierarchy in a common coordinate system.
    """
    try:
        repository_candidate = compile_candidate_graph(repository, candidate_id)
    except (KeyError, ValueError):
        return compile_catalog_candidate_graph(catalog, codepoint, candidate_id)

    def contains_ternary(node) -> bool:
        return node.operator in {"⿲", "⿳"} or any(
            contains_ternary(child) for child in node.children
        )

    if contains_ternary(repository_candidate.root):
        try:
            return compile_catalog_candidate_graph(catalog, codepoint, candidate_id)
        except (KeyError, ValueError):
            pass
    return repository_candidate


def load_blind_case_specs(path: Path) -> list[dict]:
    """Load metadata-only audit cases without directed annotation truth."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(
            "blind cases must be a JSON array or an object with a cases array"
        )
    cases = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"blind case {index} must be an object")
        missing = [
            key
            for key in ("unicode", "source", "candidateGlyphId")
            if row.get(key) is None
        ]
        if missing:
            raise ValueError(f"blind case {index} is missing {', '.join(missing)}")
        raw_unicode = row["unicode"]
        codepoint = (
            int(raw_unicode.removeprefix("U+"), 16)
            if isinstance(raw_unicode, str)
            else int(raw_unicode)
        )
        source = str(row["source"])
        if len(source) not in {1, 2}:
            raise ValueError(f"blind case {index} has invalid source {source!r}")
        cases.append(
            {
                "unicode": codepoint,
                "source": source,
                "candidateGlyphId": int(row["candidateGlyphId"]),
            }
        )
    return cases


def _pixel_path(mask: np.ndarray) -> str:
    scale = 100 / mask.shape[0]
    return "".join(
        f"M{x * scale:.3f},{y * scale:.3f}h{scale:.3f}v{scale:.3f}h-{scale:.3f}z"
        for y, x in np.argwhere(mask)
    )


def _points(points) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


def _ids_segmentation(
    candidate: CandidateGraph,
    selected_leaf_alignments,
    fits: list[dict],
) -> dict:
    """Serialize the selected leaf geometry into the recursive candidate IDS.

    Candidate bounds describe the ideal repository glyph.  This view instead
    uses the final, route-refined leaf placements and unions them upward, so
    every compound rectangle reports where the blind algorithm actually put
    that subtree on the PDF skeleton.
    """
    alignments = dict(selected_leaf_alignments)
    fits_by_path: dict[tuple[int, ...], list[dict]] = {}
    serialized_by_path: dict[tuple[int, ...], dict] = {}
    for fit in fits:
        fits_by_path.setdefault(tuple(fit["componentPath"]), []).append(fit)

    def route_bounds(items: list[dict]) -> tuple[float, float, float, float] | None:
        points = [point for item in items for point in item["route"]]
        if not points:
            return None
        coordinates = np.asarray(points, dtype=float)
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        return (
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        )

    def union_bounds(children: list[dict]):
        bounds = [child["actualBounds"] for child in children if child["actualBounds"]]
        if not bounds:
            return None
        return [
            min(item[0] for item in bounds),
            min(item[1] for item in bounds),
            max(item[2] for item in bounds),
            max(item[3] for item in bounds),
        ]

    def serialize(node: CandidateComponent) -> dict:
        children = [serialize(child) for child in node.children]
        path = tuple(node.path)
        alignment = alignments.get(path)
        actual_bounds = (
            list(alignment.bounds)
            if alignment is not None
            else union_bounds(children)
            or list(route_bounds(fits_by_path.get(path, ())) or node.bounds)
        )
        serialized = {
            "glyphId": node.glyph_id,
            "componentPath": list(path),
            "kind": node.kind,
            "operator": node.operator,
            "candidateBounds": list(node.bounds),
            "actualBounds": actual_bounds,
            "strokeIndices": list(node.stroke_indices),
            "children": children,
        }
        serialized_by_path[path] = serialized
        return serialized

    root = serialize(candidate.root)
    leaf_paths = sorted({stroke.component_path for stroke in candidate.strokes})
    return {
        "root": root,
        "leaves": [
            {
                "componentPath": list(path),
                "glyphId": next(
                    stroke.leaf_id
                    for stroke in candidate.strokes
                    if stroke.component_path == path
                ),
                "colourIndex": index,
                "strokeIndices": [
                    item["strokeIndex"] for item in fits_by_path.get(path, ())
                ],
                "actualBounds": (
                    list(alignments[path].bounds)
                    if path in alignments
                    else serialized_by_path[path]["actualBounds"]
                ),
            }
            for index, path in enumerate(leaf_paths)
        ],
    }


def _truth_strokes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for annotation in payload.get("annotations", ()):
        if annotation.get("type") in {"lasso", "polygon"}:
            continue
        points = annotation.get("fixedPoints") or annotation.get("points") or ()
        output.append(
            {
                "type": annotation.get("type"),
                "label": annotation.get("label"),
                "points": _points(points),
            }
        )
    return output


def _truth_consistency(
    payload: dict,
    expected_features: list[str],
    expected_leaf_ids: list[int],
    truth: list[dict],
    candidate: CandidateGraph | None = None,
) -> dict:
    """Expose stale labels/order metadata before reporting evaluation metrics."""
    issues = []
    declared = payload.get("metadata", {}).get("candidateStrokeOrder")
    if declared is not None and list(declared) != expected_features:
        issues.append(
            {
                "kind": "candidate-stroke-order",
                "declared": list(declared),
                "expected": expected_features,
            }
        )
    if len(truth) != len(expected_features):
        issues.append(
            {
                "kind": "stroke-count",
                "actual": len(truth),
                "expected": len(expected_features),
            }
        )
    for index, (item, leaf_id) in enumerate(zip(truth, expected_leaf_ids)):
        label = item.get("label")
        if label is not None and str(label) != str(leaf_id):
            issues.append(
                {
                    "kind": "leaf-id",
                    "strokeIndex": index,
                    "actual": str(label),
                    "expected": str(leaf_id),
                }
            )

    if candidate is not None and len(truth) == len(candidate.strokes):
        issues.extend(_leaf_stroke_order_issues(candidate, truth))

    direction_warnings = []
    for index, (item, feature) in enumerate(zip(truth, expected_features)):
        points = np.asarray(item.get("points", ()), dtype=float)
        if len(points) < 2:
            continue
        dx, dy = points[-1] - points[0]
        adx, ady = abs(float(dx)), abs(float(dy))
        reason = None
        if feature == "横" and ady > 2 * max(adx, 1e-9):
            reason = "horizontal-candidate-has-vertical-truth-chord"
        elif feature == "竖" and adx > 2 * max(ady, 1e-9):
            reason = "vertical-candidate-has-horizontal-truth-chord"
        elif feature == "点" and adx < 0.1 * max(ady, 1e-9):
            reason = "dot-candidate-has-nearly-vertical-truth-chord"
        elif feature == "撇" and (dx >= 0 or dy <= 0):
            reason = "left-falling-candidate-has-incompatible-truth-direction"
        elif feature == "捺" and (dx <= 0 or dy <= 0):
            reason = "right-falling-candidate-has-incompatible-truth-direction"
        elif feature == "提" and (dx <= 0 or dy >= 0):
            reason = "rising-candidate-has-incompatible-truth-direction"
        if reason:
            direction_warnings.append(
                {
                    "strokeIndex": index,
                    "feature": feature,
                    "dx": float(dx),
                    "dy": float(dy),
                    "reason": reason,
                }
            )
    return {
        "valid": not issues,
        "issues": issues,
        "directionWarnings": direction_warnings,
        "corrections": payload.get("metadata", {}).get("truthCorrections", []),
    }


def _leaf_stroke_order_issues(
    candidate: CandidateGraph, truth: list[dict]
) -> list[dict]:
    """Find leaf-local permutations after removing translation and scale.

    Comparing raw candidate/PDF coordinates confuses ordinary affine layout
    differences with stroke-order errors. Each leaf occurrence is therefore
    normalized independently before a directed minimum-cost assignment. A
    large non-identity improvement is evidence that the paths were recorded
    in a different order, while similarly shaped neighboring strokes remain
    deliberately below the conservative threshold.
    """

    def normalize(polylines) -> list[np.ndarray]:
        arrays = [np.asarray(points, dtype=float) for points in polylines]
        all_points = np.vstack(arrays)
        minimum = all_points.min(axis=0)
        span = np.maximum(all_points.max(axis=0) - minimum, 1e-6)
        return [(points - minimum) / span for points in arrays]

    def directed_cost(first: np.ndarray, second: np.ndarray) -> float:
        first = resample_polyline(first, 41)
        second = resample_polyline(second, 41)
        endpoint = np.linalg.norm(first[0] - second[0]) + np.linalg.norm(
            first[-1] - second[-1]
        )
        return symmetric_chamfer(first, second) + 0.5 * endpoint

    indices_by_leaf: dict[tuple[tuple[int, ...], int], list[int]] = {}
    for stroke in candidate.strokes:
        key = stroke.component_path, stroke.leaf_id
        indices_by_leaf.setdefault(key, []).append(stroke.index)
    issues = []
    for (component_path, leaf_id), indices in indices_by_leaf.items():
        if len(indices) < 2:
            continue
        candidate_paths = normalize(
            [candidate.strokes[index].points for index in indices]
        )
        truth_paths = normalize([truth[index]["points"] for index in indices])
        costs = np.asarray(
            [
                [directed_cost(first, second) for second in truth_paths]
                for first in candidate_paths
            ]
        )
        rows, columns = linear_sum_assignment(costs)
        permutation = [None] * len(indices)
        for row, column in zip(rows, columns):
            permutation[int(row)] = int(column)
        identity_cost = float(np.mean(np.diag(costs)))
        optimum_cost = float(np.mean(costs[rows, columns]))
        improvement = identity_cost - optimum_cost
        if permutation != list(range(len(indices))) and improvement > 0.03:
            issues.append(
                {
                    "kind": "leaf-stroke-order",
                    "componentPath": list(component_path),
                    "leafId": leaf_id,
                    "strokeIndices": indices,
                    "permutation": permutation,
                    "identityCost": identity_cost,
                    "optimumCost": optimum_cost,
                }
            )
    return issues


def _minimum_distances(points: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    best = np.full(len(points), np.inf)
    for polyline in polylines:
        for start, end in pairwise(polyline):
            vector = end - start
            denominator = float(np.dot(vector, vector))
            if denominator <= 1e-12:
                distance = np.linalg.norm(points - start, axis=1)
            else:
                ratio = np.clip(((points - start) @ vector) / denominator, 0, 1)
                projection = start + ratio[:, None] * vector
                distance = np.linalg.norm(points - projection, axis=1)
            best = np.minimum(best, distance)
    return best


def _adaptive_leaf_alignment_solution(
    skeleton: np.ndarray,
    candidate,
    pools,
    solution,
    *,
    ids_trigger: float = 0.12,
    local_score_trigger: float = 2.5,
    search=None,
):
    """Expand only ambiguous leaf pools when the recursive IDS solve is strained."""
    if solution is None or solution.ids_structure_cost <= ids_trigger:
        return pools, solution, ()
    search = search or search_leaf_alignments
    selected = {item.component_path: item for item in solution.alignments}
    suspicious = sorted(
        (path for path, item in selected.items() if item.score >= local_score_trigger),
        key=lambda path: (-len(path), -selected[path].score, path),
    )
    working = dict(pools)
    best = solution
    refinements = []
    for path in suspicious:
        expanded = search(path, count=400, allow_extreme_scales=True)
        trial_pools = {**working, path: expanded}
        trial = select_joint_leaf_alignments(trial_pools, candidate, skeleton=skeleton)
        accepted = trial is not None and trial.score < best.score - 1e-9
        refinements.append(
            {
                "componentPath": list(path),
                "oldPoolSize": len(working[path]),
                "expandedPoolSize": len(expanded),
                "oldScore": best.score,
                "trialScore": trial.score if trial else None,
                "accepted": accepted,
            }
        )
        if accepted:
            working = trial_pools
            best = trial
            if best.ids_structure_cost <= ids_trigger:
                break
    return working, best, tuple(refinements)


def build_primitive_fit_cases(
    pdf_path: Path,
    bbox_path: Path,
    glyph_data_path: Path,
    candidate_catalog_path: Path,
    annotation_paths: list[Path] | None = None,
    *,
    blind_cases: list[dict] | None = None,
    size: int = 256,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Fit first, then load directed truth solely for held-out evaluation."""
    if annotation_paths and blind_cases:
        raise ValueError("annotation paths and blind cases are mutually exclusive")
    if not annotation_paths and not blind_cases:
        raise ValueError("at least one annotation path or blind case is required")
    cells = parse_cells(bbox_path)
    repository = GlyphRepository.load(glyph_data_path)
    catalog = json.loads(candidate_catalog_path.read_text(encoding="utf-8"))
    cache = PrimitiveAuditCache(cache_dir) if cache_dir is not None else None
    pdf_identity = file_identity(pdf_path)
    module_dir = Path(__file__).parent
    skeleton_algorithm_identity = file_identity(module_dir / "skeleton.py")
    leaf_algorithm_identity = file_identity(module_dir / "leaf_alignment.py")
    pages: dict[int, str] = {}
    cases = []
    requests = [
        {"metadata": load_annotation_metadata(path), "annotationPath": path}
        for path in (annotation_paths or [])
    ]
    requests.extend(
        {"metadata": metadata, "annotationPath": None}
        for metadata in (blind_cases or [])
    )
    for request in requests:
        metadata = request["metadata"]
        annotation_path = request["annotationPath"]
        if not all(
            metadata.get(key) is not None
            for key in ("unicode", "source", "candidateGlyphId")
        ):
            continue
        raw_unicode = metadata["unicode"]
        codepoint = (
            raw_unicode
            if isinstance(raw_unicode, int)
            else int(str(raw_unicode).removeprefix("U+"), 16)
        )
        source = str(metadata["source"])
        candidate_id = int(metadata["candidateGlyphId"])
        cell = find_cell(cells, codepoint, source)
        if cell.page not in pages:
            if cache is None:
                pages[cell.page] = export_page_svg(pdf_path, cell.page)
            else:
                page_key = stable_key(pdf_identity, cell.page)
                pages[cell.page] = cache.text(
                    "page-svg",
                    page_key,
                    lambda page=cell.page: export_page_svg(pdf_path, page),
                )
        geometry = extract_cell_geometry(pages[cell.page], cell)
        if cache is None:
            ink, skeleton = generate_skeleton(geometry, size)
        else:
            skeleton_key = stable_key(
                pdf_identity,
                cell.page,
                cell.unicode,
                cell.source,
                cell.bbox,
                size,
                skeleton_algorithm_identity,
            )
            ink, skeleton = cache.arrays(
                "skeleton",
                skeleton_key,
                lambda geometry=geometry: generate_skeleton(geometry, size),
            )
            ink = ink.astype(bool, copy=False)
            skeleton = skeleton.astype(bool, copy=False)
        skeleton_identity = array_identity(skeleton)
        if cache is None:
            distance_field = distance_transform_edt(~skeleton, return_indices=True)
        else:
            distance_key = stable_key(skeleton_identity)
            distance_field = cache.arrays(
                "distance-field",
                distance_key,
                lambda skeleton=skeleton: distance_transform_edt(
                    ~skeleton, return_indices=True
                ),
            )
        topology_certificate = certify_skeleton(ink, skeleton)
        multiscale_certificate = certify_multiscale(geometry)
        candidate = _compile_audit_candidate(
            repository, catalog, codepoint, candidate_id
        )
        candidate_identity = stable_key(repr(candidate))

        def search_pool(
            path,
            *,
            count,
            allow_extreme_scales=False,
            skeleton=skeleton,
            candidate=candidate,
            distance_field=distance_field,
            skeleton_identity=skeleton_identity,
            candidate_identity=candidate_identity,
        ):
            def produce():
                return search_leaf_alignments(
                    skeleton,
                    candidate,
                    path,
                    count=count,
                    allow_extreme_scales=allow_extreme_scales,
                    distance_field=distance_field,
                )

            if cache is None:
                return produce()
            key = stable_key(
                skeleton_identity,
                candidate_identity,
                leaf_algorithm_identity,
                path,
                count,
                allow_extreme_scales,
            )
            return cache.object("leaf-pool", key, produce)

        # Blind boundary: every fit is frozen before the annotation paths load.
        blind_fits = []
        fit_failures = []
        leaf_paths = sorted({stroke.component_path for stroke in candidate.strokes})
        leaf_alignment_pools = {
            path: search_pool(path, count=128) for path in leaf_paths
        }
        leaf_solution = select_joint_leaf_alignments(
            leaf_alignment_pools, candidate, skeleton=skeleton
        )
        leaf_alignment_pools, leaf_solution, pool_refinements = (
            _adaptive_leaf_alignment_solution(
                skeleton,
                candidate,
                leaf_alignment_pools,
                leaf_solution,
                search=search_pool,
            )
        )
        shortest_path_cache = {}
        if leaf_solution:
            selected_leaf_alignments, leaf_refinements = (
                refine_suspicious_leaf_alignments(
                    skeleton,
                    candidate,
                    leaf_alignment_pools,
                    leaf_solution,
                    shortest_path_cache=shortest_path_cache,
                )
            )
        else:
            selected_leaf_alignments, leaf_refinements = {}, ()
        alternatives_by_stroke = tuple(
            enumerate_primitive_routes(
                skeleton,
                candidate,
                stroke,
                leaf_alignment=selected_leaf_alignments.get(stroke.component_path),
                shortest_path_cache=shortest_path_cache,
            )
            for stroke in candidate.strokes
        )
        joint = select_joint_routes(
            alternatives_by_stroke, int(skeleton.sum()), candidate=candidate
        )
        selected_by_stroke = (
            {route.stroke_index: route for route in joint.routes} if joint else {}
        )
        for stroke in candidate.strokes:
            route = selected_by_stroke.get(stroke.index)
            grammar = canonical_stroke_grammar(stroke.feature, stroke.commands)
            if route is None or not grammar.commands:
                fit_failures.append(
                    {
                        "strokeIndex": stroke.index,
                        "feature": stroke.feature,
                        "commands": list(grammar.commands),
                        "reason": (
                            "no-joint-route" if route is None else "empty-grammar"
                        ),
                    }
                )
                continue
            fitted = route.fit
            blind_fits.append(
                {
                    "strokeIndex": stroke.index,
                    "feature": stroke.feature,
                    "leafId": stroke.leaf_id,
                    "componentPath": list(stroke.component_path),
                    "commands": list(grammar.commands),
                    "tool": grammar.tool,
                    "route": _points(fitted.route),
                    "fittedPath": _points(fitted.fitted_path),
                    "primitives": [
                        {
                            "command": primitive.command,
                            "kind": primitive.kind,
                            "controls": _points(primitive.controls),
                            "routeStart": primitive.route_start,
                            "routeEnd": primitive.route_end,
                        }
                        for primitive in fitted.primitives
                    ],
                    "fitRmse": fitted.rmse,
                    "maximumError": fitted.maximum_error,
                    "controlPointCount": fitted.control_point_count,
                    "routePointCount": len(fitted.route),
                    "compressionRatio": fitted.compression_ratio,
                    "selectionScore": route.score,
                    "alternativeCount": route.alternative_count,
                    "endpointCost": route.endpoint_cost,
                    "guideCost": route.guide_cost,
                    "lengthCost": route.length_cost,
                    "directionCost": route.direction_cost,
                    "regionCost": route.region_cost,
                    "placementConfidence": route.placement_confidence,
                }
            )

        skeleton_xy = np.asarray(
            [
                ((x + 0.5) * 100 / size, (y + 0.5) * 100 / size)
                for y, x in np.argwhere(skeleton)
            ]
        )
        fit_polylines = [
            np.asarray(item["fittedPath"], dtype=float) for item in blind_fits
        ]
        distances = (
            _minimum_distances(skeleton_xy, fit_polylines)
            if fit_polylines
            else np.full(len(skeleton_xy), np.inf)
        )
        residual = skeleton_xy[distances > 0.8]
        frozen = {
            "key": f"U+{codepoint:04X}-{source}-{candidate_id}",
            "unicode": f"U+{codepoint:04X}",
            "character": chr(codepoint),
            "source": source,
            "candidateGlyphId": candidate_id,
            "annotationBasename": annotation_path.name if annotation_path else None,
            "truthUsage": (
                "fit frozen before directed annotation paths load"
                if annotation_path
                else "no directed annotation truth supplied"
            ),
            "skeletonPath": _pixel_path(skeleton),
            "skeletonPixelCount": int(skeleton.sum()),
            "residualPoints": _points(residual),
            "fits": blind_fits,
            "fitFailures": fit_failures,
            "unmatchedStrokes": [
                index
                for index, alternatives in enumerate(alternatives_by_stroke)
                if not alternatives
            ],
            "jointSelection": (
                {
                    "score": joint.score,
                    "coveredPixelCount": joint.covered_pixel_count,
                    "repeatedPixelCount": joint.repeated_pixel_count,
                    "residualPixelCount": joint.residual_pixel_count,
                    "idsStructureCost": joint.ids_structure_cost,
                    "strokeRelationCost": joint.stroke_relation_cost,
                }
                if joint
                else None
            ),
            "leafAlignmentSelection": (
                {
                    "score": leaf_solution.score,
                    "repeatedClaimCount": leaf_solution.repeated_claim_count,
                    "idsStructureCost": leaf_solution.ids_structure_cost,
                    "alignments": [
                        {
                            "componentPath": list(item.component_path),
                            "bounds": list(item.bounds),
                            "score": item.score,
                            "scaleX": item.scale_x,
                            "scaleY": item.scale_y,
                            "offsetX": item.offset_x,
                            "offsetY": item.offset_y,
                            "claimCount": len(item.claimed_pixels),
                            "alternativeCount": len(
                                leaf_alignment_pools[item.component_path]
                            ),
                        }
                        for item in leaf_solution.alignments
                    ],
                }
                if leaf_solution
                else None
            ),
            "leafRouteRefinements": [
                {
                    "componentPath": list(item.component_path),
                    "originalBounds": list(item.original.bounds),
                    "selectedBounds": list(item.selected.bounds),
                    "originalRouteScore": item.original_route_score,
                    "selectedRouteScore": item.selected_route_score,
                    "evaluatedCount": item.evaluated_count,
                    "changed": item.original != item.selected,
                }
                for item in leaf_refinements
            ],
            "leafPoolRefinements": list(pool_refinements),
            "certificates": {
                "topology": topology_certificate.certified,
                "multiscale": multiscale_certificate.certified,
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        # Presentation-only recursive IDS data is derived after the blind
        # fingerprint is frozen; adding or changing the audit UI cannot make
        # an unchanged inference result appear different.
        frozen["idsSegmentation"] = _ids_segmentation(
            candidate, selected_leaf_alignments, blind_fits
        )

        frozen["blindFingerprint"] = fingerprint
        frozen["expectedStrokeCount"] = len(candidate.strokes)
        frozen["truthAvailable"] = annotation_path is not None
        fit_scores = [fit["fitRmse"] for fit in frozen["fits"]]
        frozen["meanFitRmse"] = float(np.mean(fit_scores)) if fit_scores else None
        frozen["residualFraction"] = len(residual) / max(len(skeleton_xy), 1)
        if annotation_path is None:
            for fit in frozen["fits"]:
                fit["truth"] = None
            frozen["truthConsistency"] = None
            frozen["truthStrokeCount"] = len(candidate.strokes)
            frozen["meanTruthChamfer"] = None
            frozen["meanOracleRmse"] = None
            cases.append(frozen)
            continue

        truth_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        truth = _truth_strokes(annotation_path)
        truth_consistency = _truth_consistency(
            truth_payload,
            [stroke.feature for stroke in candidate.strokes],
            [stroke.leaf_id for stroke in candidate.strokes],
            truth,
            candidate,
        )
        if not truth_consistency["valid"]:
            raise ValueError(
                f"{annotation_path.name}: inconsistent truth source: "
                f"{truth_consistency['issues']}"
            )
        for fit in frozen["fits"]:
            index = fit["strokeIndex"]
            if index >= len(truth):
                fit["truth"] = None
                continue
            truth_points = truth[index]["points"]
            truth_stroke = candidate.strokes[index]
            grammar = canonical_stroke_grammar(
                truth_stroke.feature, truth_stroke.commands
            )
            try:
                truth_fit = fit_stroke_primitives(
                    truth_points,
                    grammar.commands,
                    truth_stroke.points,
                    expected_vectors=grammar.direction_vectors,
                    segment_ratios=grammar.segment_ratios,
                )
                oracle = {
                    "rmse": truth_fit.rmse,
                    "maximumError": truth_fit.maximum_error,
                    "controlPointCount": truth_fit.control_point_count,
                    "fittedPath": _points(truth_fit.fitted_path),
                }
            except ValueError as error:
                oracle = {"error": str(error)}
            fit["truth"] = {
                **truth[index],
                "oracle": oracle,
                "chamfer": symmetric_chamfer(fit["fittedPath"], truth_points),
                "startError": float(
                    np.linalg.norm(
                        np.asarray(fit["fittedPath"][0]) - np.asarray(truth_points[0])
                    )
                ),
                "endError": float(
                    np.linalg.norm(
                        np.asarray(fit["fittedPath"][-1]) - np.asarray(truth_points[-1])
                    )
                ),
            }
        frozen["truthConsistency"] = truth_consistency
        frozen["truthStrokeCount"] = len(truth)
        truth_scores = [
            fit["truth"]["chamfer"] for fit in frozen["fits"] if fit["truth"]
        ]
        frozen["meanTruthChamfer"] = (
            float(np.mean(truth_scores)) if truth_scores else None
        )
        oracle_scores = [
            fit["truth"]["oracle"]["rmse"]
            for fit in frozen["fits"]
            if fit["truth"] and "rmse" in fit["truth"]["oracle"]
        ]
        frozen["meanOracleRmse"] = (
            float(np.mean(oracle_scores)) if oracle_scores else None
        )
        cases.append(frozen)
    return cases


_TEMPLATE = r"""<!doctype html><meta charset="utf-8"><title>固定笔画语法的骨架原语拟合</title>
<style>*{box-sizing:border-box}body{margin:0;font:14px system-ui;color:#172033;background:#eef2f7}header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #cbd5e1;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}h1{font-size:18px;margin:0}select,button{padding:6px 9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}.card{background:#fff;border:1px solid #cbd5e1;border-radius:9px;padding:10px;overflow:auto}.wide{grid-column:1/-1}.canvases{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pane h2{font-size:14px;margin:0 0 5px}.pane svg{width:100%;aspect-ratio:1;border:1px solid #dbe2ea;background:#fff}.skeleton{fill:#dbe2ea}.residual{fill:#ef4444}.route{fill:none;stroke:#64748b;stroke-width:.45;stroke-dasharray:1 1}.fit{fill:none;stroke:var(--colour);stroke-width:1.15;stroke-linecap:round;stroke-linejoin:round}.truth{fill:none;stroke:#06b6d4;stroke-width:.75;stroke-dasharray:1.3 .65}.control{fill:#fff;stroke:#111827;stroke-width:.35}.handle{stroke:#94a3b8;stroke-width:.25}table{width:100%;border-collapse:collapse}th,td{padding:5px;border-bottom:1px solid #e2e8f0;text-align:left}tr{cursor:pointer}tr.active{background:#dbeafe}pre{white-space:pre-wrap;background:#f8fafc;padding:8px}.bad{color:#b91c1c}.note{color:#475569;font-size:12px}@media(max-width:1000px){.grid,.canvases{grid-template-columns:1fr}.wide{grid-column:auto}}</style>
<header><h1>固定笔画语法的 PDF 骨架拟合</h1><button id="prev">←</button><select id="case"></select><button id="next">→</button><label><input id="truth" type="checkbox"> <span id="truth-label">显示人工真值（只参与事后评分）</span></label></header>
<main class="grid"><section class="card wide"><div id="summary"></div></section><section class="card wide canvases"><div class="pane"><h2>全字：灰＝骨架，彩色＝最小原语，红＝未解释残差</h2><svg id="all" viewBox="0 0 100 100"></svg></div><div class="pane"><h2>所选笔画：灰虚线＝盲路径，彩色＝拟合，青色＝人工真值</h2><svg id="one" viewBox="0 0 100 100"></svg></div></section><section class="card"><table><thead><tr><th>#</th><th>笔画</th><th>语法</th><th>点</th><th>盲拟合 RMSE</th><th>真值 Chamfer</th><th>真值自身最小拟合</th></tr></thead><tbody id="rows"></tbody></table></section><section class="card"><h2>所选笔画</h2><pre id="details"></pre><p class="note">候选 command 序列、段数与控制点数均为硬约束。人工路径是在 blindFingerprint 固化后才加载，不能改变拟合；“真值自身最小拟合”只回答该固定语法是否足以压缩人工路径。</p></section></main>
<script>const CASES=__DATA__,NS='http://www.w3.org/2000/svg',COLOURS=['#2563eb','#16a34a','#9333ea','#ea580c','#0891b2','#db2777','#65a30d','#dc2626','#0d9488','#7c3aed'];let ci=0,si=0;const byId=id=>document.getElementById(id),metric=x=>x==null?'—':x.toFixed(3);for(const c of CASES){const o=document.createElement('option');o.textContent=`${c.unicode} ${c.character} · ${c.source} · ${c.candidateGlyphId}`;byId('case').append(o)}function el(name,attrs={}){const x=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))x.setAttribute(k,v);return x}function pts(p){return p.map(x=>x.map(v=>v.toFixed(3)).join(',')).join(' ')}function base(svg,c){svg.replaceChildren(el('path',{d:c.skeletonPath,class:'skeleton'}))}function drawFit(svg,f,index){const colour=COLOURS[index%COLOURS.length];svg.append(el('polyline',{points:pts(f.fittedPath),class:'fit',style:`--colour:${colour}`}))}function render(){const c=CASES[ci],hasTruth=c.truthAvailable!==false;byId('case').selectedIndex=ci;si=Math.min(si,Math.max(0,c.fits.length-1));byId('truth').disabled=!hasTruth;if(!hasTruth)byId('truth').checked=false;byId('truth-label').textContent=hasTruth?'显示人工真值（只参与事后评分）':'人工有向真值未提供（严格盲测）';const truthMetric=hasTruth?`事后真值 Chamfer ${metric(c.meanTruthChamfer)}`:'人工有向真值未提供';byId('summary').innerHTML=`<b>${c.unicode} ${c.character} · ${c.source}</b>　candidate ${c.candidateGlyphId}　成功拟合 ${c.fits.length}/${c.expectedStrokeCount??c.truthStrokeCount}　失败 ${c.fitFailures.length}　平均拟合 RMSE ${metric(c.meanFitRmse)}　${truthMetric}　残差 ${(100*c.residualFraction).toFixed(1)}%　证书 ${c.certificates.topology?'拓扑✓':'拓扑✗'}/${c.certificates.multiscale?'多尺度✓':'多尺度✗'}<br><span class="note">blind fingerprint: ${c.blindFingerprint}</span>${c.fitFailures.length?`<pre class="bad">${JSON.stringify(c.fitFailures,null,2)}</pre>`:''}`;const all=byId('all');base(all,c);for(const p of c.residualPoints)all.append(el('circle',{cx:p[0],cy:p[1],r:.23,class:'residual'}));c.fits.forEach((f,i)=>drawFit(all,f,i));const rows=byId('rows');rows.replaceChildren();c.fits.forEach((f,i)=>{const tr=document.createElement('tr');tr.className=i===si?'active':'';tr.innerHTML=`<td>${f.strokeIndex+1}</td><td>${f.feature}</td><td>${f.commands.join('→')}</td><td>${f.controlPointCount}/${f.routePointCount}</td><td>${f.fitRmse.toFixed(3)}</td><td>${f.truth?f.truth.chamfer.toFixed(3):'—'}</td>`;tr.onclick=()=>{si=i;render()};rows.append(tr)});const f=c.fits[si],one=byId('one');base(one,c);if(!f){byId('details').textContent='没有可显示的成功拟合';return}one.append(el('polyline',{points:pts(f.route),class:'route'}));drawFit(one,f,si);for(const primitive of f.primitives){const controls=primitive.controls;if(primitive.kind==='cubic'){one.append(el('polyline',{points:pts(controls),class:'handle'}))}for(const p of controls)one.append(el('circle',{cx:p[0],cy:p[1],r:.65,class:'control'}))}if(byId('truth').checked&&f.truth)one.append(el('polyline',{points:pts(f.truth.points),class:'truth'}));byId('details').textContent=JSON.stringify({strokeIndex:f.strokeIndex,feature:f.feature,leafId:f.leafId,componentPath:f.componentPath,commands:f.commands,primitives:f.primitives.map(p=>({command:p.command,kind:p.kind,controls:p.controls})),fitRmse:f.fitRmse,maximumError:f.maximumError,compressionRatio:f.compressionRatio,truth:hasTruth?(byId('truth').checked?f.truth:'隐藏'):'未提供'},null,2)}byId('case').onchange=e=>{ci=e.target.selectedIndex;si=0;render()};byId('prev').onclick=()=>{ci=(ci-1+CASES.length)%CASES.length;si=0;render()};byId('next').onclick=()=>{ci=(ci+1)%CASES.length;si=0;render()};byId('truth').onchange=render;render();</script>"""

_ORACLE_ENHANCEMENT = r"""<script>
const renderWithoutOracle=render;
render=function(){
  renderWithoutOracle();
  const c=CASES[ci];
  const summary=byId('summary');
  summary.firstChild.insertAdjacentHTML(
    'afterend',
    c.truthAvailable===false?'　人工有向真值未提供':`　真值自身最小拟合 ${metric(c.meanOracleRmse)}`
  );
  const refinements=(c.leafRouteRefinements||[]).filter(item=>item.changed);
  if(refinements.length){
    summary.insertAdjacentHTML(
      'beforeend',
      `<br><span class="note">叶框逐笔复核：${refinements.map(item=>
        `path ${JSON.stringify(item.componentPath)}，${metric(item.originalRouteScore)} → ${metric(item.selectedRouteScore)}，复核 ${item.evaluatedCount} 个框`
      ).join('；')}</span>`
    );
  }
  [...byId('rows').rows].forEach((row,index)=>{
    const fit=c.fits[index],cell=document.createElement('td');
    cell.textContent=fit.truth?.oracle?.rmse==null?'—':fit.truth.oracle.rmse.toFixed(3);
    row.append(cell);
  });
};
render();
</script>"""

_OVERVIEW_ENHANCEMENT = r"""<script>
const summaryCard=byId('summary').closest('section');
const overview=document.createElement('section');
overview.className='card wide';
overview.innerHTML=`<h2 style="font-size:14px;margin:0 0 6px">跨案例盲测摘要（点击切换）</h2>
  <table><thead><tr><th>字源</th><th>candidate</th><th>成功笔画</th><th>真值 Chamfer（若有）</th><th>骨架残差</th><th>盲拟合 RMSE</th></tr></thead><tbody></tbody></table>`;
summaryCard.after(overview);
const overviewBody=overview.querySelector('tbody');
CASES.forEach((c,index)=>{
  const row=document.createElement('tr');
  row.innerHTML=`<td>${c.unicode} ${c.character} · ${c.source}</td><td>${c.candidateGlyphId}</td><td>${c.fits.length}/${c.expectedStrokeCount??c.truthStrokeCount}</td><td>${c.truthAvailable===false?'未提供':metric(c.meanTruthChamfer)}</td><td>${(100*c.residualFraction).toFixed(1)}%</td><td>${metric(c.meanFitRmse)}</td>`;
  row.onclick=()=>{ci=index;si=0;render()};
  overviewBody.append(row);
});
const renderWithoutOverview=render;
render=function(){
  renderWithoutOverview();
  [...overviewBody.rows].forEach((row,index)=>row.classList.toggle('active',index===ci));
};
render();
</script>"""

_IDS_SEGMENTATION_ENHANCEMENT = r"""<script>
(()=>{
const style=document.createElement('style');
style.textContent=`
.ids-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ids-view svg{width:100%;aspect-ratio:1;border:1px solid #dbe2ea;background:#fff}
.component-route{fill:none;stroke:var(--colour);stroke-width:1.35;stroke-linecap:round;stroke-linejoin:round}
.ids-bound{fill:none;stroke-width:.36;stroke-dasharray:1.2 .7;vector-effect:non-scaling-stroke}
.ids-bound.compound{stroke:#334155}.ids-label{font-size:2.25px;font-weight:700;fill:#0f172a;paint-order:stroke;stroke:#fff;stroke-width:.7px;stroke-linejoin:round}
.ids-legend{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}.ids-chip{border:1px solid #cbd5e1;border-radius:999px;padding:3px 7px;background:#fff;font-size:12px}
.ids-chip i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;background:var(--colour)}
.ids-tree{margin-top:12px;padding:9px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;overflow:auto}
.ids-node{border:1px solid #94a3b8;border-radius:7px;padding:6px;background:#fff;min-width:145px}
.ids-node.leaf{border-width:2px;border-color:var(--colour)}.ids-node-title{font-size:12px;font-weight:700;white-space:nowrap;margin-bottom:3px}
.ids-node svg{width:112px;height:112px;display:block;margin:auto;border:1px solid #e2e8f0;background:white}.ids-node .context{fill:#edf2f7}.ids-node .subtree-route{fill:none;stroke:var(--colour);stroke-width:1.25;stroke-linecap:round;stroke-linejoin:round}
.ids-children{display:flex;gap:8px;margin-top:7px;align-items:flex-start}.ids-children.vertical{flex-direction:column}.ids-children.horizontal{flex-direction:row}.ids-strokes{font-size:11px;color:#475569;margin-top:3px;text-align:center}
@media(max-width:1000px){.ids-grid{grid-template-columns:1fr}.ids-children.horizontal{flex-wrap:wrap}}
`;
document.head.append(style);
const originalCanvases=byId('all').closest('section');
const section=document.createElement('section');
section.className='card wide';
section.innerHTML=`<h2 style="font-size:15px;margin:0 0 8px">算法眼中的 IDS 递归分割（仅使用 blind inference）</h2>
<div class="ids-grid">
  <div class="ids-view"><h3>按叶部件着色：同一 componentPath 的笔画同色</h3><svg id="component-view" viewBox="0 0 100 100"></svg><div id="component-legend" class="ids-legend"></div></div>
  <div class="ids-view"><h3>实际 IDS 分区：彩色叶笔画＋递归子树边界</h3><svg id="ids-overlay" viewBox="0 0 100 100"></svg><p class="note">实线彩色路径是最终所选笔画；虚线框由最终 leaf alignment 自底向上合并。框不是 PDF 的机械裁刀，而是算法当前认为各 IDS 子树占据的区域。</p></div>
</div>
<h3>完整 IDS 树：每个节点放入该子树实际拥有的笔画</h3><div id="ids-tree" class="ids-tree"></div>`;
originalCanvases.after(section);
const pathKey=path=>path.length?path.join('.'):'root';
const operatorLabel=operator=>({'⿰':'左右','⿱':'上下','⿲':'左中右','⿳':'上中下','⿸':'左上包围','⿹':'右上包围'}[operator]||'复合');
const leafFor=(c,path)=>c.idsSegmentation.leaves.find(item=>pathKey(item.componentPath)===pathKey(path));
const colourFor=(c,path)=>COLOURS[(leafFor(c,path)?.colourIndex??0)%COLOURS.length];
function addRoute(svg,fit,colour,className='component-route'){
  svg.append(el('polyline',{points:pts(fit.fittedPath),class:className,style:`--colour:${colour}`}));
}
function descendants(node){return new Set(node.strokeIndices)}
function drawNodeBounds(svg,c,node,depth=0){
  const [x0,y0,x1,y1]=node.actualBounds;
  const leaf=!node.children.length;
  const colour=leaf?colourFor(c,node.componentPath):'#334155';
  svg.append(el('rect',{x:x0,y:y0,width:Math.max(.1,x1-x0),height:Math.max(.1,y1-y0),rx:.7,class:`ids-bound ${leaf?'leaf':'compound'}`,style:`stroke:${colour};opacity:${Math.max(.28,1-depth*.12)}`}));
  const labelY=leaf?Math.max(2.8,y0+2.7):Math.min(98.5,y1-1.1-depth*1.7);
  const label=el('text',{x:x0+.8+depth*.5,y:labelY,class:'ids-label'});
  label.textContent=`${leaf?'叶':operatorLabel(node.operator)} · ${node.glyphId} · ${pathKey(node.componentPath)}`;
  svg.append(label);
  node.children.forEach(child=>drawNodeBounds(svg,c,child,depth+1));
}
function miniSvg(c,node){
  const svg=el('svg',{viewBox:'0 0 100 100'});
  svg.append(el('path',{d:c.skeletonPath,class:'context'}));
  const owned=descendants(node);
  c.fits.filter(f=>owned.has(f.strokeIndex)).forEach(f=>addRoute(svg,f,colourFor(c,f.componentPath),'subtree-route'));
  return svg;
}
function treeNode(c,node){
  const leaf=!node.children.length;
  const box=document.createElement('div');
  box.className=`ids-node ${leaf?'leaf':'compound'}`;
  if(leaf)box.style.setProperty('--colour',colourFor(c,node.componentPath));
  const title=document.createElement('div');title.className='ids-node-title';
  title.textContent=`${leaf?'叶部件':operatorLabel(node.operator)} · id ${node.glyphId} · path ${pathKey(node.componentPath)}`;
  box.append(title,miniSvg(c,node));
  if(leaf){
    const strokes=document.createElement('div');strokes.className='ids-strokes';
    strokes.textContent=node.strokeIndices.map(index=>{const f=c.fits.find(item=>item.strokeIndex===index);return f?`#${index+1} ${f.feature}`:`#${index+1} 未匹配`}).join(' · ');
    box.append(strokes);
  }
  if(node.children.length){
    const children=document.createElement('div');
    children.className=`ids-children ${['⿱','⿳'].includes(node.operator)?'vertical':'horizontal'}`;
    node.children.forEach(child=>children.append(treeNode(c,child)));
    box.append(children);
  }
  return box;
}
function renderIds(){
  const c=CASES[ci],seg=c.idsSegmentation;
  const component=byId('component-view');base(component,c);
  c.fits.forEach(f=>addRoute(component,f,colourFor(c,f.componentPath)));
  const overlay=byId('ids-overlay');base(overlay,c);
  c.fits.forEach(f=>addRoute(overlay,f,colourFor(c,f.componentPath)));
  drawNodeBounds(overlay,c,seg.root);
  const legend=byId('component-legend');legend.replaceChildren();
  seg.leaves.forEach(leaf=>{const chip=document.createElement('span');chip.className='ids-chip';chip.style.setProperty('--colour',COLOURS[leaf.colourIndex%COLOURS.length]);chip.innerHTML=`<i></i>id ${leaf.glyphId} · path ${pathKey(leaf.componentPath)} · 笔画 ${leaf.strokeIndices.map(i=>i+1).join(',')}`;legend.append(chip)});
  const tree=byId('ids-tree');tree.replaceChildren(treeNode(c,seg.root));
}
const previousRender=render;
render=function(){previousRender();renderIds()};
render();
})();
</script>"""

_QUERY_CASE_ENHANCEMENT = r"""<script>
const requestedCase=Number(new URLSearchParams(location.search).get('case'));
if(Number.isInteger(requestedCase)&&requestedCase>=0&&requestedCase<CASES.length){
  ci=requestedCase;si=0;render();
}
</script>"""


def render_primitive_fit_audit(cases: list[dict]) -> str:
    if not cases:
        raise ValueError("no primitive-fit cases")
    return (
        _TEMPLATE.replace(
            "__DATA__", json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
        )
        + _ORACLE_ENHANCEMENT
        + _OVERVIEW_ENHANCEMENT
        + _IDS_SEGMENTATION_ENHANCEMENT
        + _QUERY_CASE_ENHANCEMENT
    )
