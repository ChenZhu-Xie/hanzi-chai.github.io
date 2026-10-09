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


def _pixel_path(mask: np.ndarray) -> str:
    scale = 100 / mask.shape[0]
    return "".join(
        f"M{x * scale:.3f},{y * scale:.3f}h{scale:.3f}v{scale:.3f}h-{scale:.3f}z"
        for y, x in np.argwhere(mask)
    )


def _points(points) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in points]


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
    annotation_paths: list[Path],
    *,
    size: int = 256,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Fit first, then load directed truth solely for held-out evaluation."""
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
    for annotation_path in annotation_paths:
        metadata = load_annotation_metadata(annotation_path)
        if not all(
            metadata.get(key) is not None
            for key in ("unicode", "source", "candidateGlyphId")
        ):
            continue
        codepoint = int(str(metadata["unicode"]).removeprefix("U+"), 16)
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
            "annotationBasename": annotation_path.name,
            "truthUsage": "fit frozen before directed annotation paths load",
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
        frozen["blindFingerprint"] = fingerprint
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
        fit_scores = [fit["fitRmse"] for fit in frozen["fits"]]
        frozen["meanFitRmse"] = float(np.mean(fit_scores)) if fit_scores else None
        frozen["residualFraction"] = len(residual) / max(len(skeleton_xy), 1)
        cases.append(frozen)
    return cases


_TEMPLATE = r"""<!doctype html><meta charset="utf-8"><title>固定笔画语法的骨架原语拟合</title>
<style>*{box-sizing:border-box}body{margin:0;font:14px system-ui;color:#172033;background:#eef2f7}header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #cbd5e1;padding:10px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}h1{font-size:18px;margin:0}select,button{padding:6px 9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}.card{background:#fff;border:1px solid #cbd5e1;border-radius:9px;padding:10px;overflow:auto}.wide{grid-column:1/-1}.canvases{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pane h2{font-size:14px;margin:0 0 5px}.pane svg{width:100%;aspect-ratio:1;border:1px solid #dbe2ea;background:#fff}.skeleton{fill:#dbe2ea}.residual{fill:#ef4444}.route{fill:none;stroke:#64748b;stroke-width:.45;stroke-dasharray:1 1}.fit{fill:none;stroke:var(--colour);stroke-width:1.15;stroke-linecap:round;stroke-linejoin:round}.truth{fill:none;stroke:#06b6d4;stroke-width:.75;stroke-dasharray:1.3 .65}.control{fill:#fff;stroke:#111827;stroke-width:.35}.handle{stroke:#94a3b8;stroke-width:.25}table{width:100%;border-collapse:collapse}th,td{padding:5px;border-bottom:1px solid #e2e8f0;text-align:left}tr{cursor:pointer}tr.active{background:#dbeafe}pre{white-space:pre-wrap;background:#f8fafc;padding:8px}.bad{color:#b91c1c}.note{color:#475569;font-size:12px}@media(max-width:1000px){.grid,.canvases{grid-template-columns:1fr}.wide{grid-column:auto}}</style>
<header><h1>固定笔画语法的 PDF 骨架拟合</h1><button id="prev">←</button><select id="case"></select><button id="next">→</button><label><input id="truth" type="checkbox"> 显示人工真值（只参与事后评分）</label></header>
<main class="grid"><section class="card wide"><div id="summary"></div></section><section class="card wide canvases"><div class="pane"><h2>全字：灰＝骨架，彩色＝最小原语，红＝未解释残差</h2><svg id="all" viewBox="0 0 100 100"></svg></div><div class="pane"><h2>所选笔画：灰虚线＝盲路径，彩色＝拟合，青色＝人工真值</h2><svg id="one" viewBox="0 0 100 100"></svg></div></section><section class="card"><table><thead><tr><th>#</th><th>笔画</th><th>语法</th><th>点</th><th>盲拟合 RMSE</th><th>真值 Chamfer</th><th>真值自身最小拟合</th></tr></thead><tbody id="rows"></tbody></table></section><section class="card"><h2>所选笔画</h2><pre id="details"></pre><p class="note">候选 command 序列、段数与控制点数均为硬约束。人工路径是在 blindFingerprint 固化后才加载，不能改变拟合；“真值自身最小拟合”只回答该固定语法是否足以压缩人工路径。</p></section></main>
<script>const CASES=__DATA__,NS='http://www.w3.org/2000/svg',COLOURS=['#2563eb','#16a34a','#9333ea','#ea580c','#0891b2','#db2777','#65a30d','#dc2626','#0d9488','#7c3aed'];let ci=0,si=0;const byId=id=>document.getElementById(id),metric=x=>x==null?'—':x.toFixed(3);for(const c of CASES){const o=document.createElement('option');o.textContent=`${c.unicode} ${c.character} · ${c.source} · ${c.candidateGlyphId}`;byId('case').append(o)}function el(name,attrs={}){const x=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))x.setAttribute(k,v);return x}function pts(p){return p.map(x=>x.map(v=>v.toFixed(3)).join(',')).join(' ')}function base(svg,c){svg.replaceChildren(el('path',{d:c.skeletonPath,class:'skeleton'}))}function drawFit(svg,f,index){const colour=COLOURS[index%COLOURS.length];svg.append(el('polyline',{points:pts(f.fittedPath),class:'fit',style:`--colour:${colour}`}))}function render(){const c=CASES[ci];byId('case').selectedIndex=ci;si=Math.min(si,Math.max(0,c.fits.length-1));byId('summary').innerHTML=`<b>${c.unicode} ${c.character} · ${c.source}</b>　candidate ${c.candidateGlyphId}　成功拟合 ${c.fits.length}/${c.truthStrokeCount}　失败 ${c.fitFailures.length}　平均拟合 RMSE ${metric(c.meanFitRmse)}　事后真值 Chamfer ${metric(c.meanTruthChamfer)}　残差 ${(100*c.residualFraction).toFixed(1)}%　证书 ${c.certificates.topology?'拓扑✓':'拓扑✗'}/${c.certificates.multiscale?'多尺度✓':'多尺度✗'}<br><span class="note">blind fingerprint: ${c.blindFingerprint}</span>${c.fitFailures.length?`<pre class="bad">${JSON.stringify(c.fitFailures,null,2)}</pre>`:''}`;const all=byId('all');base(all,c);for(const p of c.residualPoints)all.append(el('circle',{cx:p[0],cy:p[1],r:.23,class:'residual'}));c.fits.forEach((f,i)=>drawFit(all,f,i));const rows=byId('rows');rows.replaceChildren();c.fits.forEach((f,i)=>{const tr=document.createElement('tr');tr.className=i===si?'active':'';tr.innerHTML=`<td>${f.strokeIndex+1}</td><td>${f.feature}</td><td>${f.commands.join('→')}</td><td>${f.controlPointCount}/${f.routePointCount}</td><td>${f.fitRmse.toFixed(3)}</td><td>${f.truth?f.truth.chamfer.toFixed(3):'—'}</td>`;tr.onclick=()=>{si=i;render()};rows.append(tr)});const f=c.fits[si],one=byId('one');base(one,c);if(!f){byId('details').textContent='没有可显示的成功拟合';return}one.append(el('polyline',{points:pts(f.route),class:'route'}));drawFit(one,f,si);for(const primitive of f.primitives){const controls=primitive.controls;if(primitive.kind==='cubic'){one.append(el('polyline',{points:pts(controls),class:'handle'}))}for(const p of controls)one.append(el('circle',{cx:p[0],cy:p[1],r:.65,class:'control'}))}if(byId('truth').checked&&f.truth)one.append(el('polyline',{points:pts(f.truth.points),class:'truth'}));byId('details').textContent=JSON.stringify({strokeIndex:f.strokeIndex,feature:f.feature,leafId:f.leafId,componentPath:f.componentPath,commands:f.commands,primitives:f.primitives.map(p=>({command:p.command,kind:p.kind,controls:p.controls})),fitRmse:f.fitRmse,maximumError:f.maximumError,compressionRatio:f.compressionRatio,truth:byId('truth').checked?f.truth:'隐藏'},null,2)}byId('case').onchange=e=>{ci=e.target.selectedIndex;si=0;render()};byId('prev').onclick=()=>{ci=(ci-1+CASES.length)%CASES.length;si=0;render()};byId('next').onclick=()=>{ci=(ci+1)%CASES.length;si=0;render()};byId('truth').onchange=render;render();</script>"""

_ORACLE_ENHANCEMENT = r"""<script>
const renderWithoutOracle=render;
render=function(){
  renderWithoutOracle();
  const c=CASES[ci];
  const summary=byId('summary');
  summary.firstChild.insertAdjacentHTML(
    'afterend',
    `　真值自身最小拟合 ${metric(c.meanOracleRmse)}`
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
  <table><thead><tr><th>字源</th><th>candidate</th><th>成功笔画</th><th>真值 Chamfer</th><th>骨架残差</th><th>盲拟合 RMSE</th></tr></thead><tbody></tbody></table>`;
summaryCard.after(overview);
const overviewBody=overview.querySelector('tbody');
CASES.forEach((c,index)=>{
  const row=document.createElement('tr');
  row.innerHTML=`<td>${c.unicode} ${c.character} · ${c.source}</td><td>${c.candidateGlyphId}</td><td>${c.fits.length}/${c.truthStrokeCount}</td><td>${metric(c.meanTruthChamfer)}</td><td>${(100*c.residualFraction).toFixed(1)}%</td><td>${metric(c.meanFitRmse)}</td>`;
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


def render_primitive_fit_audit(cases: list[dict]) -> str:
    if not cases:
        raise ValueError("no primitive-fit cases")
    return (
        _TEMPLATE.replace(
            "__DATA__", json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
        )
        + _ORACLE_ENHANCEMENT
        + _OVERVIEW_ENHANCEMENT
    )
