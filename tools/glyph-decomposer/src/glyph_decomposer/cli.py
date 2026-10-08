"""Command line boundary for the deterministic decomposer."""

from __future__ import annotations

import argparse
from pathlib import Path

from .candidate_graph import compile_candidate_graph, render_candidate_graph
from .cut_hypotheses import build_cut_audit, render_cut_audit
from .domain import DecompositionRequest
from .grammar import GlyphRepository
from .pipeline import decompose_with_artifacts
from .review import render_review_html
from .skeleton import audit_annotations, render_skeleton_audit
from .trajectory import (
    complete_historic_routes,
    load_historic_review,
    render_trajectory_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyph-decomposer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="run read-only root decomposition")
    audit.add_argument("request", type=Path)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--review-html", type=Path)
    trajectory = subparsers.add_parser(
        "trajectory-audit", help="extend historic directed routes on the PDF skeleton"
    )
    trajectory.add_argument("historic_review", type=Path)
    trajectory.add_argument("--output", type=Path, required=True)
    skeleton = subparsers.add_parser(
        "skeleton-audit", help="generate deterministic PDF skeletons before truth"
    )
    skeleton.add_argument("--pdf", type=Path, required=True)
    skeleton.add_argument("--bbox-cache", type=Path, required=True)
    skeleton.add_argument("--annotations", type=Path, nargs="+", required=True)
    skeleton.add_argument("--size", type=int, default=256)
    skeleton.add_argument("--output", type=Path, required=True)
    candidate = subparsers.add_parser(
        "candidate-audit", help="compile candidate strokes and recursive IDS"
    )
    candidate.add_argument("request", type=Path)
    candidate.add_argument("--output", type=Path, required=True)
    cut = subparsers.add_parser(
        "cut-audit", help="rank skeleton-only root IDS cut/no-cut hypotheses"
    )
    cut.add_argument("request", type=Path)
    cut.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cut-audit":
        request = DecompositionRequest.model_validate_json(
            args.request.read_text(encoding="utf-8")
        )
        audit = build_cut_audit(request)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_cut_audit(audit), encoding="utf-8")
        return 0
    if args.command == "candidate-audit":
        request = DecompositionRequest.model_validate_json(
            args.request.read_text(encoding="utf-8")
        )
        repository = GlyphRepository.load(Path(request.glyph_data_path))
        graph = compile_candidate_graph(repository, request.candidate_glyph_id)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_candidate_graph(graph), encoding="utf-8")
        return 0
    if args.command == "skeleton-audit":
        cases = audit_annotations(
            args.pdf, args.bbox_cache, args.annotations, size=args.size
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_skeleton_audit(cases), encoding="utf-8")
        return 0
    if args.command == "trajectory-audit":
        payload = load_historic_review(args.historic_review)
        completed, evidence = complete_historic_routes(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            render_trajectory_review(payload, completed, evidence), encoding="utf-8"
        )
        return 0
    if args.command != "audit":
        raise AssertionError(args.command)
    request = DecompositionRequest.model_validate_json(
        args.request.read_text(encoding="utf-8")
    )
    artifacts = decompose_with_artifacts(request)
    payload = artifacts.result.model_dump_json(by_alias=True, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.review_html:
        args.review_html.parent.mkdir(parents=True, exist_ok=True)
        args.review_html.write_text(
            render_review_html(request, artifacts), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
