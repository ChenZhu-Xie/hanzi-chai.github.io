#!/usr/bin/env python3
"""Validate, provision, and run immutable historical ink experts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from unihan_experts.hashing import sha256_file
from unihan_experts.models import expert_content_identity
from unihan_experts.registry import (
    load_case_registry,
    load_expert_registry,
    load_shared_artifacts,
    resolve_annotation,
    validate_expert_registry,
)
from unihan_experts.runner import CellSpec, merge_matrix_records, run_cell
from unihan_experts.report import generate_reports
from unihan_experts.worktrees import (
    ensure_expert_worktree,
    resolve_commit,
    source_tree_hash,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXPERTS_PATH = SCRIPT_DIR / "unihan-experts.json"
CASES_PATH = SCRIPT_DIR / "unihan-expert-cases.json"
RESOURCE_TIMEOUTS = {"small": 900.0, "medium": 1800.0, "large": 3600.0}


def interpreter_identity() -> dict[str, object]:
    dependencies: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "opencv-python", "psutil"):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    return {
        "executable": str(Path(sys.executable).resolve()),
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }


def select(items, selected_id, attribute):
    if selected_id is None:
        return tuple(items)
    selected = tuple(item for item in items if getattr(item, attribute) == selected_id)
    if not selected:
        raise ValueError(f"unknown selection: {selected_id}")
    return selected


def load_inputs(repo: Path):
    experts = load_expert_registry(EXPERTS_PATH)
    validate_expert_registry(
        experts, commit_resolver=lambda revision: resolve_commit(repo, revision)
    )
    cases = load_case_registry(CASES_PATH)
    relative = load_shared_artifacts(CASES_PATH)
    shared = {alias: (repo / path).resolve() for alias, path in relative.items()}
    missing = [str(path) for path in shared.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing shared artifacts: " + ", ".join(missing))
    return experts, cases, shared


def annotation_roots(repo: Path, values: list[Path]) -> tuple[Path, ...]:
    roots = [(repo / ".local").resolve()]
    roots.extend(path.resolve() for path in values)
    return tuple(dict.fromkeys(roots))


def command_validate(args) -> int:
    repo = args.repo.resolve()
    experts, cases, shared = load_inputs(repo)
    experts = select(experts, args.expert, "expert_id")
    cases = select(cases, args.case, "case_id")
    roots = annotation_roots(repo, args.annotation_root)
    result = {
        "experts": [
            {
                "expertId": expert.expert_id,
                "commit": resolve_commit(repo, expert.commit),
                "sourceTreeHash": source_tree_hash(repo, expert.commit),
            }
            for expert in experts
        ],
        "cases": [
            {
                "caseId": case.case_id,
                "annotation": str(resolve_annotation(case, roots)),
                "annotationSha256": sha256_file(resolve_annotation(case, roots)),
            }
            for case in cases
        ],
        "sharedArtifacts": {
            alias: {"path": str(path), "sha256": sha256_file(path)}
            for alias, path in shared.items()
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_provision(args) -> int:
    repo = args.repo.resolve()
    experts, _cases, _shared = load_inputs(repo)
    experts = select(experts, args.expert, "expert_id")
    root = repo / ".local" / "expert-worktrees"
    for expert in experts:
        path = ensure_expert_worktree(repo, root, expert)
        print(f"{expert.expert_id}\t{resolve_commit(path, 'HEAD')}\t{path}")
    return 0


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_id(args, runs_root: Path) -> str:
    if args.run_id:
        return args.run_id
    latest = runs_root / "latest.txt"
    if args.resume and latest.is_file():
        return latest.read_text(encoding="utf-8").strip()
    return _new_run_id()


def command_run(args) -> int:
    repo = args.repo.resolve()
    experts, cases, shared = load_inputs(repo)
    experts = select(experts, args.expert, "expert_id")
    cases = select(cases, args.case, "case_id")
    roots = annotation_roots(repo, args.annotation_root)
    shared_hashes = {alias: sha256_file(path) for alias, path in shared.items()}
    identity = interpreter_identity()
    worktree_root = repo / ".local" / "expert-worktrees"
    runs_root = repo / ".local" / "ink-experts" / "runs"
    run_id = _run_id(args, runs_root)
    run_root = runs_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    (runs_root / "latest.txt").write_text(run_id + "\n", encoding="utf-8")

    specs: list[CellSpec] = []
    for expert in experts:
        worktree = ensure_expert_worktree(repo, worktree_root, expert)
        commit = resolve_commit(repo, expert.commit)
        tree_hash = source_tree_hash(repo, commit)
        content_id = expert_content_identity(
            expert,
            resolved_commit=commit,
            source_tree_hash=tree_hash,
            interpreter_identity=identity,
        )
        resolved_rules = {
            artifact.alias: (repo / artifact.path).resolve()
            for artifact in expert.rule_artifacts
        }
        for artifact in expert.rule_artifacts:
            path = resolved_rules[artifact.alias]
            if not path.is_file() or sha256_file(path) != artifact.sha256:
                raise ValueError(
                    f"rule artifact mismatch for {expert.expert_id}: {artifact.alias}"
                )
        input_hashes = {
            **shared_hashes,
            **{f"rule:{item.alias}": item.sha256 for item in expert.rule_artifacts},
        }
        for case in cases:
            specs.append(
                CellSpec(
                    expert=expert,
                    case=case,
                    expert_content_id=content_id,
                    input_hashes=input_hashes,
                    python_executable=Path(sys.executable),
                    interpreter_identity=identity,
                    worktree=worktree,
                    shared_artifacts=shared,
                    resolved_rule_artifacts=resolved_rules,
                    run_root=run_root,
                    timeout_seconds=RESOURCE_TIMEOUTS[expert.resource_class],
                )
            )

    results = []
    for index, spec in enumerate(
        sorted(specs, key=lambda item: (item.expert.expert_id, item.case.case_id)),
        start=1,
    ):
        print(
            f"[{index}/{len(specs)}] {spec.expert.expert_id} {spec.case.case_id}",
            flush=True,
        )
        result = run_cell(
            spec,
            annotation_resolver=lambda case: resolve_annotation(case, roots),
            resume=args.resume,
        )
        results.append(result)
        print(f"  -> {result['status']}", flush=True)
    matrix_path = run_root / "matrix.json"
    existing = []
    if matrix_path.is_file():
        loaded = json.loads(matrix_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"existing run matrix is not a list: {matrix_path}")
        existing = loaded
    combined = merge_matrix_records(existing, results)
    matrix_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"runId={run_id}")
    print(f"matrix={matrix_path}")
    return 0 if all(item["status"] not in {"process-error", "timeout"} for item in results) else 1


def command_report(args) -> int:
    repo = args.repo.resolve()
    runs_root = repo / ".local" / "ink-experts" / "runs"
    run_id = args.run
    if run_id == "latest":
        latest = runs_root / "latest.txt"
        if not latest.is_file():
            raise FileNotFoundError("no latest expert benchmark run is recorded")
        run_id = latest.read_text(encoding="utf-8").strip()
    matrix_path = runs_root / run_id / "matrix.json"
    cells = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(cells, list):
        raise ValueError(f"run matrix is not a list: {matrix_path}")
    experts = load_expert_registry(EXPERTS_PATH)
    declared = {
        expert.expert_id: expert.declared_capabilities for expert in experts
    }
    output_root = repo / ".local" / "ink-experts" / "reports" / run_id
    outputs = generate_reports(
        cells,
        output_root,
        run_id=run_id,
        declared_capabilities=declared,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("validate", command_validate),
        ("provision", command_provision),
        ("run", command_run),
    ):
        command = subparsers.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--expert")
        if name != "provision":
            command.add_argument("--case")
            command.add_argument(
                "--annotation-root", type=Path, action="append", default=[]
            )
        if name == "run":
            command.add_argument("--run-id")
            command.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report")
    report.set_defaults(function=command_report)
    report.add_argument("--run", default="latest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
