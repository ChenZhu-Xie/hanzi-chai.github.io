"""Named PDF regressions for the residual-ink decoder.

The runner deliberately distinguishes a missing local case from a passing
case.  It invokes the ordinary review CLI, then reduces its machine-readable
audit to a small set of stable topology assertions.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_CASES = (
    {"unicode": "U+66DA", "source": "J", "componentId": 228, "kind": "grass"},
    {"unicode": "U+6726", "source": "J", "componentId": 228, "kind": "grass"},
    {"unicode": "U+6418", "source": "G", "componentId": 220, "kind": "hand"},
    {"unicode": "U+6418", "source": "G", "componentId": 439, "kind": "old-head"},
    {"unicode": "U+64CE", "source": "T", "componentId": 486, "kind": "grass-order"},
)


def requirement_key(requirement: dict) -> str:
    return f"{requirement['unicode']}-{requirement['source']}:{requirement['componentId']}"


def _case_key(observation: dict) -> tuple[str, str]:
    return str(observation["unicode"]).upper(), str(observation["source"]).upper()


def _component_strokes(observation: dict, component_id: int) -> list[dict]:
    return [
        stroke
        for stroke in observation.get("strokes", ())
        if int(stroke.get("componentId", -1)) == component_id
    ]


def _horizontal_vertical_leak(strokes: list[dict]) -> bool:
    vertical = {"N", "S"}
    return any(
        edge.get("outsideContact", False)
        and edge.get("startSector") in vertical
        and edge.get("endSector") in vertical
        for stroke in strokes
        if stroke.get("feature") == "横"
        for edge in stroke.get("selectedHalfEdges", ())
    )


def _named_failures(observation: dict, requirement: dict) -> list[str]:
    strokes = _component_strokes(observation, int(requirement["componentId"]))
    failures = []
    if not strokes:
        return ["required-component-not-present"]
    if any(not stroke.get("selectedHalfEdges") for stroke in strokes):
        failures.append("component-route-incomplete")
    if requirement["kind"] in {"grass", "grass-order"} and _horizontal_vertical_leak(strokes):
        failures.append("horizontal-owns-vertical-half-edge")
    if requirement["kind"] == "grass-order":
        features = [stroke.get("feature") for stroke in strokes]
        if features[:4] != ["横", "竖", "竖", "横"]:
            failures.append("verified-source-order-mismatch")
    if requirement["kind"] == "old-head":
        if not strokes or strokes[0].get("feature") != "横":
            failures.append("old-head-does-not-start-with-horizontal")
    if requirement["kind"] == "hand":
        hook_roads = {
            edge.get("roadId")
            for stroke in strokes
            if stroke.get("feature") == "竖钩"
            for edge in stroke.get("selectedHalfEdges", ())
            if edge.get("outsideContact", False)
        }
        rising_roads = {
            edge.get("roadId")
            for stroke in strokes
            if stroke.get("feature") == "提"
            for edge in stroke.get("selectedHalfEdges", ())
            if edge.get("outsideContact", False)
        }
        if hook_roads & rising_roads:
            failures.append("vertical-hook-owns-later-rising-road")
    return list(dict.fromkeys(failures))


def build_named_report(observations: list[dict]) -> dict:
    by_case = {_case_key(item): item for item in observations}
    cases = []
    for requirement in REQUIRED_CASES:
        observation = by_case.get((requirement["unicode"], requirement["source"]))
        if observation is None:
            cases.append(
                {
                    "key": requirement_key(requirement),
                    "kind": requirement["kind"],
                    "status": "missing",
                    "failures": ["local-case-missing"],
                }
            )
            continue
        failures = _named_failures(observation, requirement)
        cases.append(
            {
                "key": requirement_key(requirement),
                "kind": requirement["kind"],
                "status": "failed" if failures else "passed",
                "failures": failures,
                "decoderStatus": observation.get("status"),
                "html": observation.get("html"),
                "audit": observation.get("audit"),
            }
        )
    counts = {
        status: sum(item["status"] == status for item in cases)
        for status in ("passed", "failed", "missing")
    }
    review_required = sum(
        item.get("decoderStatus") != "safe-candidate"
        for item in cases
        if item["status"] != "missing"
    )
    return {
        "schemaVersion": 1,
        "model": "named-residual-pdf-regressions",
        "status": (
            "failed"
            if counts["failed"] or counts["missing"]
            else "review-required"
            if review_required
            else "passed"
        ),
        "summary": {**counts, "decoderReviewRequired": review_required},
        "cases": cases,
    }


def _annotation_metadata(path: Path) -> dict | None:
    document = json.loads(path.read_text("utf-8-sig"))
    metadata = document.get("metadata") if isinstance(document, dict) else None
    if not metadata or not metadata.get("unicode") or not metadata.get("source"):
        return None
    return metadata


def _run_case(args, annotation: Path, metadata: dict) -> dict:
    unicode_label = str(metadata["unicode"]).upper()
    source = str(metadata["source"]).upper()
    glyph_id = int(metadata["candidateGlyphId"])
    stem = f"{unicode_label.lower().replace('+', '')}-{source.lower()}-{glyph_id}"
    html_path = args.html_dir / f"{stem}.html"
    audit_path = args.html_dir / f"{stem}.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("unihan-ink-diffusion.py")),
        "--decoder", "residual",
        "--bbox-cache", str(args.bbox_cache),
        "--pdf", str(args.pdf),
        "--candidates", str(args.candidates),
        "--unicode", unicode_label,
        "--source", source,
        "--glyph-id", str(glyph_id),
        "--annotations", str(annotation),
        "--canvas", str(args.canvas),
        "--beam-width", str(args.beam_width),
        "--output", str(html_path),
        "--audit-output", str(audit_path),
    ]
    if args.learned_model:
        command.extend(("--learned-rules", str(args.learned_model)))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        return {
            "unicode": unicode_label,
            "source": source,
            "componentIds": [],
            "strokes": [],
            "status": "needs-review",
            "hardViolations": ["runner-failed"],
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    audit = json.loads(audit_path.read_text("utf-8"))
    decision = audit.get("decision") or {}
    return {
        "unicode": unicode_label,
        "source": source,
        "componentIds": sorted({int(item["componentId"]) for item in audit.get("strokes", ())}),
        "strokes": audit.get("strokes", []),
        "status": decision.get("status"),
        "hardViolations": [
            reason
            for reason in decision.get("reviewReasons", ())
            if reason.startswith(("no-feasible-route", "closed-component", "forbidden"))
        ],
        "html": str(html_path),
        "audit": str(audit_path),
    }


def _index_html(report: dict) -> str:
    rows = []
    for item in report["cases"]:
        link = (
            f'<a href="{html.escape(Path(item["html"]).name)}">打开逐笔审计</a>'
            if item.get("html")
            else "无本地页面"
        )
        rows.append(
            f"<tr><td>{html.escape(item['key'])}</td><td>{item['status']}</td>"
            f"<td>{html.escape(str(item.get('decoderStatus') or '—'))}</td>"
            f"<td>{html.escape(', '.join(item['failures']) or '—')}</td><td>{link}</td></tr>"
        )
    return """<!doctype html><meta charset="utf-8"><title>Residual PDF regressions</title>
<style>body{font:14px/1.5 Segoe UI,sans-serif;margin:24px;color:#172033}table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd5e1;padding:8px;text-align:left}th{background:#f1f5f9}</style>
<h1>Residual PDF named regressions</h1><p>Named assertion 通过不等于整字可自动写入；请同时检查 decoder status。</p><table><tr><th>case</th><th>named assertion</th><th>decoder status</th><th>failures</th><th>review</th></tr>""" + "".join(rows) + "</table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--bbox-cache", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--learned-model", type=Path)
    parser.add_argument("--canvas", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-dir", type=Path, required=True)
    args = parser.parse_args()
    args.html_dir.mkdir(parents=True, exist_ok=True)
    wanted = {(item["unicode"], item["source"]) for item in REQUIRED_CASES}
    observations = []
    seen = set()
    for annotation in args.annotations:
        metadata = _annotation_metadata(annotation)
        if metadata is None:
            continue
        key = (str(metadata["unicode"]).upper(), str(metadata["source"]).upper())
        if key not in wanted or key in seen:
            continue
        seen.add(key)
        observations.append(_run_case(args, annotation, metadata))
    report = build_named_report(observations)
    report["observations"] = observations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    (args.html_dir / "index.html").write_text(_index_html(report), "utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
