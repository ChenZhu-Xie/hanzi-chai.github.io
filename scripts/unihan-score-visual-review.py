"""Score blind human or multimodal choices without treating abstentions as errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()

    answer_key = json.loads(args.answer_key.read_text("utf-8"))
    review = json.loads(args.review.read_text("utf-8"))
    answered = 0
    correct = 0
    invalid = []
    disagreements = []
    for filename, expected in answer_key.items():
        choice = review.get(filename, {}).get("choice", "uncertain")
        if choice == "uncertain":
            continue
        if choice not in {"A", "B", "C", "D", "E", "F"}:
            invalid.append(filename)
            continue
        answered += 1
        if choice == expected["expected"]:
            correct += 1
        else:
            disagreements.append(
                {
                    "file": filename,
                    "choice": choice,
                    "expected": expected["expected"],
                }
            )
    payload = {
        "total": len(answer_key),
        "answered": answered,
        "coverage": answered / len(answer_key) if answer_key else 0,
        "correct": correct,
        "answeredAccuracy": correct / answered if answered else None,
        "invalid": invalid,
        "disagreements": disagreements,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
