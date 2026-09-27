"""Compare held-out NUMS tool evaluations before switching models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation report must contain cases")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("evaluation case has no valid id")
        if row["id"] in by_id:
            raise ValueError(f"duplicate evaluation id: {row['id']}")
        if not isinstance(row.get("tool_correct"), bool) or not isinstance(
            row.get("arguments_correct"), bool
        ):
            raise ValueError(f"evaluation case {row['id']} lacks correctness scores")
        by_id[row["id"]] = row
    return by_id


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    before = _cases(baseline)
    after = _cases(candidate)
    if before.keys() != after.keys():
        raise ValueError("reports must contain the same held-out case ids")
    for identifier in before:
        expected = (before[identifier].get("expected_tool"),
                    before[identifier].get("expected_arguments"))
        actual = (after[identifier].get("expected_tool"),
                  after[identifier].get("expected_arguments"))
        if expected != actual:
            raise ValueError(f"held-out label changed for {identifier}")

    before_choice = sum(row["tool_correct"] for row in before.values())
    after_choice = sum(row["tool_correct"] for row in after.values())
    before_full = sum(row["arguments_correct"] for row in before.values())
    after_full = sum(row["arguments_correct"] for row in after.values())
    reasons = []
    if after_full <= before_full:
        reasons.append("full tool and argument accuracy did not improve")
    if after_choice < before_choice:
        reasons.append("tool choice accuracy regressed")

    per_tool_before: defaultdict[str, int] = defaultdict(int)
    per_tool_after: defaultdict[str, int] = defaultdict(int)
    for identifier, row in before.items():
        tool = row["expected_tool"]
        per_tool_before[tool] += row["arguments_correct"]
        per_tool_after[tool] += after[identifier]["arguments_correct"]
    for tool in sorted(per_tool_before):
        if per_tool_after[tool] < per_tool_before[tool]:
            reasons.append(f"{tool} accuracy regressed")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        eligible, reasons = compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if eligible:
        print("Candidate improved held-out accuracy without per-tool regressions.")
        return
    print("Candidate is not ready: " + "; ".join(reasons))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
