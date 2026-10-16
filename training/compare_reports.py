"""Compare held-out NUMS tool evaluations before switching models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if report.get("report_version") not in {None, 1}:
        raise ValueError("unsupported evaluation report version")
    digest = report.get("dataset_digest")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("evaluation report has an invalid dataset digest")
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
        if row["arguments_correct"] and not row["tool_correct"]:
            raise ValueError(
                f"evaluation case {row['id']} has correct arguments with the wrong tool"
            )
        if row.get("error") and (row["tool_correct"] or row["arguments_correct"]):
            raise ValueError(f"evaluation case {row['id']} has an error but is scored correct")
        by_id[row["id"]] = row
    summary_fields = {"total", "tool_correct", "arguments_correct"}
    present = summary_fields.intersection(report)
    if present and present != summary_fields:
        raise ValueError("evaluation report has an incomplete score summary")
    if present:
        expected = {
            "total": len(rows),
            "tool_correct": sum(row["tool_correct"] for row in rows),
            "arguments_correct": sum(row["arguments_correct"] for row in rows),
        }
        for field, value in expected.items():
            if report[field] != value:
                raise ValueError(f"evaluation report {field} does not match its cases")
    return by_id


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    before = _cases(baseline)
    after = _cases(candidate)
    before_digest = baseline.get("dataset_digest")
    after_digest = candidate.get("dataset_digest")
    if (before_digest is None) != (after_digest is None):
        raise ValueError("evaluation dataset digest is missing from one report")
    if before_digest is not None and before_digest != after_digest:
        raise ValueError("reports use different evaluation datasets")
    if before.keys() != after.keys():
        raise ValueError("reports must contain the same held-out case ids")
    for identifier in before:
        before_fingerprint = before[identifier].get("case_fingerprint")
        after_fingerprint = after[identifier].get("case_fingerprint")
        if (before_fingerprint is None) != (after_fingerprint is None):
            raise ValueError(f"held-out fingerprint missing for {identifier}")
        if before_fingerprint is not None:
            if not isinstance(before_fingerprint, str) or not isinstance(after_fingerprint, str):
                raise ValueError(f"held-out fingerprint is invalid for {identifier}")
            if before_fingerprint != after_fingerprint:
                raise ValueError(f"held-out input changed for {identifier}")
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
    before_errors = sum(bool(row.get("error")) for row in before.values())
    after_errors = sum(bool(row.get("error")) for row in after.values())
    if after_errors > before_errors:
        reasons.append("evaluation response errors increased")
    newly_failing = sorted(
        identifier for identifier, row in after.items()
        if row.get("error") and not before[identifier].get("error")
    )
    if newly_failing:
        reasons.append("new evaluation response errors: " + ", ".join(newly_failing))

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
