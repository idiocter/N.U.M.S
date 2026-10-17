"""Prepare NUMS traces for human review and export approved training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from training.build_dataset import load_examples


def _read_jsonl(path: Path, kind: str) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{kind} {path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{kind} {path}:{number}: record must be a JSON object")
        rows.append(row)
    return rows


def _validate_trace(trace: dict, path: Path, number: int) -> None:
    prompt = trace.get("prompt")
    calls = trace.get("tool_calls")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"trace {path}:{number}: prompt must be a nonempty string")
    if not isinstance(calls, list):
        raise ValueError(f"trace {path}:{number}: tool_calls must be a list")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str) or not call["tool"]:
            raise ValueError(f"trace {path}:{number}: each tool call needs a nonempty tool")
        if not isinstance(call.get("arguments"), dict):
            raise ValueError(f"trace {path}:{number}: tool call arguments must be an object")
    if trace.get("reply") is not None and not isinstance(trace["reply"], str):
        raise ValueError(f"trace {path}:{number}: reply must be a string or null")
    if trace.get("error") is not None and not isinstance(trace["error"], str):
        raise ValueError(f"trace {path}:{number}: error must be a string or null")


def _validate_review_rows(rows: list[dict], path: Path) -> None:
    seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"review {path}:{number}: id must be a nonempty string")
        if identifier in seen:
            raise ValueError(f"review {path}:{number}: duplicate id: {identifier}")
        seen.add(identifier)
        if not isinstance(row.get("user"), str) or not row["user"].strip():
            raise ValueError(f"review {path}:{number}: user must be a nonempty string")
        for key in ("proposed_tool", "tool", "decision"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"review {path}:{number}: {key} must be a string")
        for key in ("proposed_arguments", "arguments"):
            if not isinstance(row.get(key), dict):
                raise ValueError(f"review {path}:{number}: {key} must be an object")
        if not isinstance(row.get("reviewed"), bool):
            raise ValueError(f"review {path}:{number}: reviewed must be a boolean")
        for key in ("answer", "proposed_answer", "source_error", "review_notes", "group"):
            if row.get(key) is not None and not isinstance(row[key], str):
                raise ValueError(f"review {path}:{number}: {key} must be a string or null")
        fingerprint = row.get("source_fingerprint")
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 16
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(
                f"review {path}:{number}: source_fingerprint must be a 16-character hex digest"
            )


def _write_private_jsonl(path: Path, rows: list[dict], validate: bool = False) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if validate and rows:
            load_examples(temporary, minimum=1)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(trace_path: Path, review_path: Path) -> int:
    rows = _read_jsonl(review_path, "review") if review_path.exists() else []
    for row in rows:
        row.setdefault("proposed_tool", "")
        row.setdefault("proposed_arguments", {})
        row.setdefault("proposed_answer", row.get("answer"))
        row.setdefault("decision", "")
        row.setdefault("group", "")
        row.setdefault("review_notes", "")
        row.setdefault("source_error", None)
    _validate_review_rows(rows, review_path)
    existing_ids = {row["id"] for row in rows}
    existing_fingerprints = {
        row["source_fingerprint"] for row in rows if row.get("source_fingerprint")
    }
    traces = _read_jsonl(trace_path, "trace")
    for number, trace in enumerate(traces, 1):
        _validate_trace(trace, trace_path, number)
    legacy_by_signature: dict[str, list[dict]] = {}
    for row in rows:
        if "source_fingerprint" not in row:
            signature = json.dumps(
                [row.get("user"), row.get("proposed_tool"), row.get("proposed_arguments")],
                sort_keys=True,
            )
            legacy_by_signature.setdefault(signature, []).append(row)
    signatures = []
    for trace in traces:
        calls = trace.get("tool_calls") or []
        first = calls[0] if calls else {
            "tool": "" if trace.get("error") else "none", "arguments": {},
        }
        signatures.append(json.dumps(
            [trace.get("prompt"), first.get("tool", "none"), first.get("arguments", {})],
            sort_keys=True,
        ))
    signature_counts = Counter(signatures)
    for number, trace in enumerate(traces, 1):
        calls = trace.get("tool_calls") or []
        first = calls[0] if calls else {
            "tool": "" if trace.get("error") else "none", "arguments": {},
        }
        source = {
            "at": trace.get("at"), "prompt": trace.get("prompt"), "tool_calls": calls,
            "error": trace.get("error"),
        }
        if source["at"] is None:
            source["line"] = number
        fingerprint = hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest()[:16]
        identifier = f"trace-{fingerprint}"
        if fingerprint in existing_fingerprints or identifier in existing_ids:
            continue
        signature = signatures[number - 1]
        legacy = legacy_by_signature.get(signature, [])
        if signature_counts[signature] == 1 and len(legacy) == 1:
            legacy[0]["source_fingerprint"] = fingerprint
            existing_fingerprints.add(fingerprint)
            continue
        rows.append({
            "id": identifier,
            "source_fingerprint": fingerprint,
            "user": trace.get("prompt", ""),
            "proposed_tool": first.get("tool", "none"),
            "proposed_arguments": first.get("arguments", {}),
            "tool": first.get("tool", "none"),
            "arguments": first.get("arguments", {}),
            "answer": trace.get("reply") if not calls else None,
            "proposed_answer": trace.get("reply") if not calls else None,
            "source_error": trace.get("error"),
            "reviewed": False,
            "decision": "",
            "group": "",
            "review_notes": "",
        })
        existing_ids.add(identifier)
        existing_fingerprints.add(fingerprint)
    _write_private_jsonl(review_path, rows)
    return len(rows)


def export(review_path: Path, output_path: Path) -> int:
    examples = []
    rows = _read_jsonl(review_path, "review")
    for row in rows:
        row.setdefault("proposed_tool", "")
        row.setdefault("proposed_arguments", {})
        row.setdefault("proposed_answer", row.get("answer"))
        row.setdefault("decision", "")
        row.setdefault("group", "")
        row.setdefault("review_notes", "")
        row.setdefault("source_error", None)
    _validate_review_rows(rows, review_path)
    for row in rows:
        if row.get("reviewed") is not True:
            continue
        decision = row.get("decision")
        if decision not in {"accept", "corrected", "reject"}:
            raise ValueError(
                f"{row.get('id', 'review row')}: review decision must be accept, corrected, or reject"
            )
        if decision == "reject":
            if not (row.get("review_notes") or "").strip():
                raise ValueError(
                    f"{row.get('id', 'review row')}: reject decision needs review_notes"
                )
            continue
        if decision == "accept" and (
            row.get("tool") != row.get("proposed_tool")
            or row.get("arguments") != row.get("proposed_arguments")
            or (
                row.get("tool") == "none"
                and row.get("answer") != row.get("proposed_answer")
            )
        ):
            raise ValueError(
                f"{row.get('id', 'review row')}: accept decision does not match the proposal"
            )
        if decision == "corrected" and (
            row.get("tool") == row.get("proposed_tool")
            and row.get("arguments") == row.get("proposed_arguments")
            and (
                row.get("tool") != "none"
                or row.get("answer") == row.get("proposed_answer")
            )
        ):
            raise ValueError(
                f"{row.get('id', 'review row')}: corrected decision does not change the proposal"
            )
        example = {key: row[key] for key in ("id", "user", "tool", "arguments")}
        if row.get("group") is not None and row.get("group") != "":
            example["group"] = row["group"]
        if row["tool"] == "none":
            example["answer"] = row.get("answer") or ""
        examples.append(example)
    _write_private_jsonl(output_path, examples, validate=True)
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("trace", type=Path)
    prepare_parser.add_argument("review", type=Path)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("review", type=Path)
    export_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = prepare(args.trace, args.review) if args.command == "prepare" else export(args.review, args.output)
    print(f"{count} records written")


if __name__ == "__main__":
    main()
