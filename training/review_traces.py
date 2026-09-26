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
    rows = [json.loads(line) for line in review_path.read_text().splitlines() if line.strip()] \
        if review_path.exists() else []
    existing_ids = {row["id"] for row in rows}
    existing_fingerprints = {
        row["source_fingerprint"] for row in rows if row.get("source_fingerprint")
    }
    traces = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
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
        first = calls[0] if calls else {"tool": "none", "arguments": {}}
        signatures.append(json.dumps(
            [trace.get("prompt"), first.get("tool", "none"), first.get("arguments", {})],
            sort_keys=True,
        ))
    signature_counts = Counter(signatures)
    for number, trace in enumerate(traces, 1):
        calls = trace.get("tool_calls") or []
        first = calls[0] if calls else {"tool": "none", "arguments": {}}
        source = {
            "at": trace.get("at"), "prompt": trace.get("prompt"), "tool_calls": calls,
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
            "reviewed": False,
            "review_notes": "",
        })
        existing_ids.add(identifier)
        existing_fingerprints.add(fingerprint)
    _write_private_jsonl(review_path, rows)
    return len(rows)


def export(review_path: Path, output_path: Path) -> int:
    examples = []
    for line in review_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("reviewed") is not True:
            continue
        example = {key: row[key] for key in ("id", "user", "tool", "arguments")}
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
