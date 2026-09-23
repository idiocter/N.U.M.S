"""Prepare NUMS traces for human review and export approved training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
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
    for number, line in enumerate(trace_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        trace = json.loads(line)
        calls = trace.get("tool_calls") or []
        first = calls[0] if calls else {"tool": "none", "arguments": {}}
        identifier = hashlib.sha256(
            f"{number}:{trace.get('at')}:{trace.get('prompt')}".encode()
        ).hexdigest()[:16]
        if f"trace-{identifier}" in existing_ids:
            continue
        rows.append({
            "id": f"trace-{identifier}",
            "user": trace.get("prompt", ""),
            "proposed_tool": first.get("tool", "none"),
            "proposed_arguments": first.get("arguments", {}),
            "tool": first.get("tool", "none"),
            "arguments": first.get("arguments", {}),
            "answer": trace.get("reply") if not calls else None,
            "reviewed": False,
            "review_notes": "",
        })
        existing_ids.add(f"trace-{identifier}")
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
