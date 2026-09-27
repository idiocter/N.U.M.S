"""Validate labeled NUMS tool examples and create MLX-LM data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from nums.agent import SYSTEM_PROMPT
from nums.tools import TOOL_SCHEMAS


SCHEMAS = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}


def load_examples(path: Path, minimum: int = 15) -> list[dict]:
    examples = []
    seen = set()
    seen_prompts: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            identifier, user, name, arguments = (
                item["id"], item["user"], item["tool"], item["arguments"]
            )
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("id must be a nonempty string")
            if identifier in seen:
                raise ValueError(f"duplicate id: {identifier}")
            if not isinstance(user, str) or not user.strip():
                raise ValueError("user must be a nonempty string")
            normalized_prompt = " ".join(user.split()).casefold()
            if normalized_prompt in seen_prompts:
                raise ValueError(
                    f"duplicate user prompt in {seen_prompts[normalized_prompt]} and {identifier}"
                )
            if (name != "none" and name not in SCHEMAS) or not isinstance(arguments, dict):
                raise ValueError("invalid tool or arguments")
            if name == "none" and not (isinstance(item.get("answer"), str) and item["answer"].strip()):
                raise ValueError("a no-tool example needs a nonempty answer")
            required = set(SCHEMAS[name]["parameters"]["required"]) if name != "none" else set()
            allowed = set(SCHEMAS[name]["parameters"]["properties"]) if name != "none" else set()
            if not required <= arguments.keys() or not arguments.keys() <= allowed:
                raise ValueError("missing or unexpected argument")
            if not all(isinstance(value, str) for value in arguments.values()):
                raise ValueError("arguments must be strings")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        seen.add(identifier)
        seen_prompts[normalized_prompt] = identifier
        examples.append(item)
    if len(examples) < minimum:
        raise ValueError(f"at least {minimum} labeled examples are needed")
    return examples


def convert(item: dict) -> dict:
    name = item["tool"]
    assistant = (
        {"role": "assistant", "content": item["answer"]}
        if name == "none" else
        {"role": "assistant", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(item["arguments"])},
        }]}
    )
    return {
        "id": item["id"],
        "expected_tool": name,
        "expected_arguments": item["arguments"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["user"]},
            assistant,
        ],
        "tools": TOOL_SCHEMAS,
    }


def build(source: Path, destination: Path) -> dict[str, int]:
    examples = sorted(
        load_examples(source, minimum=15),
        key=lambda item: hashlib.sha256(item["id"].encode()).hexdigest(),
    )
    groups = {name: [] for name in ("train", "valid", "test")}
    manifest_path = destination / "split_manifest.json"
    if manifest_path.exists():
        assignments = json.loads(manifest_path.read_text())
    else:
        assignments = {}
        for name in groups:
            prior = destination / f"{name}.jsonl"
            if prior.exists():
                for line in prior.read_text().splitlines():
                    if line.strip():
                        assignments[json.loads(line)["id"]] = name
    if not isinstance(assignments, dict) or any(
        value not in groups for value in assignments.values()
    ):
        raise ValueError(f"Invalid split manifest at {manifest_path}")
    initial_split = not assignments
    train_end = len(examples) * 6 // 10
    valid_end = len(examples) * 8 // 10
    for index, item in enumerate(examples):
        identifier = item["id"]
        if identifier not in assignments:
            if initial_split:
                name = "train" if index < train_end else "valid" if index < valid_end else "test"
            else:
                bucket = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % 100
                name = "train" if bucket < 60 else "valid" if bucket < 80 else "test"
            assignments[identifier] = name
        groups[assignments[identifier]].append(convert(item))
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, group in groups.items():
        output = destination / f"{name}.jsonl"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination,
                prefix=f".{name}.", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                for item in group:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination,
            prefix=".split_manifest.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(assignments, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {name: len(group) for name, group in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(build(args.source, args.destination))


if __name__ == "__main__":
    main()
