"""Validate labeled NUMS tool examples and create MLX-LM data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from nums.agent import SYSTEM_PROMPT
from nums.tools import TOOL_SCHEMAS, validate_tool_arguments


SCHEMAS = {item["function"]["name"]: item["function"] for item in TOOL_SCHEMAS}


def _example_fingerprint(item: dict) -> str:
    protected = {
        key: item.get(key) for key in ("user", "tool", "arguments", "answer", "group")
    }
    return hashlib.sha256(
        json.dumps(protected, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def load_examples(path: Path, minimum: int = 15) -> list[dict]:
    examples = []
    seen = set()
    seen_prompts: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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
            if "group" in item and (
                not isinstance(item["group"], str) or not item["group"].strip()
            ):
                raise ValueError("group must be a nonempty string when provided")
            normalized_prompt = " ".join(user.split()).casefold()
            if normalized_prompt in seen_prompts:
                raise ValueError(
                    f"duplicate user prompt in {seen_prompts[normalized_prompt]} and {identifier}"
                )
            if (name != "none" and name not in SCHEMAS) or not isinstance(arguments, dict):
                raise ValueError("invalid tool or arguments")
            if name == "none" and not (isinstance(item.get("answer"), str) and item["answer"].strip()):
                raise ValueError("a no-tool example needs a nonempty answer")
            if name == "none" and arguments:
                raise ValueError("a no-tool example must have empty arguments")
            if name != "none":
                argument_error = validate_tool_arguments(name, arguments)
                if argument_error:
                    raise ValueError(argument_error)
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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["user"]},
        assistant,
    ]
    evaluation_input = {
        "messages": messages[:-1],
        "tools": TOOL_SCHEMAS,
        "expected_tool": name,
        "expected_arguments": item["arguments"],
    }
    case_fingerprint = hashlib.sha256(
        json.dumps(evaluation_input, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return {
        "id": item["id"],
        "case_fingerprint": case_fingerprint,
        "expected_tool": name,
        "expected_arguments": item["arguments"],
        "messages": messages,
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
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid split manifest at {manifest_path}: {exc}") from exc
        if isinstance(manifest, dict) and manifest.get("version") == 2:
            assignments = manifest.get("assignments")
        else:
            assignments = manifest
    else:
        assignments = {}
        for name in groups:
            prior = destination / f"{name}.jsonl"
            if prior.exists():
                try:
                    for line in prior.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            assignments[json.loads(line)["id"]] = name
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid prior split at {prior}: {exc}") from exc
    if not isinstance(assignments, dict) or any(not isinstance(key, str) for key in assignments):
        raise ValueError(f"Invalid split manifest at {manifest_path}")
    for value in assignments.values():
        if isinstance(value, str):
            valid = value in groups
        else:
            valid = (
                isinstance(value, dict)
                and value.get("split") in groups
                and isinstance(value.get("fingerprint"), str)
            )
        if not valid:
            raise ValueError(f"Invalid split manifest at {manifest_path}")
    initial_split = not assignments
    train_end = len(examples) * 6 // 10
    valid_end = len(examples) * 8 // 10
    updated_assignments = {}
    group_assignments: dict[str, str] = {}
    for item in examples:
        group = item.get("group")
        prior = assignments.get(item["id"])
        if group is None or prior is None:
            continue
        name = prior if isinstance(prior, str) else prior["split"]
        if group in group_assignments and group_assignments[group] != name:
            raise ValueError(f"Group {group} is assigned across multiple splits")
        group_assignments[group] = name
    for index, item in enumerate(examples):
        identifier = item["id"]
        group = item.get("group")
        fingerprint = _example_fingerprint(item)
        prior = assignments.get(identifier)
        if prior is None:
            if group in group_assignments:
                name = group_assignments[group]
            elif initial_split:
                name = "train" if index < train_end else "valid" if index < valid_end else "test"
            else:
                assignment_key = f"group:{group}" if group is not None else identifier
                bucket = int(hashlib.sha256(assignment_key.encode()).hexdigest()[:8], 16) % 100
                name = "train" if bucket < 60 else "valid" if bucket < 80 else "test"
        elif isinstance(prior, str):
            name = prior
        else:
            name = prior["split"]
            if prior["fingerprint"] != fingerprint:
                raise ValueError(
                    f"Example {identifier} changed after assignment to the {name} split; use a new id"
                )
        if group is not None:
            if group in group_assignments and group_assignments[group] != name:
                raise ValueError(f"Group {group} is assigned across multiple splits")
            group_assignments[group] = name
        updated_assignments[identifier] = {"split": name, "fingerprint": fingerprint}
        groups[name].append(convert(item))
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
            json.dump(
                {"version": 2, "assignments": updated_assignments},
                handle, indent=2, sort_keys=True,
            )
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
