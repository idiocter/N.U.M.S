import json
import stat
from pathlib import Path

import pytest

from training.build_dataset import build, load_examples
from training.coverage import summarize


SOURCE = Path(__file__).resolve().parents[1] / "training" / "seed_examples.jsonl"


def test_seed_examples_build_disjoint_splits(tmp_path: Path) -> None:
    destination = tmp_path / "splits"
    counts = build(SOURCE, destination)
    assert sum(counts.values()) == len(load_examples(SOURCE))
    assert all(counts.values())
    groups = [
        [json.loads(line) for line in (destination / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "valid", "test")
    ]
    ids = [item["id"] for group in groups for item in group]
    assert len(ids) == len(set(ids))
    assert all(item["tools"] for group in groups for item in group)
    assert all(
        len(item["case_fingerprint"]) == 64
        for group in groups for item in group
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE((destination / f"{name}.jsonl").stat().st_mode) == 0o600
        for name in ("train", "valid", "test")
    )


def test_unknown_tool_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"bad","user":"hello","tool":"bad_tool","arguments":{}}\n')
    with pytest.raises(ValueError, match="invalid tool"):
        load_examples(source)


def test_training_labels_must_satisfy_tool_value_constraints(tmp_path: Path) -> None:
    source = tmp_path / "bad-value.jsonl"
    source.write_text(json.dumps({
        "id": "bad-value", "user": "wait for a save window", "tool": "wait_for_app_window",
        "arguments": {"app": "TextEdit", "window": "Save", "state": "visible"},
    }) + "\n")

    with pytest.raises(ValueError, match="state must be one of"):
        load_examples(source, minimum=1)


def test_duplicate_prompts_cannot_leak_across_data_splits(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.jsonl"
    rows = [
        {"id": "one", "user": "Open Safari", "tool": "open_item", "arguments": {"target": "Safari"}},
        {"id": "two", "user": "  open   SAFARI  ", "tool": "open_item", "arguments": {"target": "Safari"}},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="duplicate user prompt.*one and two"):
        load_examples(source, minimum=1)


def test_adding_labels_keeps_existing_split_assignments(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(SOURCE.read_text())
    destination = tmp_path / "splits"
    original_count = len(load_examples(source))
    train_end = original_count * 6 // 10
    valid_end = original_count * 8 // 10
    assert build(source, destination) == {
        "train": train_end,
        "valid": valid_end - train_end,
        "test": original_count - valid_end,
    }
    original = json.loads((destination / "split_manifest.json").read_text())["assignments"]
    source.write_text(SOURCE.read_text() + json.dumps({
        "id": "new-real-case", "user": "Show local machine architecture",
        "tool": "system_info", "arguments": {},
    }) + "\n")

    counts = build(source, destination)
    updated = json.loads((destination / "split_manifest.json").read_text())["assignments"]

    assert sum(counts.values()) == original_count + 1
    assert {identifier: updated[identifier] for identifier in original} == original
    assert updated["new-real-case"]["split"] in {"train", "valid", "test"}


def test_assigned_example_cannot_change_prompt_or_label(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(SOURCE.read_text())
    destination = tmp_path / "splits"
    build(source, destination)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["arguments"] = {"path": "/tmp/changed"}
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="changed after assignment.*use a new id"):
        build(source, destination)


def test_legacy_split_manifest_is_upgraded(tmp_path: Path) -> None:
    destination = tmp_path / "splits"
    destination.mkdir()
    destination.joinpath("split_manifest.json").write_text(json.dumps({
        "read-01": "test",
    }))

    build(SOURCE, destination)

    manifest = json.loads((destination / "split_manifest.json").read_text())
    assert manifest["version"] == 2
    assert manifest["assignments"]["read-01"]["split"] == "test"
    assert len(manifest["assignments"]["read-01"]["fingerprint"]) == 64


def test_seed_examples_cover_every_runtime_tool_at_least_twice() -> None:
    report = summarize(SOURCE, minimum_per_tool=2)
    assert report["examples"] == 138
    assert report["runtime_tools"] == 57
    assert report["no_tool_examples"] == 10
    assert report["missing"] == []
    assert report["underrepresented"] == []
