import json
from pathlib import Path

import pytest

from training.build_dataset import build, load_examples


SOURCE = Path(__file__).resolve().parents[1] / "training" / "seed_examples.jsonl"


def test_seed_examples_build_disjoint_splits(tmp_path: Path) -> None:
    counts = build(SOURCE, tmp_path)
    assert sum(counts.values()) == len(load_examples(SOURCE))
    assert all(counts.values())
    groups = [
        [json.loads(line) for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()]
        for name in ("train", "valid", "test")
    ]
    ids = [item["id"] for group in groups for item in group]
    assert len(ids) == len(set(ids))
    assert all(item["tools"] for group in groups for item in group)


def test_unknown_tool_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"bad","user":"hello","tool":"bad_tool","arguments":{}}\n')
    with pytest.raises(ValueError, match="invalid tool"):
        load_examples(source)
