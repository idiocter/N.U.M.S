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


def test_optional_paraphrase_group_must_be_nonempty(tmp_path: Path) -> None:
    source = tmp_path / "bad-group.jsonl"
    source.write_text(json.dumps({
        "id": "bad-group", "group": "", "user": "show system info",
        "tool": "system_info", "arguments": {},
    }) + "\n")

    with pytest.raises(ValueError, match="group must be a nonempty string"):
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


def test_paraphrase_group_stays_in_one_split(tmp_path: Path) -> None:
    source = tmp_path / "grouped.jsonl"
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    rows[0]["group"] = "read-a-file"
    rows[-1]["group"] = "read-a-file"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    destination = tmp_path / "splits"

    build(source, destination)

    assignments = json.loads(
        (destination / "split_manifest.json").read_text()
    )["assignments"]
    assert assignments[rows[0]["id"]]["split"] == assignments[rows[-1]["id"]]["split"]


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


def test_dataset_files_are_read_as_utf8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    encodings = []
    original = Path.read_text

    def spy(self: Path, *args: object, **kwargs: object) -> str:
        encodings.append(kwargs.get("encoding"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)

    load_examples(SOURCE)
    build(SOURCE, tmp_path / "splits")
    build(SOURCE, tmp_path / "splits")

    assert encodings and set(encodings) == {"utf-8"}


def test_no_tool_example_with_arguments_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        '{"id":"bad","user":"hello","tool":"none","arguments":{"path":"/tmp"},"answer":"Hi."}\n'
    )

    with pytest.raises(ValueError, match="must have empty arguments"):
        load_examples(source, minimum=1)


def test_corrupt_split_manifest_is_a_clean_error(tmp_path: Path) -> None:
    destination = tmp_path / "splits"
    build(SOURCE, destination)
    (destination / "split_manifest.json").write_text("{not json")

    with pytest.raises(ValueError, match="Invalid split manifest"):
        build(SOURCE, destination)


def test_corrupt_legacy_split_is_a_clean_error(tmp_path: Path) -> None:
    destination = tmp_path / "splits"
    destination.mkdir()
    (destination / "train.jsonl").write_text('{"no_id": true}\n')

    with pytest.raises(ValueError, match="Invalid prior split"):
        build(SOURCE, destination)


def test_dataset_cli_exits_with_message_on_bad_input(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from training import build_dataset, coverage

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")

    monkeypatch.setattr("sys.argv", ["build_dataset", str(bad), str(tmp_path / "out")])
    with pytest.raises(SystemExit) as build_exit:
        build_dataset.main()
    monkeypatch.setattr("sys.argv", ["coverage", str(bad)])
    with pytest.raises(SystemExit) as coverage_exit:
        coverage.main()

    assert ":1:" in str(build_exit.value)
    assert ":1:" in str(coverage_exit.value)


def test_build_refuses_to_overwrite_its_own_source(tmp_path: Path) -> None:
    destination = tmp_path / "splits"
    destination.mkdir()
    source = destination / "train.jsonl"
    source.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    original = source.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="would overwrite the labels"):
        build(source, destination)

    assert source.read_text(encoding="utf-8") == original


def test_coverage_command_reports_complete_and_incomplete_sets(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from training.coverage import main

    monkeypatch.setattr("sys.argv", ["coverage", str(SOURCE)])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["missing"] == [] and report["underrepresented"] == []

    monkeypatch.setattr(
        "sys.argv", ["coverage", str(SOURCE), "--minimum-per-tool", "999", "--require-complete"]
    )
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1


def test_coverage_command_rejects_a_nonpositive_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    from training.coverage import main

    monkeypatch.setattr("sys.argv", ["coverage", str(SOURCE), "--minimum-per-tool", "0"])

    with pytest.raises(SystemExit, match="must be positive"):
        main()


def _row(number: int, **changes: object) -> dict:
    row = {"id": f"ex-{number:02d}", "user": f"unique request number {number}",
           "tool": "system_info", "arguments": {}}
    row.update(changes)
    return row


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_row(1, id=""), "id must be a nonempty string"),
        (_row(1, id=7), "id must be a nonempty string"),
        (_row(1, user="   "), "user must be a nonempty string"),
        (_row(1, group=""), "group must be a nonempty string"),
        (_row(1, group=4), "group must be a nonempty string"),
        (_row(1, tool="missing_tool"), "invalid tool or arguments"),
        (_row(1, arguments=[]), "invalid tool or arguments"),
        (_row(1, tool="none", answer=" "), "needs a nonempty answer"),
        (_row(1, tool="none", arguments={}), "needs a nonempty answer"),
        (_row(1, tool="read_file", arguments={}), "missing required arguments"),
        (_row(1, arguments={"unexpected": "x"}), "unexpected arguments"),
    ],
)
def test_invalid_labels_report_their_line(tmp_path: Path, row: dict, message: str) -> None:
    source = _write_rows(tmp_path / "labels.jsonl", [_row(0), row])

    with pytest.raises(ValueError, match=rf"labels\.jsonl:2: .*{message}"):
        load_examples(source, minimum=1)


def test_duplicate_ids_and_prompts_are_rejected(tmp_path: Path) -> None:
    duplicate_id = _write_rows(tmp_path / "a.jsonl", [_row(0), _row(1, id="ex-00")])
    with pytest.raises(ValueError, match="duplicate id: ex-00"):
        load_examples(duplicate_id, minimum=1)

    duplicate_prompt = _write_rows(
        tmp_path / "b.jsonl", [_row(0, user="Open  Safari"), _row(1, user="open safari")]
    )
    with pytest.raises(ValueError, match="duplicate user prompt in ex-00 and ex-01"):
        load_examples(duplicate_prompt, minimum=1)


def test_minimum_example_count_is_enforced(tmp_path: Path) -> None:
    source = _write_rows(tmp_path / "few.jsonl", [_row(number) for number in range(3)])

    with pytest.raises(ValueError, match="at least 15 labeled examples"):
        load_examples(source)
    assert len(load_examples(source, minimum=3)) == 3


def _split_names(destination: Path) -> dict[str, str]:
    names = {}
    for name in ("train", "valid", "test"):
        for line in (destination / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            names[json.loads(line)["id"]] = name
    return names


def test_grouped_paraphrases_stay_in_one_split(tmp_path: Path) -> None:
    rows = [_row(number) for number in range(20)]
    for number in (3, 8, 13, 18):
        rows[number]["group"] = "same-intent"
    source = _write_rows(tmp_path / "labels.jsonl", rows)

    build(source, tmp_path / "splits")

    names = _split_names(tmp_path / "splits")
    assert len({names[f"ex-{number:02d}"] for number in (3, 8, 13, 18)}) == 1


def test_rebuilding_keeps_assignments_and_new_labels_do_not_move_old_ones(tmp_path: Path) -> None:
    rows = [_row(number) for number in range(20)]
    source = _write_rows(tmp_path / "labels.jsonl", rows)
    destination = tmp_path / "splits"
    build(source, destination)
    first = _split_names(destination)

    build(source, destination)
    assert _split_names(destination) == first

    _write_rows(source, rows + [_row(number) for number in range(20, 40)])
    build(source, destination)
    second = _split_names(destination)
    assert {key: second[key] for key in first} == first
    assert len(second) == 40


def test_editing_an_assigned_label_requires_a_new_id(tmp_path: Path) -> None:
    rows = [_row(number) for number in range(20)]
    source = _write_rows(tmp_path / "labels.jsonl", rows)
    destination = tmp_path / "splits"
    build(source, destination)
    before = (destination / "train.jsonl").read_text(encoding="utf-8")

    rows[5]["user"] = "a completely different request"
    _write_rows(source, rows)

    with pytest.raises(ValueError, match="ex-05 changed after assignment .* use a new id"):
        build(source, destination)
    assert (destination / "train.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.parametrize(
    "assignments",
    [
        {"ex-00": "staging"},
        {"ex-00": {"split": "train"}},
        {"ex-00": {"split": "nowhere", "fingerprint": "x"}},
        {"ex-00": 5},
    ],
)
def test_invalid_manifest_entries_are_rejected(tmp_path: Path, assignments: dict) -> None:
    source = _write_rows(tmp_path / "labels.jsonl", [_row(number) for number in range(20)])
    destination = tmp_path / "splits"
    destination.mkdir()
    (destination / "split_manifest.json").write_text(
        json.dumps({"version": 2, "assignments": assignments})
    )

    with pytest.raises(ValueError, match="Invalid split manifest"):
        build(source, destination)


def test_a_group_split_across_assignments_is_rejected(tmp_path: Path) -> None:
    rows = [_row(number) for number in range(20)]
    rows[0]["group"] = rows[1]["group"] = "intent"
    source = _write_rows(tmp_path / "labels.jsonl", rows)
    destination = tmp_path / "splits"
    destination.mkdir()
    (destination / "split_manifest.json").write_text(json.dumps({
        "version": 2, "assignments": {"ex-00": "train", "ex-01": "test"},
    }))

    with pytest.raises(ValueError, match="Group intent is assigned across multiple splits"):
        build(source, destination)
