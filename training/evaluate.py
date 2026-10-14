"""Compare first tool calls against held-out labels without executing any tools."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_cases(data: Path) -> list[dict[str, Any]]:
    cases = []
    seen = set()
    for number, line in enumerate(data.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("case must be a JSON object")
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("id must be a nonempty string")
            if identifier in seen:
                raise ValueError(f"duplicate id: {identifier}")
            if not isinstance(item.get("messages"), list) or len(item["messages"]) < 2:
                raise ValueError("messages must contain the prompt and expected reply")
            if not all(isinstance(message, dict) for message in item["messages"]):
                raise ValueError("every message must be an object")
            if not isinstance(item.get("tools"), list):
                raise ValueError("tools must be a list")
            if not isinstance(item.get("expected_tool"), str) or not item["expected_tool"]:
                raise ValueError("expected_tool must be a nonempty string")
            if not isinstance(item.get("expected_arguments"), dict):
                raise ValueError("expected_arguments must be an object")
            fingerprint = item.get("case_fingerprint")
            if fingerprint is not None and (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise ValueError("case_fingerprint must be a lowercase SHA-256 digest")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{data}:{number}: {exc}") from exc
        seen.add(identifier)
        cases.append(item)
    return cases


def _build_report(model: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    per_tool: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tool_correct": 0, "arguments_correct": 0}
    )
    for case in cases:
        scores = per_tool[case["expected_tool"]]
        scores["total"] += 1
        scores["tool_correct"] += int(case["tool_correct"])
        scores["arguments_correct"] += int(case["arguments_correct"])
    total = len(cases)
    tool_correct = sum(int(case["tool_correct"]) for case in cases)
    arguments_correct = sum(int(case["arguments_correct"]) for case in cases)
    return {
        "model": model,
        "total": total,
        "tool_correct": tool_correct,
        "arguments_correct": arguments_correct,
        "tool_accuracy": tool_correct / total if total else 0,
        "arguments_accuracy": arguments_correct / total if total else 0,
        "by_expected_tool": dict(sorted(per_tool.items())),
        "cases": cases,
    }


def _resume_cases(prior: dict[str, Any] | None, model: str) -> dict[str, dict[str, Any]]:
    if prior is None:
        return {}
    if prior.get("model") != model:
        raise ValueError("resume report model does not match the requested model")
    rows = prior.get("cases")
    if not isinstance(rows, list):
        raise ValueError("resume report must contain cases")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("resume report contains an invalid case")
        if row["id"] in result:
            raise ValueError(f"resume report contains duplicate id: {row['id']}")
        result[row["id"]] = row
    return result


def evaluate(
    data: Path, model: str, url: str, prior: dict[str, Any] | None = None,
    checkpoint: Path | None = None, retries: int = 0,
) -> dict[str, Any]:
    if not 0 <= retries <= 3:
        raise ValueError("retries must be between 0 and 3")
    cases: list[dict[str, Any]] = []
    resumed = _resume_cases(prior, model)
    for item in load_cases(data):
        previous = resumed.get(item["id"])
        if previous is not None and not previous.get("error"):
            if previous.get("case_fingerprint") != item.get("case_fingerprint"):
                raise ValueError(f"resume fingerprint changed for {item['id']}")
            if (
                previous.get("expected_tool") != item["expected_tool"]
                or previous.get("expected_arguments") != item["expected_arguments"]
                or not isinstance(previous.get("tool_correct"), bool)
                or not isinstance(previous.get("arguments_correct"), bool)
            ):
                raise ValueError(f"resume label or scores changed for {item['id']}")
            cases.append(previous)
            print(f"{item['id']}: resumed")
            if checkpoint:
                write_report(checkpoint, _build_report(model, cases))
            continue
        payload = json.dumps({
            "model": model,
            "messages": item["messages"][:-1],
            "tools": item["tools"],
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        request = urllib.request.Request(
            f"{url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    result = json.load(response)
                    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
                        raise ValueError("missing assistant message")
                    calls = result["message"].get("tool_calls") or []
                    if not isinstance(calls, list) or any(
                        not isinstance(call, dict) or not isinstance(call.get("function"), dict)
                        for call in calls
                    ):
                        raise ValueError("malformed tool calls")
                    function = calls[0]["function"] if calls else {}
                    actual = function.get("arguments", {})
                    if isinstance(actual, str):
                        actual = json.loads(actual)
                    if not isinstance(actual, dict):
                        raise ValueError("tool arguments are not a JSON object")
                    error = None
                break
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                calls = []
                function = {}
                actual = None
                error = f"Invalid model response: {exc}"
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                calls = []
                function = {}
                actual = None
                error = f"Evaluation request failed: {type(exc).__name__}: {exc}"
                if attempt < retries:
                    print(f"{item['id']}: transient request failure; retry {attempt + 1}/{retries}")
        if error:
            name_ok = args_ok = False
        elif item["expected_tool"] == "none":
            name_ok = len(calls) == 0
            args_ok = name_ok
        else:
            name_ok = len(calls) == 1 and function.get("name") == item["expected_tool"]
            args_ok = name_ok and actual == item["expected_arguments"]
        cases.append(
            {
                "id": item["id"],
                "case_fingerprint": item.get("case_fingerprint"),
                "expected_tool": item["expected_tool"],
                "actual_tool": function.get("name") if calls else (None if error else "none"),
                "expected_arguments": item["expected_arguments"],
                "actual_arguments": actual,
                "tool_correct": name_ok,
                "arguments_correct": args_ok,
                "error": error,
            }
        )
        print(
            f"{item['id']}: tool={'ok' if name_ok else 'wrong'} "
            f"args={'ok' if args_ok else 'wrong'}" + (f" ({error})" if error else "")
        )
        if checkpoint:
            write_report(checkpoint, _build_report(model, cases))
    return _build_report(model, cases)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--json-out", type=Path, help="Write a detailed JSON report")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed cases in --json-out and checkpoint after every case",
    )
    parser.add_argument(
        "--retries", type=int, choices=range(4), default=0,
        help="Retry each transient Ollama request up to 3 times",
    )
    args = parser.parse_args()
    if args.resume and args.json_out is None:
        parser.error("--resume requires --json-out")
    prior = None
    if args.resume and args.json_out.exists():
        prior = json.loads(args.json_out.read_text())
    report = evaluate(
        args.data, args.model, args.url, prior=prior,
        checkpoint=args.json_out if args.resume else None, retries=args.retries,
    )
    if not report["total"]:
        raise SystemExit("empty evaluation set")
    if args.json_out:
        write_report(args.json_out, report)
    print(
        f"Tool choice: {report['tool_correct']}/{report['total']}; "
        f"tool and arguments: {report['arguments_correct']}/{report['total']}"
    )


if __name__ == "__main__":
    main()
