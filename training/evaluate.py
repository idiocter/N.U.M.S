"""Compare first tool calls against held-out labels without executing any tools."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


def evaluate(data: Path, model: str, url: str) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for line in data.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
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
        with urllib.request.urlopen(request, timeout=180) as response:
            try:
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
                error = None
            except (ValueError, TypeError) as exc:
                calls = []
                function = {}
                actual = None
                error = f"Invalid model response: {exc}"
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
    args = parser.parse_args()
    report = evaluate(args.data, args.model, args.url)
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
