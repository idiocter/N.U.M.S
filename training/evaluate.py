"""Compare first tool calls against held-out labels without executing any tools."""

from __future__ import annotations

import argparse
import json
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
            result = json.load(response)
        calls = result.get("message", {}).get("tool_calls") or []
        function = calls[0].get("function", {}) if calls else {}
        actual = function.get("arguments", {})
        if isinstance(actual, str):
            try:
                actual = json.loads(actual)
            except json.JSONDecodeError:
                actual = {}
        if item["expected_tool"] == "none":
            name_ok = len(calls) == 0
            args_ok = name_ok
        else:
            name_ok = len(calls) == 1 and function.get("name") == item["expected_tool"]
            args_ok = name_ok and actual == item["expected_arguments"]
        cases.append(
            {
                "id": item["id"],
                "expected_tool": item["expected_tool"],
                "actual_tool": function.get("name") if calls else "none",
                "tool_correct": name_ok,
                "arguments_correct": args_ok,
            }
        )
        print(f"{item['id']}: tool={'ok' if name_ok else 'wrong'} args={'ok' if args_ok else 'wrong'}")

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
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Tool choice: {report['tool_correct']}/{report['total']}; "
        f"tool and arguments: {report['arguments_correct']}/{report['total']}"
    )


if __name__ == "__main__":
    main()
