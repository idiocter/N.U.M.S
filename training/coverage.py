"""Report labeled-example coverage for every NUMS runtime tool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nums.tools import TOOL_SCHEMAS
try:
    from training.build_dataset import load_examples
except ModuleNotFoundError:  # direct script execution from the training directory
    from build_dataset import load_examples


def summarize(source: Path, minimum_per_tool: int = 2) -> dict:
    examples = load_examples(source)
    counts = Counter(item["tool"] for item in examples)
    runtime_tools = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    missing = [name for name in runtime_tools if counts[name] == 0]
    underrepresented = [
        name for name in runtime_tools if 0 < counts[name] < minimum_per_tool
    ]
    return {
        "examples": len(examples),
        "runtime_tools": len(runtime_tools),
        "no_tool_examples": counts["none"],
        "minimum_per_tool": minimum_per_tool,
        "missing": missing,
        "underrepresented": underrepresented,
        "counts": {name: counts[name] for name in [*runtime_tools, "none"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--minimum-per-tool", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.minimum_per_tool < 1:
        raise SystemExit("--minimum-per-tool must be positive")
    try:
        report = summarize(args.source, args.minimum_per_tool)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2))
    if args.require_complete and (report["missing"] or report["underrepresented"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
