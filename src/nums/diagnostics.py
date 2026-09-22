"""Non-destructive end-to-end diagnostics for NUMS."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .config import Settings
from .ollama import OllamaClient, OllamaError
from .tools import MacTools, TOOL_SCHEMAS
from .wake import voice_dependencies
from .wake import WhisperStream


def self_test(settings: Settings) -> tuple[bool, list[str]]:
    results = []
    ok = True
    tools = MacTools("unrestricted")
    with tempfile.TemporaryDirectory(prefix="nums-self-test-") as directory:
        path = Path(directory) / "probe.txt"
        write = json.loads(tools.write_file({"path": str(path), "content": "NUMS probe"}))
        file_ok = write.get("bytes") == 10 and tools.read_file({"path": str(path)}) == "NUMS probe"
        search = json.loads(tools.search_files({"query": "NUMS", "path": directory}))
        search_ok = search.get("exit_code") == 0 and "NUMS probe" in search.get("output", "")
        ok &= file_ok and search_ok
        results.append(f"File tools: {'ok' if file_ok and search_ok else 'failed'}")

    whisper, model_file = voice_dependencies(settings.whisper_model)
    voice_ok = whisper and model_file
    results.append(f"Voice dependencies: {'ok' if voice_ok else 'incomplete'}")

    system_schema = [item for item in TOOL_SCHEMAS if item["function"]["name"] == "system_info"]
    try:
        response = OllamaClient(settings.ollama_url, settings.model).chat(
            [{"role": "user", "content": "Use the system_info tool now."}], system_schema
        )
        calls = response["message"].get("tool_calls") or []
        model_ok = len(calls) == 1 and calls[0].get("function", {}).get("name") == "system_info"
        results.append(f"Model tool call: {'ok' if model_ok else 'failed'}")
        ok &= model_ok
    except OllamaError as exc:
        results.append(f"Model tool call: failed ({exc})")
        ok = False
    return bool(ok), results


def voice_test(settings: Settings, timeout_seconds: float = 15) -> str | None:
    stream = WhisperStream(settings.whisper_model, settings.capture_device)
    transcripts = stream.transcripts(timeout_seconds=timeout_seconds)
    try:
        return next(transcripts, None)
    finally:
        transcripts.close()
        stream.stop()
