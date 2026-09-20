import json
import stat
from pathlib import Path
from typing import Any

import pytest

from nums.agent import Agent
from nums.config import Settings
from nums.history import HistoryStore


def test_opt_in_history_survives_restart_and_is_private(tmp_path: Path) -> None:
    class ReplyClient:
        def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
            return {"message": {"role": "assistant", "content": "hello"}}

    path = tmp_path / "private" / "history.json"
    settings = Settings(history_file=str(path))
    first = Agent(settings)
    first.client = ReplyClient()  # type: ignore[assignment]
    assert first.run("hi") == "hello"

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())[0]["content"] == "hi"
    second = Agent(settings)
    assert [item["role"] for item in second.messages] == ["system", "user", "assistant"]


def test_corrupt_history_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="Cannot read NUMS history"):
        HistoryStore(path).load()
