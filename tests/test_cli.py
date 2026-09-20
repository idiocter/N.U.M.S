from contextlib import nullcontext

import pytest

from nums.cli import doctor, wake_mode
from nums.config import Settings
from nums.ollama import OllamaError


def test_wake_listener_survives_model_disconnect(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    transcripts = iter(["hey numnum", "open Safari"])

    class FakeStream:
        def __init__(self, *args: object) -> None:
            pass

        def transcripts(self):
            try:
                yield next(transcripts)
            except StopIteration:
                raise KeyboardInterrupt

        def stop(self) -> None:
            pass

    class FailingAgent:
        def run(self, prompt: str) -> str:
            raise OllamaError("Ollama disconnected")

    monkeypatch.setattr("nums.cli.ListenerLock", nullcontext)
    monkeypatch.setattr("nums.cli.WhisperStream", FakeStream)
    monkeypatch.setattr("nums.cli.subprocess.run", lambda *args, **kwargs: None)

    wake_mode(FailingAgent(), Settings())  # type: ignore[arg-type]

    output = capsys.readouterr().out
    assert "NUMS error > Ollama disconnected" in output
    assert "NUMS wake listener stopped" in output


def test_doctor_returns_failure_when_model_is_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("nums.cli.OllamaClient.has_model", lambda self: False)
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda _: (True, True))

    assert doctor(Settings()) == 1
    assert "ollama serve" in capsys.readouterr().out
