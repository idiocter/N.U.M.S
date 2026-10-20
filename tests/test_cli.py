import subprocess
from contextlib import nullcontext

import pytest

from nums.cli import (
    _ollama_reachable, _say, doctor, ensure_ollama_running, main, prepare_voice,
    verify_text_model, verify_voice_model, wake_mode,
)
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
    monkeypatch.setattr("nums.cli.probe_mac_ui", lambda: (True, None))

    assert doctor(Settings()) == 1
    assert "ollama serve" in capsys.readouterr().out


def test_doctor_reports_installed_model_that_cannot_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("nums.cli.OllamaClient.has_model", lambda self: True)
    monkeypatch.setattr(
        "nums.cli.probe_model_tool_call", lambda settings: (False, "model could not load"),
    )
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda _: (True, True))
    monkeypatch.setattr("nums.cli.probe_mac_ui", lambda: (True, None))

    assert doctor(Settings()) == 1
    output = capsys.readouterr().out
    assert "Model listed by Ollama: yes" in output
    assert "Model tool call: failed (model could not load)" in output


def test_doctor_reports_mac_ui_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("nums.cli.OllamaClient.has_model", lambda self: True)
    monkeypatch.setattr("nums.cli.probe_model_tool_call", lambda settings: (True, None))
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda _: (True, True))
    monkeypatch.setattr("nums.cli.probe_mac_ui", lambda: (False, "System Events unavailable"))

    assert doctor(Settings()) == 1
    assert "Mac app UI: failed (System Events unavailable)" in capsys.readouterr().out


def test_wake_listener_pauses_after_empty_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    launches = 0
    pauses = []

    class EmptyStream:
        def __init__(self, *args: object) -> None:
            nonlocal launches
            launches += 1

        def transcripts(self):
            if launches > 1:
                raise KeyboardInterrupt
            if False:
                yield "unused"

        def stop(self) -> None:
            pass

    monkeypatch.setattr("nums.cli.ListenerLock", nullcontext)
    monkeypatch.setattr("nums.cli.WhisperStream", EmptyStream)
    monkeypatch.setattr("nums.cli.time.sleep", pauses.append)

    wake_mode(object(), Settings())  # type: ignore[arg-type]

    assert launches == 2
    assert pauses == [1]


def test_voice_preparation_downloads_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    downloads = []
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda _: (True, False))
    monkeypatch.setattr("nums.cli.download_voice_model", downloads.append)

    prepare_voice(Settings(whisper_model="/tmp/test-whisper.bin"))

    assert downloads == ["/tmp/test-whisper.bin"]


def test_voice_preparation_reuses_existing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda _: (True, True))
    monkeypatch.setattr(
        "nums.cli.download_voice_model", lambda _: pytest.fail("downloaded existing model"),
    )

    prepare_voice(Settings())


def test_voice_preparation_explains_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda _: (False, False))

    with pytest.raises(SystemExit, match="brew install whisper-cpp"):
        prepare_voice(Settings())


@pytest.mark.parametrize("flag", ["--voice", "--wake"])
def test_voice_flags_start_wake_mode(monkeypatch: pytest.MonkeyPatch, flag: str) -> None:
    calls = []
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", flag])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.prepare_voice", lambda settings: calls.append("prepared"))
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: calls.append("ollama"))
    monkeypatch.setattr("nums.cli.verify_voice_model", lambda settings: None)
    monkeypatch.setattr("nums.cli.Agent", lambda settings: object())
    monkeypatch.setattr("nums.cli.wake_mode", lambda agent, settings: calls.append("voice"))

    main()

    assert calls == ["prepared", "ollama", "voice"]


def test_default_command_starts_text_mode(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.Agent", lambda settings: object())
    monkeypatch.setattr("nums.cli.prepare_voice", lambda settings: pytest.fail("voice setup ran"))
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.verify_text_model", lambda settings: None)
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError))

    main()

    assert "Type /help for commands" in capsys.readouterr().out


def test_text_mode_explains_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.OllamaClient.has_model", lambda self: False)
    with pytest.raises(SystemExit, match="uv run nums --pull"):
        verify_text_model(Settings())


def test_local_ollama_starts_when_needed(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    checks = iter([False, False, True])
    launches = []

    class Process:
        def poll(self):
            return None

    monkeypatch.setattr("nums.cli._ollama_reachable", lambda _: next(checks))
    monkeypatch.setattr("nums.cli.shutil.which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr("nums.cli.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "nums.cli.subprocess.Popen",
        lambda command, **kwargs: launches.append(command) or Process(),
    )
    monkeypatch.setattr("nums.cli.time.sleep", lambda _: None)

    ensure_ollama_running(Settings())

    assert launches == [["/usr/local/bin/ollama", "serve"]]
    assert "Started local Ollama" in capsys.readouterr().out


def test_reachability_probe_swallows_socket_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("name resolution failed")

    monkeypatch.setattr("nums.cli.urllib.request.urlopen", fail)

    assert _ollama_reachable("http://127.0.0.1:11434") is False


def test_remote_ollama_is_not_started_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli._ollama_reachable", lambda _: False)
    monkeypatch.setattr("nums.cli.subprocess.Popen", lambda *args, **kwargs: pytest.fail("started"))

    with pytest.raises(SystemExit, match="Cannot reach Ollama"):
        ensure_ollama_running(Settings(ollama_url="https://example.com"))


def test_running_ollama_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli._ollama_reachable", lambda _: True)
    monkeypatch.setattr("nums.cli.subprocess.Popen", lambda *args, **kwargs: pytest.fail("started"))

    ensure_ollama_running(Settings())


def test_voice_mode_stops_before_microphone_when_model_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--voice"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.prepare_voice", lambda settings: None)
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    def failed_model(settings):
        raise SystemExit("The local model is not ready for voice mode: load failed")

    monkeypatch.setattr("nums.cli.verify_voice_model", failed_model)
    monkeypatch.setattr("nums.cli.wake_mode", lambda *args: pytest.fail("microphone started"))

    with pytest.raises(SystemExit, match="load failed"):
        main()


def test_voice_model_probe_reports_ollama_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise OllamaError("Metal allocation failed")

    monkeypatch.setattr("nums.cli.OllamaClient.chat", fail)

    with pytest.raises(SystemExit, match="Metal allocation failed"):
        verify_voice_model(Settings())


def test_say_guards_option_like_text_and_uses_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "nums.cli.subprocess.run", lambda command, **kwargs: calls.append((command, kwargs))
    )

    _say("-v hello")

    assert calls == [(["say", "--", "-v hello"], {"check": False, "timeout": 120})]


@pytest.mark.parametrize("failure", [FileNotFoundError("say"), subprocess.TimeoutExpired("say", 120)])
def test_say_swallows_speech_failures(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("nums.cli.subprocess.run", fail)

    _say("hello")


def test_blank_one_shot_request_is_rejected_before_starting_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "  ", ""])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: pytest.fail("started ollama"))

    with pytest.raises(SystemExit, match="request is empty"):
        main()


def test_one_shot_request_is_trimmed(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    prompts = []

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, prompt: str) -> str:
            prompts.append(prompt)
            return "done"

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", " open", "Safari "])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.Agent", FakeAgent)
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.verify_text_model", lambda settings: None)

    main()

    assert prompts == ["open Safari"]
    assert capsys.readouterr().out.strip() == "done"


def test_pull_explains_missing_ollama_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--pull"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.shutil.which", lambda _: None)
    monkeypatch.setattr("nums.cli.subprocess.call", lambda *args, **kwargs: pytest.fail("pulled"))

    with pytest.raises(SystemExit, match="ollama command is missing"):
        main()


def test_pull_runs_the_resolved_ollama_command(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--pull"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings(model="qwen:test"))
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.shutil.which", lambda _: "/opt/bin/ollama")
    monkeypatch.setattr("nums.cli.subprocess.call", lambda command: commands.append(command) or 0)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 0
    assert commands == [["/opt/bin/ollama", "pull", "qwen:test"]]


def test_interrupting_a_request_keeps_the_session_running(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    prompts = iter(["slow request", "quick request"])
    runs = []

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, prompt: str) -> str:
            runs.append(prompt)
            if prompt == "slow request":
                raise KeyboardInterrupt
            return "done"

    def read_prompt(_: str) -> str:
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("nums.cli.sys.argv", ["nums"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.Agent", FakeAgent)
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.verify_text_model", lambda settings: None)
    monkeypatch.setattr("builtins.input", read_prompt)

    main()

    output = capsys.readouterr().out
    assert runs == ["slow request", "quick request"]
    assert "Request interrupted" in output
    assert "NUMS > done" in output


def test_mistyped_slash_command_is_not_sent_to_the_model(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    prompts = iter(["/quite", "/tmp/notes.txt"])
    runs = []

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, prompt: str) -> str:
            runs.append(prompt)
            return "ok"

    def read_prompt(_: str) -> str:
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("nums.cli.sys.argv", ["nums"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.Agent", FakeAgent)
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.verify_text_model", lambda settings: None)
    monkeypatch.setattr("builtins.input", read_prompt)

    main()

    assert "Unknown command /quite" in capsys.readouterr().out
    assert runs == ["/tmp/notes.txt"]


def test_context_limit_too_small_for_the_system_prompt_fails_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "hello"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings(max_context_chars=1000))
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: pytest.fail("started ollama"))

    with pytest.raises(SystemExit, match="NUMS_MAX_CONTEXT_CHARS must be at least"):
        main()


@pytest.mark.parametrize("failure", [OSError("disk full"), ValueError("Cannot save incomplete NUMS history")])
def test_storage_failure_does_not_end_the_session(
    monkeypatch: pytest.MonkeyPatch, capsys, failure: Exception
) -> None:
    prompts = iter(["first", "second"])
    runs = []

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, prompt: str) -> str:
            runs.append(prompt)
            if prompt == "first":
                raise failure
            return "fine"

    def read_prompt(_: str) -> str:
        try:
            return next(prompts)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("nums.cli.sys.argv", ["nums"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.Agent", FakeAgent)
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.verify_text_model", lambda settings: None)
    monkeypatch.setattr("builtins.input", read_prompt)

    main()

    output = capsys.readouterr().out
    assert runs == ["first", "second"]
    assert f"NUMS error > {failure}" in output
    assert "NUMS > fine" in output


@pytest.mark.parametrize("failure", [OSError("disk full"), ValueError("bad history")])
def test_wake_listener_survives_storage_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Exception
) -> None:
    transcripts = iter(["hey numnum, open Safari", "open Notes"])
    spoken = []

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
            raise failure

    monkeypatch.setattr("nums.cli.ListenerLock", nullcontext)
    monkeypatch.setattr("nums.cli.WhisperStream", FakeStream)
    monkeypatch.setattr("nums.cli._say", spoken.append)
    monkeypatch.setattr("nums.cli.subprocess.run", lambda *args, **kwargs: None)

    wake_mode(FailingAgent(), Settings())  # type: ignore[arg-type]

    output = capsys.readouterr().out
    assert f"NUMS error > {failure}" in output
    assert "NUMS wake listener stopped" in output
    assert spoken == ["I hit an internal error."] * 2
