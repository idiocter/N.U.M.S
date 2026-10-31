import subprocess
from contextlib import nullcontext

import pytest

from nums.cli import (
    _ollama_reachable, _play_sound, _say, doctor, ensure_ollama_running, main, prepare_voice,
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
            yield from transcripts
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
            yield from transcripts
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


def test_sound_cue_uses_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        "nums.cli.subprocess.run", lambda command, **kwargs: calls.append((command, kwargs))
    )

    _play_sound("/System/Library/Sounds/Glass.aiff")

    assert calls == [(["afplay", "/System/Library/Sounds/Glass.aiff"], {"check": False, "timeout": 30})]


@pytest.mark.parametrize("failure", [FileNotFoundError("afplay"), subprocess.TimeoutExpired("afplay", 30)])
def test_sound_cue_failure_is_ignored(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("nums.cli.subprocess.run", fail)

    _play_sound("/System/Library/Sounds/Glass.aiff")


def test_failed_service_install_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(settings: Settings) -> None:
        raise subprocess.CalledProcessError(5, ["launchctl", "bootstrap"])

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--install-service"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.install_service", fail)

    with pytest.raises(SystemExit, match="Could not install the NUMS service"):
        main()


def test_failed_service_removal_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--uninstall-service"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.uninstall_service", fail)

    with pytest.raises(SystemExit, match="Could not remove the NUMS service.*not permitted"):
        main()


def test_failed_voice_model_download_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: str) -> None:
        raise subprocess.CalledProcessError(22, ["curl"])

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--setup-voice"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.download_voice_model", fail)

    with pytest.raises(SystemExit, match="Could not set up the Whisper voice model"):
        main()


def test_voice_test_failure_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(settings: Settings) -> None:
        raise RuntimeError("whisper-stream is missing; run `brew install whisper-cpp`")

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--voice-test"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.voice_test", fail)

    with pytest.raises(SystemExit, match="whisper-stream is missing"):
        main()


def _patch_startup(monkeypatch: pytest.MonkeyPatch, settings: Settings, agent_class: type) -> None:
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: settings)
    monkeypatch.setattr("nums.cli.Agent", agent_class)
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.verify_text_model", lambda settings: None)


def test_interactive_help_status_and_reset_commands(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    prompts = iter(["/help", "/status", "/reset", "/quit", "never reached"])
    resets = []

    class FakeAgent:
        messages = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}]
        last_run = {"status": "completed", "steps": 2, "tool_calls": 1, "tool_errors": 0}
        trace_error = "trace limit reached"

        def __init__(self, settings: Settings) -> None:
            pass

        def reset(self) -> None:
            resets.append(True)

        def run(self, prompt: str) -> str:
            pytest.fail("commands must not reach the model")

    monkeypatch.setattr("nums.cli.sys.argv", ["nums"])
    _patch_startup(monkeypatch, Settings(model="qwen:test"), FakeAgent)
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))

    main()

    output = capsys.readouterr().out
    assert "/reset — clear conversation" in output
    assert "Model: qwen:test | History: memory only | Messages: 2" in output
    assert "Last run: completed | Steps: 2 | Tools: 1 | Tool errors: 0" in output
    assert "Trace error: trace limit reached" in output
    assert "Conversation cleared." in output
    assert resets == [True]


def test_one_shot_request_prints_and_speaks_the_response(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    spoken = []

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, prompt: str) -> str:
            return f"answer to {prompt}"

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--speak", "what", "time"])
    _patch_startup(monkeypatch, Settings(), FakeAgent)
    monkeypatch.setattr("nums.cli._say", spoken.append)

    main()

    assert capsys.readouterr().out.strip() == "answer to what time"
    assert spoken == ["answer to what time"]


def test_one_shot_model_failure_exits_with_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingAgent:
        def __init__(self, settings: Settings) -> None:
            pass

        def run(self, prompt: str) -> str:
            raise OllamaError("Ollama disconnected")

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "hello"])
    _patch_startup(monkeypatch, Settings(), FailingAgent)

    with pytest.raises(SystemExit, match="Ollama disconnected"):
        main()


def test_voice_mode_prepares_checks_then_listens_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []

    class FakeAgent:
        def __init__(self, settings: Settings) -> None:
            events.append("agent")

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--voice"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.Agent", FakeAgent)
    monkeypatch.setattr("nums.cli.prepare_voice", lambda settings: events.append("prepare"))
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: events.append("ollama"))
    monkeypatch.setattr("nums.cli.verify_voice_model", lambda settings: events.append("verify"))
    monkeypatch.setattr("nums.cli.wake_mode", lambda agent, settings: events.append("listen"))

    main()

    assert events == ["prepare", "ollama", "verify", "agent", "listen"]


def test_doctor_passes_when_everything_is_ready(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("nums.cli.OllamaClient.has_model", lambda self: True)
    monkeypatch.setattr("nums.cli.probe_model_tool_call", lambda settings: (True, None))
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda path: (True, False))
    monkeypatch.setattr("nums.cli.probe_mac_ui", lambda: (True, None))

    assert doctor(Settings(model="qwen:test")) == 0

    output = capsys.readouterr().out
    assert "Model: qwen:test" in output
    assert "Model listed by Ollama: yes" in output
    assert "Model tool call: ok" in output
    assert "Whisper stream: yes" in output and "Wake model: no" in output
    assert "Mac app UI: ok" in output


def test_doctor_fails_when_mac_ui_is_blocked(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("nums.cli.OllamaClient.has_model", lambda self: True)
    monkeypatch.setattr("nums.cli.probe_model_tool_call", lambda settings: (True, None))
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda path: (True, True))
    monkeypatch.setattr("nums.cli.probe_mac_ui", lambda: (False, "grant Accessibility access"))

    assert doctor(Settings()) == 1
    assert "Mac app UI: failed (grant Accessibility access)" in capsys.readouterr().out


def test_prepare_voice_requires_whisper_and_downloads_a_missing_model(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda path: (False, False))
    with pytest.raises(SystemExit, match="brew install whisper-cpp"):
        prepare_voice(Settings())

    downloads = []
    monkeypatch.setattr("nums.cli.voice_dependencies", lambda path: (True, False))
    monkeypatch.setattr("nums.cli.download_voice_model", downloads.append)
    prepare_voice(Settings(whisper_model="/models/voice.bin"))
    assert downloads == ["/models/voice.bin"]
    assert "Whisper voice model ready." in capsys.readouterr().out

    monkeypatch.setattr("nums.cli.voice_dependencies", lambda path: (True, True))
    monkeypatch.setattr("nums.cli.download_voice_model", lambda path: pytest.fail("downloaded"))
    prepare_voice(Settings())


def test_voice_model_check_reports_a_model_that_cannot_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self, messages, tools):
        raise OllamaError("model failed to load")

    monkeypatch.setattr("nums.cli.OllamaClient.chat", fail)
    with pytest.raises(SystemExit, match="not ready for voice mode: model failed to load"):
        verify_voice_model(Settings())

    monkeypatch.setattr("nums.cli.OllamaClient.chat", lambda self, messages, tools: {"message": {}})
    verify_voice_model(Settings())


def test_ollama_start_fails_clearly_without_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli._ollama_reachable", lambda _: False)
    monkeypatch.setattr("nums.cli.shutil.which", lambda _: None)

    with pytest.raises(SystemExit, match="Ollama is missing"):
        ensure_ollama_running(Settings())


@pytest.mark.parametrize("url", ["http://192.168.1.5:11434", "http://localhost:9999", "https://localhost:11434"])
def test_only_the_default_local_ollama_is_ever_started(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setattr("nums.cli._ollama_reachable", lambda _: False)
    monkeypatch.setattr("nums.cli.shutil.which", lambda _: pytest.fail("looked for ollama"))

    with pytest.raises(SystemExit, match="Cannot reach Ollama"):
        ensure_ollama_running(Settings(ollama_url=url))


def test_ollama_launch_failure_and_early_exit_are_reported(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("nums.cli._ollama_reachable", lambda _: False)
    monkeypatch.setattr("nums.cli.shutil.which", lambda _: "/usr/local/bin/ollama")
    monkeypatch.setattr("nums.cli.Path.home", lambda: tmp_path)
    monkeypatch.setattr("nums.cli.time.sleep", lambda _: None)

    def cannot_launch(*args: object, **kwargs: object) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr("nums.cli.subprocess.Popen", cannot_launch)
    with pytest.raises(SystemExit, match="Could not start Ollama: .*not permitted"):
        ensure_ollama_running(Settings())

    class CrashedProcess:
        def poll(self) -> int:
            return 1

    monkeypatch.setattr("nums.cli.subprocess.Popen", lambda *args, **kwargs: CrashedProcess())
    with pytest.raises(SystemExit, match="Ollama did not start. Check .*NUMS-Ollama.log"):
        ensure_ollama_running(Settings())
    assert (tmp_path / "Library" / "Logs").is_dir()


def test_invalid_environment_settings_exit_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def invalid() -> Settings:
        raise ValueError("NUMS_HISTORY_TURNS must be positive")

    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--doctor"])
    monkeypatch.setattr("nums.cli.Settings.from_env", invalid)

    with pytest.raises(SystemExit, match="NUMS_HISTORY_TURNS must be positive"):
        main()


def test_voice_test_without_speech_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--voice-test"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.voice_test", lambda settings: None)

    with pytest.raises(SystemExit, match="No speech was transcribed"):
        main()


def test_voice_test_prints_the_transcript(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--voice-test"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.voice_test", lambda settings: "hello there")

    main()

    assert "Transcript: hello there" in capsys.readouterr().out


def test_pull_without_the_ollama_command_exits_with_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--pull"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings())
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.shutil.which", lambda name: None)

    with pytest.raises(SystemExit, match="ollama command is missing"):
        main()


def test_pull_exits_with_the_ollama_status(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--pull"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings(model="qwen:test"))
    monkeypatch.setattr("nums.cli.ensure_ollama_running", lambda settings: None)
    monkeypatch.setattr("nums.cli.shutil.which", lambda name: "/bin/ollama")
    monkeypatch.setattr("nums.cli.subprocess.call", lambda command: calls.append(command) or 7)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 7
    assert calls == [["/bin/ollama", "pull", "qwen:test"]]


def test_setup_voice_reports_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    downloads = []
    monkeypatch.setattr("nums.cli.sys.argv", ["nums", "--setup-voice"])
    monkeypatch.setattr("nums.cli.Settings.from_env", lambda: Settings(whisper_model="/tmp/m.bin"))
    monkeypatch.setattr("nums.cli.download_voice_model", downloads.append)

    main()

    assert downloads == ["/tmp/m.bin"]
    assert "Wake model ready." in capsys.readouterr().out


class _ScriptedStream:
    """A microphone stream that plays a script, then interrupts the listener."""

    instances: list["_ScriptedStream"] = []

    def __init__(self, script, *args: object) -> None:
        self.script = script
        self.stopped = False
        _ScriptedStream.instances.append(self)

    def transcripts(self):
        for item in self.script():
            yield item
        raise KeyboardInterrupt

    def stop(self) -> None:
        self.stopped = True


def test_speech_heard_while_the_model_works_is_answered_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    thinking = threading.Event()
    heard_while_thinking = threading.Event()
    prompts = []

    class SlowAgent:
        def run(self, prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                thinking.set()
                assert heard_while_thinking.wait(5)
            return f"done {len(prompts)}"

    def script():
        yield "hey numnum open Safari"
        assert thinking.wait(5)
        yield "and then open Notes"
        heard_while_thinking.set()
        # Stay open until the second answer finishes, as a real microphone would.
        deadline = threading.Event()
        deadline.wait(0.5)

    _ScriptedStream.instances = []
    spoken = []
    monkeypatch.setattr("nums.cli.ListenerLock", nullcontext)
    monkeypatch.setattr("nums.cli.WhisperStream", lambda *a: _ScriptedStream(script, *a))
    monkeypatch.setattr("nums.cli._say", spoken.append)
    monkeypatch.setattr("nums.cli.time.sleep", lambda seconds: None)

    wake_mode(SlowAgent(), Settings())  # type: ignore[arg-type]

    assert prompts == ["open Safari", "and then open Notes"]
    assert spoken == ["done 1", "done 2"]


def test_one_microphone_stream_serves_many_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    class Agent:
        def run(self, prompt: str) -> str:
            return prompt

    def script():
        yield "hey numnum"
        yield "first request"
        yield "second request"

    _ScriptedStream.instances = []
    monkeypatch.setattr("nums.cli.ListenerLock", nullcontext)
    monkeypatch.setattr("nums.cli.WhisperStream", lambda *a: _ScriptedStream(script, *a))
    monkeypatch.setattr("nums.cli._say", lambda text: None)
    monkeypatch.setattr("nums.cli._play_sound", lambda path: None)
    monkeypatch.setattr("nums.cli.time.sleep", lambda seconds: None)

    wake_mode(Agent(), Settings())  # type: ignore[arg-type]

    assert len(_ScriptedStream.instances) == 1
    assert _ScriptedStream.instances[0].stopped


def test_speech_during_the_spoken_reply_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The microphone hears NUMS's own voice while it speaks; that must not become a command."""
    prompts = []

    class Agent:
        def run(self, prompt: str) -> str:
            prompts.append(prompt)
            return "reply"

    def script():
        yield "hey numnum"
        yield "open Safari"
        while not spoke.is_set():
            spoke.wait(0.01)
        yield "echo of the reply"

    import threading

    spoke = threading.Event()

    def say(text: str) -> None:
        spoke.set()
        # Give the listener thread time to hear the echo before the reply ends.
        threading.Event().wait(0.2)

    _ScriptedStream.instances = []
    monkeypatch.setattr("nums.cli.ListenerLock", nullcontext)
    monkeypatch.setattr("nums.cli.WhisperStream", lambda *a: _ScriptedStream(script, *a))
    monkeypatch.setattr("nums.cli._say", say)
    monkeypatch.setattr("nums.cli._play_sound", lambda path: None)
    monkeypatch.setattr("nums.cli.time.sleep", lambda seconds: None)

    wake_mode(Agent(), Settings())  # type: ignore[arg-type]

    assert prompts == ["open Safari"]
