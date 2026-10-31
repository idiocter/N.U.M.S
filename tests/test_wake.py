import os
import stat
from pathlib import Path

import pytest

from nums.wake import ListenerLock, WakePhraseDetector, WhisperStream, download_voice_model, voice_dependencies


def test_wake_phrase_with_command() -> None:
    event = WakePhraseDetector().feed("Hey Num Num, open Safari")

    assert event is not None
    assert event.kind == "command"
    assert event.text == "open Safari"


def test_common_phonetic_transcription_wakes() -> None:
    event = WakePhraseDetector().feed("Hey Newman, open my calendar")

    assert event is not None
    assert event.kind == "command"
    assert event.text == "open my calendar"


def test_command_can_follow_wake_phrase() -> None:
    detector = WakePhraseDetector()

    wake = detector.feed("Hey numnum")
    command = detector.feed("what is the time")

    assert wake is not None and wake.kind == "wake"
    assert command is not None and command.text == "what is the time"


def test_unrelated_speech_does_not_wake() -> None:
    assert WakePhraseDetector().feed("open Safari") is None


def test_mentioning_wake_phrase_inside_sentence_does_not_wake() -> None:
    assert WakePhraseDetector().feed("I heard someone say hey numnum today") is None


def test_repeated_transcript_is_ignored() -> None:
    detector = WakePhraseDetector()

    assert detector.feed("Hey numnum") is not None
    assert detector.feed("Hey numnum") is None


def test_overlapping_windows_do_not_repeat_a_command() -> None:
    detector = WakePhraseDetector()

    first = detector.feed("Hey Num Num, tell me the current time.")
    duplicate = detector.feed("Hey Num Num tell me the current time")

    assert first is not None and first.kind == "command"
    assert duplicate is None


def test_wake_starts_a_continuous_conversation() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0

    wake = detector.feed("Hey num num")
    first = detector.feed("open Safari")
    second = detector.feed("now open my calendar")

    assert wake is not None and wake.kind == "wake"
    assert first is not None and first.text == "open Safari"
    assert second is not None and second.text == "now open my calendar"


def test_sleep_phrase_ends_the_conversation() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0
    detector.feed("Hey num num")

    sleep = detector.feed("Aight baby girl, let's sleep")
    ignored = detector.feed("open Safari")

    assert sleep is not None and sleep.kind == "sleep"
    assert ignored is None


def test_sleep_words_inside_a_request_do_not_end_session() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0
    detector.feed("Hey num num")

    event = detector.feed("search for the song good night nums")

    assert event is not None and event.kind == "command"
    assert detector.active_session


def test_custom_sleep_phrase_ends_session() -> None:
    detector = WakePhraseDetector(sleep_phrases=("power down",))
    detector.cooldown_seconds = 0
    detector.feed("Hey num num")
    event = detector.feed("power down")
    assert event is not None and event.kind == "sleep"


def test_idle_session_requires_wake_phrase_again(monkeypatch: pytest.MonkeyPatch) -> None:
    moments = iter([0.0, 11.0])
    monkeypatch.setattr("nums.wake.time.monotonic", lambda: next(moments))
    detector = WakePhraseDetector(session_timeout_seconds=10)
    detector.cooldown_seconds = 0

    assert detector.feed("Hey num num") is not None
    assert detector.feed("open Safari") is None


def test_same_wake_phrase_works_after_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    moments = iter([0.0, 11.0])
    monkeypatch.setattr("nums.wake.time.monotonic", lambda: next(moments))
    detector = WakePhraseDetector(session_timeout_seconds=10)

    first = detector.feed("Hey num num")
    second = detector.feed("Hey num num")

    assert first is not None and first.kind == "wake"
    assert second is not None and second.kind == "wake"


def test_sleep_phrase_requires_an_active_session() -> None:
    assert WakePhraseDetector().feed("Aight baby girl let's sleep") is None


def test_phonetic_sleep_transcription_ends_session() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0
    detector.feed("Hey num num")

    event = detector.feed("8 baby girl, lets sleep")

    assert event is not None and event.kind == "sleep"


def test_only_one_listener_can_hold_the_microphone(tmp_path: Path) -> None:
    lock_path = tmp_path / "listener.lock"

    with ListenerLock(lock_path):
        assert lock_path.read_text() == str(os.getpid())
        with pytest.raises(RuntimeError, match="already listening"):
            with ListenerLock(lock_path):
                pass
        assert lock_path.read_text() == str(os.getpid())

    with ListenerLock(lock_path):
        pass


def test_voice_dependencies_rejects_incomplete_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    model = tmp_path / "ggml-base.en.bin"
    model.write_bytes(b"partial")
    assert voice_dependencies(str(model)) == (True, False)
    with model.open("r+b") as handle:
        handle.truncate(50_000_000)
    assert voice_dependencies(str(model)) == (True, True)


def test_voice_download_has_timeout_and_keeps_resume_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands = []

    def failed_download(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"partial")
        raise RuntimeError("offline")

    monkeypatch.setattr("nums.wake.subprocess.run", failed_download)
    model = tmp_path / "voice.bin"
    with pytest.raises(RuntimeError, match="offline"):
        download_voice_model(str(model))
    assert "--max-time" in commands[0]
    assert "--retry" in commands[0]
    assert model.with_suffix(".bin.part").read_bytes() == b"partial"


def test_timestamped_transcript_is_a_command_during_session() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0
    detector.feed("Hey num num")

    event = detector.feed(
        "[00:00:00.000 --> 00:00:02.000] >> open my calendar"
    )

    assert event is not None
    assert event.kind == "command"
    assert event.text == "open my calendar"


def test_same_command_can_be_repeated_after_response() -> None:
    detector = WakePhraseDetector()
    detector.feed("Hey num num")
    first = detector.feed("what is the time")
    detector.command_completed()
    second = detector.feed("what is the time")

    assert first is not None and first.kind == "command"
    assert second is not None and second.kind == "command"


def test_stream_reads_final_transcript_after_process_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.touch()

    class FinishedProcess:
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FinishedProcess:
        output_path = Path(command[command.index("--file") + 1])
        output_path.write_text("hey numnum, open Safari\n")
        return FinishedProcess()

    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    monkeypatch.setattr("nums.wake.subprocess.Popen", fake_popen)
    stream = WhisperStream(str(model))

    assert list(stream.transcripts()) == ["hey numnum, open Safari"]


def test_stream_reports_failed_process_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.touch()

    class FailedProcess:
        returncode = 2

        def poll(self) -> int:
            return 2

    def fake_popen(command: list[str], **kwargs: object) -> FailedProcess:
        kwargs["stdout"].write("microphone unavailable\n")  # type: ignore[union-attr]
        return FailedProcess()

    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    monkeypatch.setattr("nums.wake.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError, match="microphone unavailable"):
        list(WhisperStream(str(model)).transcripts())


def test_stream_failure_tail_is_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.touch()

    class FailedProcess:
        returncode = 2

        def poll(self) -> int:
            return 2

    def fake_popen(command: list[str], **kwargs: object) -> FailedProcess:
        log = kwargs["stdout"]  # type: ignore[index]
        log.write("EARLY-NOISE\n" + "x" * 10000 + "\nTAIL-REASON\n")  # type: ignore[union-attr]
        return FailedProcess()

    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    monkeypatch.setattr("nums.wake.subprocess.Popen", fake_popen)

    with pytest.raises(RuntimeError) as excinfo:
        list(WhisperStream(str(model)).transcripts())

    message = str(excinfo.value)
    assert "TAIL-REASON" in message
    assert "EARLY-NOISE" not in message


def test_stream_explains_startup_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.touch()
    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")

    def fail(*args: object, **kwargs: object) -> None:
        raise PermissionError("microphone access denied")

    monkeypatch.setattr("nums.wake.subprocess.Popen", fail)

    with pytest.raises(RuntimeError, match="Cannot start whisper-stream.*microphone access denied"):
        list(WhisperStream(str(model)).transcripts())


def test_stream_waits_for_complete_transcript_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.touch()
    output_path = None

    class RunningProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int = 3) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> RunningProcess:
        nonlocal output_path
        output_path = Path(command[command.index("--file") + 1])
        output_path.write_text("hey numnum, open")
        return RunningProcess()

    def finish_line(_: float) -> None:
        assert output_path is not None
        output_path.write_text("hey numnum, open Safari\n")

    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    monkeypatch.setattr("nums.wake.subprocess.Popen", fake_popen)
    monkeypatch.setattr("nums.wake.time.sleep", finish_line)
    stream = WhisperStream(str(model))
    transcripts = stream.transcripts()

    assert next(transcripts) == "hey numnum, open Safari"
    transcripts.close()


def test_stream_reads_again_after_transcript_file_is_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.bin"
    model.touch()
    output_path = None

    class RunningProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int = 3) -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> RunningProcess:
        nonlocal output_path
        output_path = Path(command[command.index("--file") + 1])
        output_path.write_text("first longer phrase\n")
        return RunningProcess()

    def rewrite(_: float) -> None:
        assert output_path is not None
        output_path.write_text("second\n")

    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    monkeypatch.setattr("nums.wake.subprocess.Popen", fake_popen)
    monkeypatch.setattr("nums.wake.time.sleep", rewrite)
    transcripts = WhisperStream(str(model)).transcripts()

    assert next(transcripts) == "first longer phrase"
    assert next(transcripts) == "second"
    transcripts.close()


def test_configured_sleep_phrase_with_apostrophe_matches() -> None:
    detector = WakePhraseDetector(sleep_phrases=("let's sleep",))
    detector.cooldown_seconds = 0
    detector.feed("Hey numnum")

    event = detector.feed("Let's sleep")

    assert event is not None
    assert event.kind == "sleep"


def test_configured_wake_phrase_with_apostrophe_matches() -> None:
    event = WakePhraseDetector("hey what's up").feed("Hey what's up, open Safari")

    assert event is not None
    assert event.kind == "command"
    assert event.text == "open Safari"


def test_listener_lock_is_private(tmp_path: Path) -> None:
    lock_path = tmp_path / "state" / "listener.lock"

    with ListenerLock(lock_path):
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(lock_path.parent.stat().st_mode) == 0o700


def test_small_voice_download_is_rejected_and_never_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def short_download(command: list[str], **kwargs: object) -> None:
        Path(command[command.index("--output") + 1]).write_bytes(b"<html>error page</html>")

    monkeypatch.setattr("nums.wake.subprocess.run", short_download)
    model = tmp_path / "voice.bin"

    with pytest.raises(RuntimeError, match="unexpectedly small"):
        download_voice_model(str(model))

    assert not model.exists()
    assert model.with_suffix(".bin.part").exists()


def test_complete_voice_download_replaces_the_partial_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def full_download(command: list[str], **kwargs: object) -> None:
        with Path(command[command.index("--output") + 1]).open("wb") as handle:
            handle.truncate(50_000_000)

    monkeypatch.setattr("nums.wake.subprocess.run", full_download)
    model = tmp_path / "nested" / "voice.bin"

    assert download_voice_model(str(model)) == model
    assert model.stat().st_size == 50_000_000
    assert not model.with_suffix(".bin.part").exists()


class _FakeProcess:
    def __init__(self, running: bool, stubborn: bool = False) -> None:
        self.running = running
        self.stubborn = stubborn
        self.events: list[str] = []

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.events.append("terminate")

    def kill(self) -> None:
        self.events.append("kill")
        self.stubborn = False

    def wait(self, timeout: float | None = None) -> int:
        if self.stubborn and timeout is not None:
            import subprocess

            raise subprocess.TimeoutExpired("whisper-stream", timeout)
        self.events.append("wait")
        return 0


def test_stop_terminates_a_running_stream_gracefully() -> None:
    stream = WhisperStream("/unused")
    stream.process = _FakeProcess(running=True)  # type: ignore[assignment]

    stream.stop()

    assert stream.process.events == ["terminate", "wait"]  # type: ignore[union-attr]


def test_stop_kills_a_stream_that_ignores_terminate() -> None:
    stream = WhisperStream("/unused")
    stream.process = _FakeProcess(running=True, stubborn=True)  # type: ignore[assignment]

    stream.stop()

    assert stream.process.events == ["terminate", "kill", "wait"]  # type: ignore[union-attr]


def test_stop_leaves_an_already_finished_stream_alone() -> None:
    stream = WhisperStream("/unused")
    stream.process = _FakeProcess(running=False)  # type: ignore[assignment]

    stream.stop()
    WhisperStream("/unused").stop()

    assert stream.process.events == []  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "transcript",
    ["Hey numnum", "hey num num", "Hey, Num-Num.", "hey nom nom", "HEY NOON NOON", "hey no no", "Hey Newman!", "hey   numb   numb"],
)
def test_common_spellings_of_the_wake_phrase_start_a_session(transcript: str) -> None:
    event = WakePhraseDetector().feed(transcript)

    assert event is not None and event.kind == "wake"


@pytest.mark.parametrize(
    "transcript",
    ["hey", "hello numnum", "numnum hey", "they numnum", "heynumnum", "", "   ", "[00:00:00.000 --> 00:00:01.000]", ">>"],
)
def test_other_speech_and_blank_transcripts_are_ignored_while_asleep(transcript: str) -> None:
    assert WakePhraseDetector().feed(transcript) is None


def test_wake_phrase_with_comma_and_timestamp_keeps_only_the_command() -> None:
    event = WakePhraseDetector().feed(
        "[00:00:03.000 --> 00:00:05.000] >> Hey numnum, what's on my calendar today?"
    )

    assert event is not None
    assert (event.kind, event.text) == ("command", "what's on my calendar today?")


def test_bracketed_noise_is_not_a_command_during_a_session() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0
    detector.feed("Hey numnum")

    assert detector.feed("[BLANK_AUDIO]") is None
    assert detector.feed("[ Silence ]") is None
    event = detector.feed("open my calendar")
    assert event is not None and event.kind == "command"


def test_a_command_inside_the_cooldown_is_dropped_then_accepted() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 60
    detector.feed("Hey numnum")

    first = detector.feed("open Safari")
    second = detector.feed("open Notes")
    detector.last_command_at -= 120
    third = detector.feed("open Mail")

    assert first is not None and second is None
    assert third is not None and third.text == "open Mail"


def test_stream_requires_the_binary_and_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("nums.wake.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="whisper-stream is missing"):
        list(WhisperStream(str(tmp_path / "model.bin")).transcripts())

    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    with pytest.raises(RuntimeError, match="wake model is missing.*--setup-voice"):
        list(WhisperStream(str(tmp_path / "missing.bin")).transcripts())


def test_stream_ends_and_stops_the_process_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.bin"
    model.touch()
    process = _FakeProcess(running=True)
    monkeypatch.setattr("nums.wake.shutil.which", lambda _: "/usr/local/bin/whisper-stream")
    monkeypatch.setattr("nums.wake.subprocess.Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr("nums.wake.time.sleep", lambda _: None)

    assert list(WhisperStream(str(model)).transcripts(timeout_seconds=0.05)) == []

    assert process.events == ["terminate", "wait"]


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Hey numnum, open Bob's notes.txt", "open Bob's notes.txt"),
        ("hey numnum open John’s file", "open John’s file"),
        ("Hey num num, don't close Mail", "don't close Mail"),
    ],
)
def test_wake_command_keeps_the_speakers_apostrophes(transcript: str, expected: str) -> None:
    event = WakePhraseDetector().feed(transcript)

    assert event is not None and event.kind == "command"
    assert event.text == expected


def test_follow_up_commands_keep_apostrophes_too() -> None:
    detector = WakePhraseDetector()
    detector.cooldown_seconds = 0
    detector.feed("Hey numnum")

    event = detector.feed("what's the weather like in St. John's")

    assert event is not None
    assert event.text == "what's the weather like in St. John's"


def test_wake_phrase_with_its_own_apostrophe_still_leaves_the_command_intact() -> None:
    event = WakePhraseDetector("hey what's up").feed("Hey what's up, read Dan's report")

    assert event is not None and event.text == "read Dan's report"


def test_listener_queues_speech_heard_while_the_caller_is_busy() -> None:
    import threading

    from nums.wake import StreamEnded, TranscriptListener

    done = threading.Event()

    class Stream:
        def transcripts(self):
            yield "one"
            yield "two"
            done.set()

        def stop(self) -> None:
            pass

    listener = TranscriptListener(Stream())  # type: ignore[arg-type]
    listener.start()
    assert done.wait(5)
    listener._thread.join(5)

    assert listener.drain() == ["one", "two"]
    with pytest.raises(StreamEnded):
        listener.get(timeout=1)


def test_listener_get_times_out_without_speech() -> None:
    import threading

    from nums.wake import TranscriptListener

    release = threading.Event()

    class Stream:
        def transcripts(self):
            release.wait(5)
            return
            yield

        def stop(self) -> None:
            release.set()

    listener = TranscriptListener(Stream())  # type: ignore[arg-type]
    listener.start()

    assert listener.get(timeout=0.05) is None
    listener.close()


def test_listener_forwards_stream_failures_after_earlier_speech() -> None:
    from nums.wake import TranscriptListener

    class Stream:
        def transcripts(self):
            yield "hello"
            raise RuntimeError("whisper-stream stopped with exit code 1")

        def stop(self) -> None:
            pass

    listener = TranscriptListener(Stream())  # type: ignore[arg-type]
    listener.start()
    listener._thread.join(5)

    assert listener.drain() == ["hello"]
    with pytest.raises(RuntimeError, match="exit code 1"):
        listener.get()


def test_closing_the_listener_stops_the_stream_and_hides_the_shutdown_error() -> None:
    import threading

    from nums.wake import TranscriptListener

    stopped = threading.Event()

    class Stream:
        def transcripts(self):
            stopped.wait(5)
            raise RuntimeError("whisper-stream stopped with exit code -15")
            yield

        def stop(self) -> None:
            stopped.set()

    listener = TranscriptListener(Stream())  # type: ignore[arg-type]
    listener.start()
    listener.close()

    assert stopped.is_set()
    assert listener.drain() == []
