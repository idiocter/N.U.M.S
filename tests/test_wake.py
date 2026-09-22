import os
from pathlib import Path

import pytest

from nums.wake import ListenerLock, WakePhraseDetector, WhisperStream


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
    transcripts = stream.transcripts()

    assert next(transcripts) == "hey numnum, open Safari"
    transcripts.close()


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
