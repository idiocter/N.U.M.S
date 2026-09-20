from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


VOICE_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
)


@dataclass(frozen=True)
class WakeEvent:
    kind: str
    text: str = ""


class ListenerLock:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path.home() / ".cache" / "nums" / "listener.lock"
        self.handle = None

    def __enter__(self) -> "ListenerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("NUMS is already listening in another process") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.handle:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class WakePhraseDetector:
    def __init__(self, phrase: str = "hey numnum") -> None:
        self.phrase = phrase.casefold().strip()
        self.active_session = False
        self.last_transcript = ""
        self.last_command = ""
        self.last_command_at = 0.0
        self.cooldown_seconds = 0.5
        variants = {
            self.phrase,
            "hey num num",
            "hey nom nom",
            "hey numb numb",
            "hey nam nam",
            "hey noom noom",
            "hey noon noon",
            "hey no no",
            "hey newman",
        }
        alternatives = "|".join(
            re.escape(item).replace(r"\ ", r"[\s,.-]+")
            for item in sorted(variants, key=len, reverse=True)
        )
        self.wake_pattern = re.compile(
            rf"\b(?:{alternatives})\b[\s,.:;!?-]*(.*)", re.I
        )
        sleep_variants = {
            "aight baby girl lets sleep",
            "eight baby girl lets sleep",
            "8 baby girl lets sleep",
            "ight baby girl lets sleep",
            "alright baby girl lets sleep",
            "all right baby girl lets sleep",
            "okay baby girl lets sleep",
            "aight baby girl go to sleep",
        }
        sleep_alternatives = "|".join(
            re.escape(item).replace(r"\ ", r"[\s,.-]+")
            for item in sorted(sleep_variants, key=len, reverse=True)
        )
        self.sleep_pattern = re.compile(rf"\b(?:{sleep_alternatives})\b", re.I)

    def feed(self, transcript: str) -> WakeEvent | None:
        cleaned = " ".join(transcript.strip().split())
        cleaned = re.sub(
            r"^\[[0-9:.]+\s*-->\s*[0-9:.]+\]\s*", "", cleaned
        )
        cleaned = re.sub(r"^>>\s*", "", cleaned)
        if not cleaned or cleaned == self.last_transcript:
            return None
        self.last_transcript = cleaned
        normalized = cleaned.replace("'", "").replace("’", "")

        if self.active_session and self.sleep_pattern.search(normalized):
            self.active_session = False
            self.last_command_at = 0.0
            return WakeEvent("sleep")

        match = self.wake_pattern.search(normalized)
        if match:
            self.active_session = True
            command = match.group(1).strip()
            if command:
                return self._command_event(command)
            return WakeEvent("wake")

        if self.active_session and not normalized.startswith("["):
            return self._command_event(normalized)
        return None

    def _command_event(self, command: str) -> WakeEvent | None:
        signature = re.sub(r"[^a-z0-9]+", " ", command.casefold()).strip()
        now = time.monotonic()
        if now - self.last_command_at < self.cooldown_seconds:
            return None
        self.last_command = signature
        self.last_command_at = now
        return WakeEvent("command", command)

    def command_completed(self) -> None:
        # The microphone stream is restarted after each response. A new stream
        # can contain the same spoken command, so clear stream-local duplicates.
        self.last_transcript = ""
        self.last_command_at = 0.0


def voice_dependencies(model_path: str) -> tuple[bool, bool]:
    return shutil.which("whisper-stream") is not None, Path(model_path).is_file()


def download_voice_model(model_path: str) -> Path:
    destination = Path(model_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            VOICE_MODEL_URL,
        ],
        check=True,
    )
    if partial.stat().st_size < 50_000_000:
        raise RuntimeError("Downloaded Whisper model is unexpectedly small")
    partial.replace(destination)
    return destination


class WhisperStream:
    def __init__(self, model_path: str, capture_device: int = -1) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.capture_device = capture_device
        self.process: subprocess.Popen[str] | None = None

    def transcripts(self) -> Iterator[str]:
        if not shutil.which("whisper-stream"):
            raise RuntimeError("whisper-stream is missing; run `brew install whisper-cpp`")
        if not Path(self.model_path).is_file():
            raise RuntimeError("the wake model is missing; run `uv run nums --setup-voice`")

        with tempfile.TemporaryDirectory(prefix="nums-wake-") as directory:
            output_path = Path(directory) / "transcript.txt"
            log_path = Path(directory) / "whisper.log"
            with log_path.open("w+") as log:
                self.process = subprocess.Popen(
                    [
                        "whisper-stream",
                        "--model",
                        self.model_path,
                        "--language",
                        "en",
                        "--capture",
                        str(self.capture_device),
                        "--step",
                        "0",
                        "--length",
                        "12000",
                        "--vad-thold",
                        "0.60",
                        "--max-tokens",
                        "32",
                        "--beam-size",
                        "5",
                        "--file",
                        str(output_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                position = 0
                try:
                    while True:
                        running = self.process.poll() is None
                        if output_path.exists():
                            with output_path.open(errors="replace") as transcript:
                                transcript.seek(position)
                                while line := transcript.readline():
                                    if running and not line.endswith("\n"):
                                        break
                                    position = transcript.tell()
                                    if line.strip():
                                        yield line.strip()
                        if not running:
                            break
                        time.sleep(0.2)
                finally:
                    self.stop()

                log.seek(0)
                tail = log.read()[-3000:].strip()
                raise RuntimeError(
                    f"whisper-stream stopped with exit code {self.process.returncode}:\n{tail}"
                )

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
