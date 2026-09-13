from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


VOICE_MODEL_URL = (
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
)


@dataclass(frozen=True)
class WakeEvent:
    kind: str
    text: str = ""


class WakePhraseDetector:
    def __init__(self, phrase: str = "hey numnum") -> None:
        self.phrase = phrase.casefold().strip()
        self.awaiting_command = False
        self.last_transcript = ""
        variants = {self.phrase, "hey num num", "hey nom nom", "hey numb numb"}
        alternatives = "|".join(
            re.escape(item).replace(r"\ ", r"[\s,.-]+")
            for item in sorted(variants, key=len, reverse=True)
        )
        self.pattern = re.compile(rf"\b(?:{alternatives})\b[\s,.:;!?-]*(.*)", re.I)

    def feed(self, transcript: str) -> WakeEvent | None:
        cleaned = " ".join(transcript.strip().split())
        if not cleaned or cleaned == self.last_transcript:
            return None
        self.last_transcript = cleaned

        match = self.pattern.search(cleaned)
        if match:
            command = match.group(1).strip()
            if command:
                self.awaiting_command = False
                return WakeEvent("command", command)
            if not self.awaiting_command:
                self.awaiting_command = True
                return WakeEvent("wake")
            return None

        if self.awaiting_command and not cleaned.startswith("["):
            self.awaiting_command = False
            return WakeEvent("command", cleaned)
        return None


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
                        "2000",
                        "--length",
                        "6000",
                        "--keep",
                        "200",
                        "--max-tokens",
                        "32",
                        "--file",
                        str(output_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                position = 0
                try:
                    while self.process.poll() is None:
                        if output_path.exists():
                            with output_path.open(errors="replace") as transcript:
                                transcript.seek(position)
                                for line in transcript:
                                    if line.strip():
                                        yield line.strip()
                                position = transcript.tell()
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
