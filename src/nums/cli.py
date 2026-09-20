from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from .agent import Agent
from .config import Settings
from .ollama import OllamaClient, OllamaError
from .wake import (
    ListenerLock,
    WakePhraseDetector,
    WhisperStream,
    download_voice_model,
    voice_dependencies,
)


LOGO = r"""
 _   _ _   _ __  __ ____
| \ | | | | |  \/  / ___|
|  \| | | | | |\/| \___ \
| |\  | |_| | |  | |___) |
|_| \_|\___/|_|  |_|____/
"""


def doctor(settings: Settings) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Ollama: {shutil.which('ollama') or 'not found'}")
    print(f"Model: {settings.model}")
    client = OllamaClient(settings.ollama_url, settings.model)
    model_ready = client.has_model()
    print(f"Model ready: {'yes' if model_ready else 'no'}")
    whisper_ready, wake_model_ready = voice_dependencies(settings.whisper_model)
    print(f"Whisper stream: {'yes' if whisper_ready else 'no'}")
    print(f"Wake model: {'yes' if wake_model_ready else 'no'} ({settings.whisper_model})")
    return 0 if model_ready else 1


def wake_mode(agent: Agent, settings: Settings) -> None:
    try:
        with ListenerLock():
            detector = WakePhraseDetector(settings.wake_phrase)
            print(LOGO)
            print(f'Listening locally for "{settings.wake_phrase}". Press Ctrl+C to stop.\n')
            while True:
                stream = WhisperStream(settings.whisper_model, settings.capture_device)
                transcripts = stream.transcripts()
                event = None
                try:
                    for transcript in transcripts:
                        event = detector.feed(transcript)
                        if event is not None:
                            break
                finally:
                    transcripts.close()
                    stream.stop()

                if event is None:
                    continue
                if event.kind == "wake":
                    print("NUMS > Online. Keep talking until you send me to sleep.")
                    subprocess.run(
                        ["afplay", "/System/Library/Sounds/Glass.aiff"], check=False
                    )
                    continue
                if event.kind == "sleep":
                    print("NUMS > Going to sleep.\n")
                    subprocess.run(["say", "Good night."], check=False)
                    continue
                print(f"You > {event.text}")
                try:
                    response = agent.run(event.text)
                except OllamaError as exc:
                    print(f"NUMS error > {exc}\n")
                    subprocess.run(["say", "I could not reach the local model."], check=False)
                    detector.command_completed()
                    continue
                print(f"NUMS > {response}\n")
                subprocess.run(["say", response], check=False)
                detector.command_completed()
    except KeyboardInterrupt:
        print("\nNUMS wake listener stopped.")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="NUMS local macOS assistant")
    parser.add_argument("prompt", nargs="*", help="Run one request and exit")
    parser.add_argument("--doctor", action="store_true", help="Check Ollama and model availability")
    parser.add_argument("--pull", action="store_true", help="Download the configured local model")
    parser.add_argument("--speak", action="store_true", help="Read responses aloud")
    parser.add_argument("--wake", action="store_true", help='Listen for "hey numnum" locally')
    parser.add_argument(
        "--setup-voice", action="store_true", help="Download the local Whisper wake model"
    )
    args = parser.parse_args()
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.pull:
        raise SystemExit(subprocess.call(["ollama", "pull", settings.model]))
    if args.setup_voice:
        print(f"Downloading local wake model to {settings.whisper_model}")
        download_voice_model(settings.whisper_model)
        print("Wake model ready.")
        return
    if args.doctor:
        raise SystemExit(doctor(settings))

    try:
        agent = Agent(settings)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.wake:
        wake_mode(agent, settings)
        return
    speak = settings.speak or args.speak
    if args.prompt:
        try:
            response = agent.run(" ".join(args.prompt))
        except OllamaError as exc:
            raise SystemExit(str(exc)) from exc
        print(response)
        if speak:
            subprocess.run(["say", response], check=False)
        return

    print(LOGO)
    print(f"Local model: {settings.model} | Type /help for commands\n")
    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            break
        if prompt == "/help":
            print("/help — commands  /status — model and history  /reset — clear conversation  /quit — exit\n")
            continue
        if prompt == "/status":
            storage = settings.history_file or "memory only"
            print(f"Model: {settings.model} | History: {storage} | Messages: {len(agent.messages) - 1}\n")
            continue
        if prompt == "/reset":
            agent.reset()
            print("Conversation cleared.\n")
            continue
        try:
            response = agent.run(prompt)
        except OllamaError as exc:
            print(f"NUMS error > {exc}\n")
            continue
        print(f"NUMS > {response}\n")
        if speak:
            subprocess.run(["say", response], check=False)


if __name__ == "__main__":
    main()
