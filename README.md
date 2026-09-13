# NUMS

NUMS is a private, local-first macOS assistant powered by Qwen through Ollama. It can inspect files, search the Mac, run commands, open apps and URLs, speak, show notifications, and automate apps with AppleScript.

The model and chat stay on the Mac through Ollama. NUMS asks before file writes, trash operations, AppleScript, and shell commands. It does not include persistence, background surveillance, credential access, or silent execution.

## Quick start

```bash
cd ~/Documents/Projects/NUMS
uv sync
uv run nums --pull
uv run nums
```

One-shot requests work too:

```bash
uv run nums "summarize the files on my Desktop"
uv run nums --speak "tell me the current system information"
```

## Wake phrase

NUMS can stay asleep until you say **“hey numnum.”** Voice recognition and the Qwen response both run locally.

Install the microphone runtime and download the small English wake model once:

```bash
brew install whisper-cpp
uv run nums --setup-voice
```

Start the listener:

```bash
uv run nums --wake
```

Say “hey numnum” and wait for “Yes?”, then speak the command. You can also say both together, such as “hey numnum, open Safari.” The first launch may trigger a macOS Microphone permission prompt for your terminal.

If the wrong microphone is selected, list the capture devices shown when the listener starts and set its number:

```bash
NUMS_CAPTURE_DEVICE=0 uv run nums --wake
```

The default model is `qwen2.5:1.5b-instruct`, a compact model close to the requested 2B size. Override it with `NUMS_MODEL`, for example:

```bash
NUMS_MODEL=qwen2.5:3b-instruct uv run nums --pull
NUMS_MODEL=qwen2.5:3b-instruct uv run nums
```

## Mac access

NUMS runs with the permissions of the Terminal app that launches it. macOS may ask for access when NUMS first touches protected locations or automates another app.

For broad access, open **System Settings → Privacy & Security** and grant your terminal only the permissions you actually want it to have, such as:

- Full Disk Access for protected files
- Automation for controlling specific apps with AppleScript
- Accessibility for UI automation scripts you choose to run

macOS permissions remain the outer security boundary. NUMS cannot and should not bypass them.

## Approval modes

The default mode asks before consequential actions. For a fully trusted local session, you can set `NUMS_AUTO_APPROVE=1`. This gives the model the same effective access as your terminal and should only be used while you are watching it:

```bash
NUMS_AUTO_APPROVE=1 uv run nums
```

## Diagnostics

```bash
uv run nums --doctor
uv run pytest
```

## Architecture

- `OllamaClient` talks only to the local Ollama HTTP API.
- `Agent` runs the tool-use loop and keeps conversation context in memory.
- `MacTools` provides files, shell, apps, speech, notifications, and AppleScript.
- `policy.classify` requires approval for consequential operations.
- `WhisperStream` listens locally and activates the agent only after the wake phrase.

## Current limits

- Wake listening requires the `whisper-cpp` Homebrew package and Microphone permission.
- A 1.5B model is fast and private but may need simple, explicit requests for long workflows.
- Conversation history is memory-only and disappears when NUMS exits.
