# Changelog

## Unreleased

- Added general Mac app controls through Accessibility: inspect labeled UI elements, click exact matches, type text, and press keys or shortcuts.
- Added `nums --voice` for wake mode, with automatic Whisper model setup and an Ollama model check; plain `nums` remains text mode.
- NUMS starts a local Ollama server when needed for either mode.
- Added recursive project file discovery and numbered code reads.
- Kept both ends of long command output and partial output from timed-out commands.
- Made `--doctor` probe a real model tool call before reporting the model operational.

## 0.2.0 - 2026-09-22

- Added configurable execution modes and repeated-tool loop detection.
- Added optional persistent history and reviewed tool traces.
- Added clipboard, reminder, and calendar tools.
- Added configurable voice session timeouts and sleep phrases.
- Added macOS LaunchAgent installation and removal.
- Added deterministic training data preparation and trace review workflows.
- Improved atomic writes, diagnostics, Ollama failures, and voice reliability.

## 0.1.0

- Initial local Qwen assistant, Mac tools, CLI, and wake phrase support.
