# Changelog

## Unreleased

- Added bounded waits for exact menu and submenu items to appear or become enabled.
- Added exact, verified option selection for inspected Mac popup controls.
- Added bounded waits for exact Mac windows and dialogs before later automation steps.
- Added bounded waits for exact Mac accessibility controls to appear or become enabled during asynchronous app workflows.
- Added window listing and precise window raising, explicit checkbox and switch state control, and more Mac navigation keys.
- Improved UI action checks, app opening, notifications, and reminder feedback.
- Improved voice model validation and download recovery; reduced accidental wake and sleep triggers; checked text model availability at startup.
- Added unlabeled-control targeting by inspected index, smaller configurable UI pages, and a read-only Mac app UI check in `--doctor`.
- Show UI control availability and toggle state, reject ambiguous or disabled menu actions, and verify text-field focus before typing.
- Paginated UI inspection; indexed control clicks; menu and submenu inspection and clicks; targeted text-field typing; actionable Mac permission errors.
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
