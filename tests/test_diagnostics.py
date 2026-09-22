from nums.config import Settings
from nums.diagnostics import self_test


def test_self_test_checks_local_files_and_model_tool_call(monkeypatch) -> None:
    monkeypatch.setattr("nums.diagnostics.voice_dependencies", lambda _: (True, True))
    monkeypatch.setattr(
        "nums.diagnostics.OllamaClient.chat",
        lambda self, messages, tools: {
            "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "system_info", "arguments": {}}}
            ]}
        },
    )

    passed, results = self_test(Settings())

    assert passed
    assert results == ["File tools: ok", "Voice dependencies: ok", "Model tool call: ok"]
