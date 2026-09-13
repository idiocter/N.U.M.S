from nums.policy import classify


def test_every_shell_command_needs_confirmation() -> None:
    assert classify("shell", {"command": "pwd && ls -la"}).needs_confirmation


def test_destructive_shell_needs_confirmation() -> None:
    assert classify("shell", {"command": "rm -rf build"}).needs_confirmation


def test_writes_need_confirmation() -> None:
    assert classify("write_file", {"path": "/tmp/demo"}).needs_confirmation
