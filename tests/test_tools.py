from pathlib import Path

from nums.tools import MacTools


def test_read_and_list(tmp_path: Path) -> None:
    file = tmp_path / "hello.txt"
    file.write_text("hello NUMS")
    tools = MacTools()
    assert tools.read_file({"path": str(file)}) == "hello NUMS"
    assert "hello.txt" in tools.list_directory({"path": str(tmp_path)})
