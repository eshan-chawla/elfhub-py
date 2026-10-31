import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import elfhub
from elfhub.cli import main
from elfhub.elfdir import scaffold
from elfhub.plugins import llms
from elfhub.testing import ScriptedLLM, call


@pytest.fixture
def cli_provider() -> Iterator[list[ScriptedLLM]]:
    made: list[ScriptedLLM] = []
    scripts: dict[str, list[object]] = {
        "plain": ["Hello from the CLI"],
        "tool": [call("word_count", text="one two three"), "three words"],
    }

    def factory(model: str) -> ScriptedLLM:
        llm = ScriptedLLM(*scripts[model])  # pyright: ignore[reportArgumentType]
        made.append(llm)
        return llm

    llms.register(factory, name="cli-test")
    yield made
    llms.unregister("cli-test")


def make_elf(tmp_path: Path, model: str | None, extra: str = "") -> Path:
    root = scaffold(tmp_path / "elf", name="ada")
    config = root / "elf.toml"
    text = config.read_text()
    if model:
        text = text.replace('# llm = "provider/model"', f'llm = "cli-test/{model}"')
    config.write_text(text + extra)
    return root


def test_version(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert elfhub.__version__ in capsys.readouterr().out


def test_new_creates_an_elf_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = main(["new", str(tmp_path / "bob"), "--name", "bob"])

    assert code == 0
    assert (tmp_path / "bob" / "elf.toml").exists()
    assert "bob" in capsys.readouterr().out


def test_new_refuses_a_non_empty_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / "keep.txt").write_text("x")

    code = main(["new", str(tmp_path)])

    assert code == 1
    assert "not empty" in capsys.readouterr().err


def test_new_names_the_elf_after_its_folder_by_default(tmp_path: Path):
    main(["new", str(tmp_path / "carol")])

    assert 'name = "carol"' in (tmp_path / "carol" / "elf.toml").read_text()


def test_schema_prints_the_elf_toml_json_schema(capsys: pytest.CaptureFixture[str]):
    code = main(["schema"])

    schema = json.loads(capsys.readouterr().out)
    assert code == 0
    assert "budget" in schema["properties"]


@pytest.mark.usefixtures("cli_provider")
def test_run_prints_the_answer(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = main(["run", str(make_elf(tmp_path, "plain")), "say", "hello"])

    assert code == 0
    assert capsys.readouterr().out.strip() == "Hello from the CLI"


def test_run_sends_the_joined_task(
    tmp_path: Path, cli_provider: list[ScriptedLLM], capsys: pytest.CaptureFixture[str]
):
    main(["run", str(make_elf(tmp_path, "plain")), "say", "hello"])

    assert cli_provider[0].requests[0].messages[-1].text == "say hello"


@pytest.mark.usefixtures("cli_provider")
def test_run_uses_the_elfs_folder_tools(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = main(["run", str(make_elf(tmp_path, "tool")), "count"])

    assert code == 0
    assert capsys.readouterr().out.strip() == "three words"


@pytest.mark.usefixtures("cli_provider")
def test_run_json_prints_the_run_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = main(["run", "--json", str(make_elf(tmp_path, "plain")), "hi"])

    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (result["status"], result["output"]) == ("completed", "Hello from the CLI")


def test_run_without_a_model_explains_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = main(["run", str(make_elf(tmp_path, None)), "hi"])

    assert code == 1
    assert "llm" in capsys.readouterr().err


ASK_EVERYTHING = '\n[[policy.rules]]\ntool = "*"\naction = "ask"\n'


def test_run_refuses_tools_needing_approval_when_nobody_can_answer(
    tmp_path: Path, cli_provider: list[ScriptedLLM], capsys: pytest.CaptureFixture[str]
):
    main(["run", str(make_elf(tmp_path, "tool", ASK_EVERYTHING)), "count"])

    [result] = cli_provider[0].requests[1].messages[-1].tool_results
    assert result.is_error


def test_run_yes_approves_tool_calls(
    tmp_path: Path, cli_provider: list[ScriptedLLM], capsys: pytest.CaptureFixture[str]
):
    code = main(["run", "--yes", str(make_elf(tmp_path, "tool", ASK_EVERYTHING)), "count"])

    [result] = cli_provider[0].requests[1].messages[-1].tool_results
    assert code == 0
    assert not result.is_error


@pytest.mark.usefixtures("cli_provider")
def test_run_with_an_invalid_session_id_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    code = main(["run", str(make_elf(tmp_path, "plain")), "hi", "--session", "../escape"])

    assert code == 1
    assert "session id" in capsys.readouterr().err


def test_new_on_an_existing_file_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "file.txt").write_text("x")

    code = main(["new", str(tmp_path / "file.txt")])

    assert code == 1
    assert "not a folder" in capsys.readouterr().err
