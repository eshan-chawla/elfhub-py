import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from elfhub import Budget, Elf, Policy, Rule
from elfhub.config import ElfConfig, config_schema, load_config
from elfhub.elfdir import scaffold
from elfhub.errors import ConfigError, ConfigWarning
from elfhub.plugins import llms, modules, tools
from elfhub.plugins.module import ElfContext, Module
from elfhub.testing import ScriptedLLM, call
from elfhub.tools import tool
from elfhub.types import TextPart


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def test_a_minimal_config_gets_defaults(tmp_path: Path):
    config = load_config(write(tmp_path / "elf.toml", 'name = "ada"\n'))

    assert config.name == "ada"
    assert config.format_version == 1
    assert config.llm.main is None
    assert config.modules == {}
    assert config.budget.max_turns == 50


def test_llm_can_be_a_plain_model_string(tmp_path: Path):
    config = load_config(write(tmp_path / "elf.toml", 'name = "ada"\nllm = "acme/large"\n'))

    assert config.llm.main == "acme/large"


def test_unknown_keys_warn_with_a_suggestion_and_are_ignored(tmp_path: Path):
    path = write(
        tmp_path / "elf.toml",
        """
        name = "ada"
        modles = {}

        [llm]
        main = "acme/x"
        reflect = "acme/small"
        """,
    )

    with pytest.warns(ConfigWarning) as record:
        config = load_config(path)

    messages = " | ".join(str(w.message) for w in record)
    assert "'modles'" in messages and "'modules'" in messages
    assert "'reflect'" in messages
    assert config.llm.main == "acme/x"
    assert "modles" not in config.model_dump()


@pytest.mark.parametrize(
    ("section", "typo", "suggestion"),
    [
        ("[budget]\nmax_turn = 3", "max_turn", "max_turns"),
        ("[budget]\nmax_cost = 1.0", "max_cost", "max_cost_usd"),
        ('[policy]\ndefualt = "deny"', "defualt", "default"),
        ('[[policy.rules]]\ntol = "safe_tool"\naction = "allow"', "tol", "tool"),
    ],
)
def test_unknown_keys_in_safety_settings_are_errors(
    tmp_path: Path, section: str, typo: str, suggestion: str
):
    # Ignoring these would silently loosen a limit or a rule.
    path = write(tmp_path / "elf.toml", f'name = "ada"\n{section}\n')

    with pytest.raises(ConfigError) as info:
        load_config(path)

    assert f"'{typo}'" in str(info.value) and f"'{suggestion}'" in str(info.value)


def test_extensions_are_kept_without_warnings(tmp_path: Path):
    path = write(
        tmp_path / "elf.toml",
        """
        name = "ada"

        [extensions."elfhub-cloud"]
        phone_number = "+15550100"
        """,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = load_config(path)

    assert config.extensions == {"elfhub-cloud": {"phone_number": "+15550100"}}


def test_python_settings_objects_reject_typos():
    with pytest.raises(ValidationError):
        Budget(max_turn=3)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValidationError):
        Rule(tol="x", action="deny")  # pyright: ignore[reportCallIssue]


def test_a_config_from_a_newer_elfhub_is_refused(tmp_path: Path):
    path = write(tmp_path / "elf.toml", 'format_version = 99\nname = "ada"\n')

    with pytest.raises(ConfigError, match="newer"):
        load_config(path)


def test_invalid_configs_name_the_file(tmp_path: Path):
    path = write(tmp_path / "elf.toml", "budget = 3\n")  # no name, wrong budget type

    with pytest.raises(ConfigError, match=r"elf\.toml"):
        load_config(path)


def test_broken_toml_names_the_file(tmp_path: Path):
    path = write(tmp_path / "elf.toml", "name = \n")

    with pytest.raises(ConfigError, match=r"elf\.toml"):
        load_config(path)


def test_the_json_schema_describes_the_config():
    schema = config_schema()

    assert "name" in schema["required"]
    assert {"llm", "modules", "policy", "budget"} <= set(schema["properties"])


class Signature(Module):
    name: ClassVar[str] = "signature"

    def setup(self, ctx: ElfContext) -> None:
        sign_off = self.config.model_dump().get("sign_off", "")
        ctx.add_context(lambda _ctx: f"Sign off as {sign_off}.")

    class Config(BaseModel):
        model_config = ConfigDict(extra="allow")


@tool
def utc_now() -> str:
    """The current time."""
    return "12:00"


@pytest.fixture
def plugins() -> Iterator[None]:
    modules.register(Signature)
    tools.register(utc_now)
    yield
    modules.unregister("signature")
    tools.unregister("utc_now")


def make_elf_dir(root: Path) -> Path:
    write(
        root / "elf.toml",
        """
        name = "ada"
        llm = "scripted-test/any"
        tools = ["utc_now"]

        [modules.signature]
        sign_off = "Ada"

        [[policy.rules]]
        tool = "delete_*"
        action = "deny"

        [budget]
        max_turns = 5
        """,
    )
    write(root / "SOUL.md", "You are Ada.\n")
    write(root / "AGENTS.md", "Keep answers short.\n")
    write(
        root / "tools" / "greet.py",
        """
        from elfhub.tools import tool

        @tool
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        """,
    )
    return root


@pytest.mark.usefixtures("plugins")
def test_from_dir_builds_an_elf_from_its_folder(tmp_path: Path):
    root = make_elf_dir(tmp_path / "ada")
    llm = ScriptedLLM(call("greet", name="Bob"), "done")

    elf = Elf.from_dir(root, llm=llm)
    result = elf.run("greet Bob")

    assert elf.name == "ada"
    assert elf.budget is not None and elf.budget.max_turns == 5
    assert llm.requests[0].system == [
        TextPart(text="You are Ada.\n"),
        TextPart(text="Keep answers short.\n"),
        TextPart(text="Sign off as Ada."),
    ]
    assert sorted(t.name for t in llm.requests[0].tools) == ["greet", "utc_now"]
    assert llm.requests[1].messages[-1].tool_results[0].content == [TextPart(text="Hello, Bob!")]
    assert (root / ".elf" / "sessions" / f"{result.session_id}.jsonl").exists()


@pytest.fixture
def scripted_provider() -> Iterator[list[dict[str, Any]]]:
    created: list[dict[str, Any]] = []

    def factory(model: str, **options: Any) -> ScriptedLLM:
        created.append({"model": model, **options})
        return ScriptedLLM("hi from config")

    llms.register(factory, name="scripted-test")
    yield created
    llms.unregister("scripted-test")


@pytest.mark.usefixtures("plugins")
def test_from_dir_resolves_the_declared_model_with_its_options(
    tmp_path: Path, scripted_provider: list[dict[str, Any]]
):
    root = make_elf_dir(tmp_path / "ada")
    toml = (root / "elf.toml").read_text().replace('llm = "scripted-test/any"\n', "")
    toml += '\n[llm]\nmain = "scripted-test/any"\noptions = { temperature = 0.2 }\n'
    (root / "elf.toml").write_text(toml)

    result = Elf.from_dir(root).run("hi")

    assert result.output == "hi from config"
    assert scripted_provider == [{"model": "any", "temperature": 0.2}]


def test_a_tool_file_that_fails_to_import_is_a_config_error(tmp_path: Path):
    write(tmp_path / "elf.toml", 'name = "ada"\n')
    write(tmp_path / "tools" / "broken.py", "import elfhub_no_such_module\n")

    with pytest.raises(ConfigError, match=r"broken\.py"):
        Elf.from_dir(tmp_path)


def test_scaffold_creates_an_elf_folder_that_loads(tmp_path: Path):
    root = scaffold(tmp_path / "my-elf", name="my-elf")

    elf = Elf.from_dir(root)

    assert elf.name == "my-elf"
    assert elf.soul and "my-elf" in elf.soul
    assert (root / "AGENTS.md").exists()
    assert ".elf/" in (root / ".gitignore").read_text()


def test_scaffold_refuses_to_overwrite_a_non_empty_folder(tmp_path: Path):
    (tmp_path / "existing.txt").write_text("keep me")

    with pytest.raises(FileExistsError):
        scaffold(tmp_path, name="ada")
    assert (tmp_path / "existing.txt").read_text() == "keep me"


def test_elf_config_round_trips_through_a_dict():
    config = ElfConfig.model_validate(
        {"name": "ada", "llm": "acme/x", "extensions": {"x": {"y": 1}}}
    )

    assert ElfConfig.model_validate(config.model_dump()) == config


def test_from_dir_keeps_what_a_subclass_declares(tmp_path: Path):
    write(tmp_path / "elf.toml", 'name = "careful"\n')
    write(
        tmp_path / "tools" / "wipe.py",
        """
        from elfhub.tools import tool

        @tool
        def wipe() -> str:
            return "wiped"
        """,
    )

    class Careful(Elf):
        policy = Policy(rules=[Rule(tool="wipe", action="deny")])
        budget = Budget(max_turns=7)
        tools = (utc_now,)

    llm = ScriptedLLM(call("wipe"), "ok")
    elf = Careful.from_dir(tmp_path, llm=llm)
    elf.run("wipe everything")

    assert llm.requests[1].messages[-1].tool_results[0].is_error  # the class policy still applies
    assert elf.budget is not None and elf.budget.max_turns == 7
    assert sorted(t.name for t in llm.requests[0].tools) == ["utc_now", "wipe"]


def test_a_folder_policy_adds_to_a_subclass_policy(tmp_path: Path):
    write(
        tmp_path / "elf.toml",
        """
        name = "careful"
        [budget]
        max_turns = 3

        [[policy.rules]]
        tool = "utc_now"
        action = "deny"
        """,
    )

    class Open(Elf):
        policy = Policy(default="allow")
        budget = Budget(max_turns=7)
        tools = (utc_now,)

    llm = ScriptedLLM(call("utc_now"), "ok")
    elf = Open.from_dir(tmp_path, llm=llm)
    elf.run("time?")

    assert llm.requests[1].messages[-1].tool_results[0].is_error
    assert elf.budget is not None and elf.budget.max_turns == 3  # set in the folder, so it wins


def test_tool_files_with_a_tool_class_and_its_instance_load_one_tool(tmp_path: Path):
    write(tmp_path / "elf.toml", 'name = "ada"\n')
    write(
        tmp_path / "tools" / "search.py",
        """
        from elfhub.tools import ToolResult, ToolSpec

        class Search:
            spec = ToolSpec(name="search", input_schema={"type": "object", "properties": {}})

            async def __call__(self, args, ctx):
                return ToolResult.text("found")

        search = Search()
        """,
    )
    llm = ScriptedLLM(call("search"), "ok")

    Elf.from_dir(tmp_path, llm=llm).run("find it")

    assert [t.name for t in llm.requests[0].tools] == ["search"]


def test_scaffold_handles_any_name(tmp_path: Path):
    name = 'Ada "the second" 😀'
    root = scaffold(tmp_path / "ada", name=name)

    assert Elf.from_dir(root).name == name


def test_scaffold_refuses_a_path_that_is_a_file(tmp_path: Path):
    (tmp_path / "file.txt").write_text("x")

    with pytest.raises(FileExistsError, match="not a folder"):
        scaffold(tmp_path / "file.txt", name="ada")
