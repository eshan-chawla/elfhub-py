import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from elfhub import Elf
from elfhub.errors import ConfigError, PluginError
from elfhub.events import Event, RunFinished
from elfhub.llm import BaseLLM
from elfhub.plugins import llms, modules
from elfhub.plugins.module import ElfContext, Module
from elfhub.policy import Policy, Rule, SuspendAll
from elfhub.testing import ScriptedLLM, call
from elfhub.tools import ToolContext, ToolResult, ToolSpec, tool
from elfhub.types import TextPart


@tool
def add(a: int, b: int) -> int:
    return a + b


def test_run_returns_the_elfs_answer():
    elf = Elf(llm=ScriptedLLM("Hello!"))

    result = elf.run("hi")

    assert (result.status, result.output) == ("completed", "Hello!")


def test_calling_an_elf_runs_it():
    assert Elf(llm=ScriptedLLM("Hello!"))("hi").output == "Hello!"


def test_overriding_invoke_changes_what_calling_does():
    class Polite(Elf):
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            return "please"

    assert Polite(llm=ScriptedLLM())("hi") == "please"


async def test_arun_in_async_code():
    result = await Elf(llm=ScriptedLLM("async hello")).arun("hi")

    assert result.output == "async hello"


async def test_sync_run_still_works_inside_a_running_event_loop():
    result = Elf(llm=ScriptedLLM("from a thread")).run("hi")

    assert result.output == "from a thread"


def test_subclasses_can_declare_an_elf_with_class_attributes():
    class Ada(Elf):
        name = "ada"
        soul = "You are Ada, a careful engineer."
        instructions = "Always show your working."
        tools = (add,)

    llm = ScriptedLLM(call("add", a=1, b=2), "3")
    result = Ada(llm=llm).run("1+2?")

    assert result.output == "3"
    assert llm.requests[0].system == [
        TextPart(text="You are Ada, a careful engineer."),
        TextPart(text="Always show your working."),
    ]
    assert [t.name for t in llm.requests[0].tools] == ["add"]


def test_plain_functions_become_tools():
    def shout(text: str) -> str:
        return text.upper()

    llm = ScriptedLLM(call("shout", text="hey"), "done")

    Elf(llm=llm, tools=[shout]).run("hi")

    assert llm.requests[1].messages[-1].tool_results[0].content == [TextPart(text="HEY")]


def test_two_tools_with_the_same_name_are_a_config_error():
    def add(a: int) -> int:  # same name as the module-level add tool
        return a

    with pytest.raises(ConfigError, match="add"):
        Elf(llm=ScriptedLLM("x"), tools=[add, add]).run("hi")


def test_an_elf_without_a_model_is_a_config_error():
    with pytest.raises(ConfigError, match="llm"):
        Elf().run("hi")


@pytest.fixture
def echo_provider() -> Iterator[None]:
    class EchoLLM(ScriptedLLM):
        def __init__(self, model: str) -> None:
            super().__init__(lambda req: f"{model} heard {req.messages[-1].text}")

    llms.register(EchoLLM, name="echo-test")
    yield
    llms.unregister("echo-test")


@pytest.mark.usefixtures("echo_provider")
def test_model_strings_are_resolved_through_the_llm_registry():
    result = Elf(llm="echo-test/tiny").run("hi")

    assert result.output == "tiny heard hi"


class Recorder(Module):
    name: ClassVar[str] = "recorder"
    log: ClassVar[list[str]] = []

    def setup(self, ctx: ElfContext) -> None:
        self.log.append(f"setup:{self.name}")

    async def start(self) -> None:
        self.log.append(f"start:{self.name}")

    async def stop(self) -> None:
        self.log.append(f"stop:{self.name}")


class Base(Recorder):
    name: ClassVar[str] = "base"


class Dependent(Recorder):
    name: ClassVar[str] = "dependent"
    requires: ClassVar[tuple[str, ...]] = ("base",)


@pytest.fixture(autouse=True)
def clear_recorder_log() -> None:
    Recorder.log.clear()


def test_modules_start_after_their_dependencies_and_stop_in_reverse():
    Elf(llm=ScriptedLLM("ok"), modules=[Dependent(), Base()]).run("hi")

    assert Recorder.log == [
        "setup:base",
        "setup:dependent",
        "start:base",
        "start:dependent",
        "stop:dependent",
        "stop:base",
    ]


def test_a_missing_dependency_is_a_config_error():
    with pytest.raises(ConfigError, match="base"):
        Elf(llm=ScriptedLLM("ok"), modules=[Dependent()]).run("hi")


def test_modules_built_for_another_plugin_api_are_rejected():
    class Future(Module):
        name: ClassVar[str] = "future"
        api_version: ClassVar[int] = 99

    with pytest.raises(PluginError, match="future"):
        Elf(llm=ScriptedLLM("ok"), modules=[Future()]).run("hi")


def test_modules_stop_even_when_the_run_fails():
    Elf(llm=ScriptedLLM(RuntimeError("model crashed")), modules=[Base()]).run("hi")

    assert Recorder.log[-1] == "stop:base"


class Greeter(Module):
    """Adds a tool, a piece of system prompt and an observer."""

    name: ClassVar[str] = "greeter"

    class Config(BaseModel):
        greeting: str = "Hello"

    seen: ClassVar[list[str]] = []

    def setup(self, ctx: ElfContext) -> None:
        greeting = self.config.greeting  # pyright: ignore[reportAttributeAccessIssue]

        def greet(name: str) -> str:
            return f"{greeting}, {name}!"

        ctx.add_tool(greet)
        ctx.add_context(lambda _ctx: f"Greet people with '{greeting}'.")
        ctx.on("run.finished", lambda e: self.seen.append(e.type))


def test_modules_add_tools_context_and_observers():
    Greeter.seen.clear()
    llm = ScriptedLLM(call("greet", name="Ada"), "done")

    Elf(llm=llm, modules=[Greeter(greeting="Ahoy")]).run("hi")

    assert llm.requests[0].system == [TextPart(text="Greet people with 'Ahoy'.")]
    assert llm.requests[1].messages[-1].tool_results[0].content == [TextPart(text="Ahoy, Ada!")]
    assert Greeter.seen == ["run.finished"]


def test_module_observers_are_not_duplicated_when_the_elf_restarts():
    Greeter.seen.clear()
    elf = Elf(llm=ScriptedLLM("one", "two"), modules=[Greeter()])

    elf.run("first")
    elf.run("second")

    assert Greeter.seen == ["run.finished", "run.finished"]


def test_invalid_module_config_is_a_config_error_naming_the_module():
    with pytest.raises(ConfigError, match="greeter"):
        Greeter(greeting=["not", "a", "string"])


@pytest.fixture
def greeter_plugin() -> Iterator[None]:
    modules.register(Greeter)
    yield
    modules.unregister("greeter")


@pytest.mark.usefixtures("greeter_plugin")
def test_modules_can_be_enabled_by_name_with_config():
    llm = ScriptedLLM("ok")

    Elf(llm=llm, modules=["greeter"], module_config={"greeter": {"greeting": "Howdy"}}).run("hi")

    assert llm.requests[0].system == [TextPart(text="Greet people with 'Howdy'.")]


async def test_async_with_keeps_modules_running_across_runs():
    elf = Elf(llm=ScriptedLLM("one", "two"), modules=[Base()])

    async with elf:
        await elf.arun("first")
        await elf.arun("second")

    assert Recorder.log == ["setup:base", "start:base", "stop:base"]


async def test_astream_yields_this_runs_events_ending_with_run_finished():
    elf = Elf(llm=ScriptedLLM(call("add", a=1, b=1), "2"), tools=[add])

    events: list[Event] = [event async for event in elf.astream("1+1?")]

    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[0].type == "run.started"
    assert isinstance(events[-1], RunFinished) and events[-1].output == "2"
    assert "tool.completed" in [e.type for e in events]


async def test_aresume_continues_after_an_approval():
    elf = Elf(
        llm=ScriptedLLM(call("add", a=2, b=2, id="c1"), "4"),
        tools=[add],
        policy=Policy(rules=[Rule(tool="add", action="ask")]),
        broker=SuspendAll(),
    )

    suspended = await elf.arun("2+2?")
    resumed = await elf.aresume(suspended.session_id, {"c1": True})

    assert suspended.status == "suspended"
    assert (resumed.status, resumed.output) == ("completed", "4")


def test_an_elf_with_a_home_keeps_its_sessions_on_disk(tmp_path: Path):
    first = Elf(llm=ScriptedLLM("noted"), home=tmp_path).run("remember 42")

    llm = ScriptedLLM("you said 42")
    Elf(llm=llm, home=tmp_path).run("what did I say?", session=first.session_id)

    assert (tmp_path / ".elf" / "sessions" / f"{first.session_id}.jsonl").exists()
    assert [m.text for m in llm.requests[0].messages] == ["remember 42", "noted", "what did I say?"]


def test_elf_accepts_any_llm_object():
    class Quiet(BaseLLM):
        ref = "quiet/1"

        async def stream(self, request: Any):  # type: ignore[override]
            from elfhub.llm import PartDone, Stop

            yield PartDone(part=TextPart(text="..."))
            yield Stop(reason="end")

    assert Elf(llm=Quiet()).run("hi").output == "..."


class Napper(Module):
    name: ClassVar[str] = "napper"
    log: ClassVar[list[str]] = []

    def setup(self, ctx: ElfContext) -> None:
        self.log.append("setup")
        ctx.on("run.finished", lambda _e: self.log.append("finished"))

    async def start(self) -> None:
        await asyncio.sleep(0.01)
        self.log.append("start")

    async def stop(self) -> None:
        self.log.append("stop")


@tool
async def nap() -> str:
    await asyncio.sleep(0.01)
    return "rested"


async def test_concurrent_runs_start_modules_once_and_stop_after_the_last_run():
    Napper.log.clear()
    elf = Elf(
        llm=ScriptedLLM(call("nap"), call("nap"), "done", "done"), tools=[nap], modules=[Napper()]
    )

    results = await asyncio.gather(elf.arun("a"), elf.arun("b"))

    assert [r.status for r in results] == ["completed", "completed"]
    assert Napper.log == ["setup", "start", "finished", "finished", "stop"]


async def test_an_explicitly_started_elf_is_not_stopped_by_its_runs():
    Napper.log.clear()
    elf = Elf(llm=ScriptedLLM("one"), modules=[Napper()])

    await elf.start()
    await elf.arun("a")
    assert Napper.log == ["setup", "start", "finished"]

    await elf.stop()
    assert Napper.log[-1] == "stop"


class SearchTool:
    """A class-based tool, the natural shape for third-party tools."""

    spec = ToolSpec(name="search", input_schema={"type": "object", "properties": {}})

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.text("found it")


def test_passing_a_tool_class_instead_of_an_instance_is_explained():
    with pytest.raises(TypeError, match="instance"):
        Elf(llm=ScriptedLLM("x"), tools=[SearchTool]).run("hi")  # pyright: ignore[reportArgumentType]


def test_tool_class_instances_work():
    llm = ScriptedLLM(call("search"), "ok")

    Elf(llm=llm, tools=[SearchTool()]).run("hi")

    assert llm.requests[1].messages[-1].tool_results[0].content == [TextPart(text="found it")]


def test_a_module_plugin_for_another_api_is_refused_before_it_is_built():
    built: list[str] = []

    class FutureModule(Module):
        name: ClassVar[str] = "future-plugin"
        api_version: ClassVar[int] = 99

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            built.append("built")
            raise RuntimeError("must not be constructed")

    modules.register(FutureModule)
    try:
        with pytest.raises(PluginError, match="v99"):
            Elf(llm=ScriptedLLM("x"), modules=["future-plugin"]).run("hi")
    finally:
        modules.unregister("future-plugin")
    assert built == []


class ZoneConfig(BaseModel):
    zone: str = "UTC"


class Zone(Module[ZoneConfig]):
    """Typed config: the Config class comes from the type argument."""

    name: ClassVar[str] = "zone"

    def setup(self, ctx: ElfContext) -> None:
        zone: str = self.config.zone  # typed as ZoneConfig, no cast needed
        ctx.add_context(lambda _ctx: f"zone={zone}")


def test_module_config_can_be_declared_with_a_type_argument():
    llm = ScriptedLLM("ok")

    Elf(llm=llm, modules=[Zone(zone="Asia/Tokyo")]).run("hi")

    assert llm.requests[0].system == [TextPart(text="zone=Asia/Tokyo")]
    with pytest.raises(ConfigError, match="zone"):
        Zone(zone=123)
