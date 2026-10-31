from __future__ import annotations  # tool hints must resolve even when they are strings

from typing import Annotated, Any

import pytest
from pydantic import Field

from elfhub.tools import ToolContext, ToolResult, as_tool, tool
from elfhub.types import TextPart


async def _discard(_event: Any) -> None:
    return None


def make_ctx() -> ToolContext:
    return ToolContext(session_id="s1", run_id="r1", call_id="c1", emit=_discard)


def test_schema_comes_from_type_hints_and_docstring():
    @tool
    def add(a: int, b: int = 2) -> int:
        """Add two numbers."""
        return a + b

    assert add.spec.name == "add"
    assert add.spec.description == "Add two numbers."
    assert add.spec.input_schema == {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer", "default": 2},
        },
        "required": ["a"],
    }


def test_field_descriptions_reach_the_schema():
    @tool
    def greet(name: Annotated[str, Field(description="Who to greet")]) -> str:
        return f"hi {name}"

    assert greet.spec.input_schema["properties"]["name"] == {
        "type": "string",
        "description": "Who to greet",
    }


async def test_call_validates_and_coerces_arguments():
    @tool
    def add(a: int, b: int = 2) -> int:
        return a + b

    result = await add({"a": "3"}, make_ctx())

    assert result == ToolResult(content=[TextPart(text="5")])


async def test_invalid_arguments_return_an_error_without_calling_the_function():
    calls: list[int] = []

    @tool
    def record(n: int) -> str:
        calls.append(n)
        return "ok"

    result = await record({"n": "not a number"}, make_ctx())

    assert result.is_error
    assert "n" in result.content[0].text  # pyright: ignore[reportAttributeAccessIssue]
    assert calls == []


async def test_async_functions_are_awaited():
    @tool
    async def shout(text: str) -> str:
        return text.upper()

    result = await shout({"text": "hey"}, make_ctx())

    assert result.content == [TextPart(text="HEY")]


async def test_tool_context_is_injected_and_hidden_from_the_schema():
    seen: list[str] = []

    @tool
    def whoami(ctx: ToolContext, prefix: str = "") -> str:
        seen.append(ctx.call_id)
        return prefix + ctx.session_id

    result = await whoami({"prefix": ">"}, make_ctx())

    assert "ctx" not in whoami.spec.input_schema["properties"]
    assert result.content == [TextPart(text=">s1")]
    assert seen == ["c1"]


async def test_tool_result_return_values_pass_through():
    @tool
    def fail() -> ToolResult:
        return ToolResult.error("nope")

    result = await fail({}, make_ctx())

    assert result.is_error
    assert result.content == [TextPart(text="nope")]


async def test_structured_return_values_are_json_encoded():
    @tool
    def info() -> dict[str, Any]:
        return {"ok": True, "items": [1, 2]}

    result = await info({}, make_ctx())

    assert result.content == [TextPart(text='{"ok":true,"items":[1,2]}')]


def test_read_only_tools_are_not_destructive():
    @tool(read_only=True, parallel_safe=True)
    def look() -> str:
        return ""

    assert look.spec.annotations.read_only
    assert not look.spec.annotations.destructive
    assert look.spec.annotations.parallel_safe


def test_unannotated_tools_get_pessimistic_defaults():
    @tool
    def act() -> str:
        return ""

    annotations = act.spec.annotations
    assert (annotations.read_only, annotations.destructive, annotations.open_world) == (
        False,
        True,
        True,
    )
    assert not annotations.parallel_safe


def test_explicit_name_and_description_win():
    @tool(name="sum_numbers", description="Sums.")
    def add(a: int) -> int:
        """Ignored docstring."""
        return a

    assert (add.spec.name, add.spec.description) == ("sum_numbers", "Sums.")


def test_variadic_parameters_are_rejected():
    def spread(*values: int) -> int:
        return sum(values)

    with pytest.raises(TypeError, match="values"):
        tool(spread)


def test_as_tool_wraps_plain_functions_and_keeps_tools():
    def ping() -> str:
        return "pong"

    wrapped = as_tool(ping)

    assert wrapped.spec.name == "ping"
    assert as_tool(wrapped) is wrapped


def test_a_parameter_named_title_survives_title_stripping():
    @tool
    def note(title: str) -> str:
        return title

    assert note.spec.input_schema["properties"] == {"title": {"type": "string"}}


def test_tool_names_must_be_safe_for_every_provider():
    with pytest.raises(ValueError):
        tool(lambda: "x")  # "<lambda>" is not a valid tool name
