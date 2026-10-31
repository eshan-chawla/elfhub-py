from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from elfhub.errors import AuthFailed, ContextOverflow
from elfhub.llm import BaseLLM, ModelEvent, ModelRequest, PartDone, Stop, TextDelta, UsageReport
from elfhub.testing import (
    ScriptedLLM,
    call,
    check_auth_error,
    check_context_overflow,
    check_llm,
)
from elfhub.types import Part, ReasoningPart, TextPart, ToolCallPart, Usage


def well_behaved(request: ModelRequest) -> Any:
    """Answers the way a real, cooperative model would."""
    last = request.messages[-1]
    if last.role == "tool":
        return "It is sunny."
    if request.tools and request.tool_choice == "required":
        if "Paris and" in last.text:
            return [call("get_weather", city="Paris"), call("get_weather", city="Tokyo")]
        return call("get_weather", city="Tokyo" if "Tokyo" in last.text else "Paris")
    if request.reasoning is not None:
        return [ReasoningPart(text="thinking", provider="scripted", opaque={"sig": "s"}), "done"]
    return "pong"


def cooperative(**capabilities: Any) -> ScriptedLLM:
    llm = ScriptedLLM(*[well_behaved] * 20, usage=Usage(input_tokens=3, output_tokens=1))
    llm.capabilities = llm.capabilities.model_copy(update=capabilities)
    return llm


async def test_a_well_behaved_adapter_passes():
    await check_llm(cooperative())


async def test_a_well_behaved_reasoning_adapter_passes():
    llm = cooperative(reasoning=True)

    await check_llm(llm)

    # The reasoning check sends another provider's reasoning data, which adapters must drop.
    history = [part for r in llm.requests for m in r.messages for part in m.parts]
    assert any(isinstance(p, ReasoningPart) and p.provider != "scripted" for p in history)


async def test_an_adapter_that_reports_no_usage_fails():
    llm = ScriptedLLM(*[well_behaved] * 20)  # usage stays at zero

    with pytest.raises(AssertionError, match="usage"):
        await check_llm(llm)


async def test_usage_checks_can_be_skipped_for_providers_without_usage():
    await check_llm(ScriptedLLM(*[well_behaved] * 20), usage=False)


async def test_tool_call_ids_reused_across_turns_fail():
    def same_ids(request: ModelRequest) -> Any:
        if request.messages[-1].role != "tool" and request.tool_choice == "required":
            return call("get_weather", city="Paris", id="call_0")
        return well_behaved(request)

    llm = ScriptedLLM(*[same_ids] * 20, usage=Usage(input_tokens=1, output_tokens=1))

    with pytest.raises(AssertionError, match="unique"):
        await check_llm(llm)


async def test_claiming_parallel_calls_without_making_them_fails():
    def one_call(request: ModelRequest) -> Any:
        if request.messages[-1].role != "tool" and request.tool_choice == "required":
            return call("get_weather", city="Paris")
        return well_behaved(request)

    llm = ScriptedLLM(*[one_call] * 20, usage=Usage(input_tokens=1, output_tokens=1))
    llm.capabilities = llm.capabilities.model_copy(update={"parallel_tools": True})

    with pytest.raises(AssertionError, match="parallel"):
        await check_llm(llm)


async def test_an_auth_failure_must_map_to_auth_failed():
    def bad_key() -> CustomLLM:
        def events(_request: ModelRequest) -> list[Any]:
            raise AuthFailed("invalid api key")

        return CustomLLM(events)

    await check_auth_error(bad_key)


async def test_an_auth_failure_reported_as_something_else_fails():
    def bad_key() -> CustomLLM:
        def events(_request: ModelRequest) -> list[Any]:
            raise RuntimeError("HTTP 401")

        return CustomLLM(events)

    with pytest.raises(AssertionError, match="AuthFailed"):
        await check_auth_error(bad_key)


async def test_an_oversized_request_must_map_to_context_overflow():
    def events(request: ModelRequest) -> list[Any]:
        if len(request.messages[-1].text) > 100:
            raise ContextOverflow("too long")
        return reply([TextPart(text="ok")])

    llm = CustomLLM(events)
    llm.capabilities = llm.capabilities.model_copy(update={"context_window": 10})

    await check_context_overflow(llm)


async def test_an_oversized_request_that_does_not_fail_is_reported():
    llm = CustomLLM(lambda _req: reply([TextPart(text="ok")]))
    llm.capabilities = llm.capabilities.model_copy(update={"context_window": 10})

    with pytest.raises(AssertionError, match="ContextOverflow"):
        await check_context_overflow(llm)


class CustomLLM(BaseLLM):
    """An adapter whose stream is built by a function, to simulate adapter bugs."""

    ref = "custom/test"

    def __init__(self, events_for: Callable[[ModelRequest], list[Any]]) -> None:
        self.events_for = events_for

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        for event in self.events_for(request):
            yield event


def reply(parts: list[Part], stop: str = "end") -> list[Any]:
    return [*(PartDone(part=p) for p in parts), UsageReport(usage=Usage()), Stop(reason=stop)]  # pyright: ignore[reportArgumentType]


async def test_a_stream_without_a_final_stop_fails():
    llm = CustomLLM(lambda _req: [PartDone(part=TextPart(text="pong"))])

    with pytest.raises(AssertionError, match="Stop"):
        await check_llm(llm, tools=False)


async def test_events_after_stop_fail():
    llm = CustomLLM(lambda _req: [*reply([TextPart(text="pong")]), TextDelta(text="late")])

    with pytest.raises(AssertionError, match="last"):
        await check_llm(llm, tools=False)


async def test_text_deltas_that_disagree_with_the_final_text_fail():
    llm = CustomLLM(lambda _req: [TextDelta(text="pon"), *reply([TextPart(text="pong")])])

    with pytest.raises(AssertionError, match="delta"):
        await check_llm(llm, tools=False)


async def test_an_empty_reply_fails():
    llm = CustomLLM(lambda _req: reply([]))

    with pytest.raises(AssertionError, match="text"):
        await check_llm(llm, tools=False)


async def test_a_tool_call_without_the_required_arguments_fails():
    def events(request: ModelRequest) -> list[Any]:
        if request.tool_choice == "required":
            return reply([ToolCallPart(id="c1", name="get_weather", arguments={})], "tool_use")
        return reply([TextPart(text="pong")])

    with pytest.raises(AssertionError, match="city"):
        await check_llm(CustomLLM(events), usage=False)


async def test_an_adapter_that_cannot_take_tool_results_fails():
    def events(request: ModelRequest) -> list[Any]:
        if request.messages[-1].role == "tool":
            raise ValueError("adapter cannot translate tool results")
        if request.tool_choice == "required":
            return reply(
                [ToolCallPart(id="c1", name="get_weather", arguments={"city": "Paris"})], "tool_use"
            )
        return reply([TextPart(text="pong")])

    with pytest.raises(ValueError, match="tool results"):
        await check_llm(CustomLLM(events), usage=False)


async def test_tool_checks_are_skipped_for_models_without_tools():
    llm = ScriptedLLM(*["pong"] * 4, usage=Usage(input_tokens=1, output_tokens=1))
    llm.capabilities = llm.capabilities.model_copy(update={"tools": False})

    await check_llm(llm)

    assert all(not r.tools for r in llm.requests)
