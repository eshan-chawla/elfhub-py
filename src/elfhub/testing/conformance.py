"""Conformance checks for LLM adapters.

Any adapter, first-party or not, can prove it implements the ElfHub LLM contract::

    from elfhub.testing import check_llm

    async def test_my_adapter():
        await check_llm(MyAdapter(model="small-model"))

The checks talk to the real model, so they are written to pass with any reasonably
capable model. Each raises :class:`AssertionError` describing the first violation it
finds. :func:`check_llm` runs every check that applies to the adapter's declared
capabilities. Error mapping can't be provoked safely by default, so
:func:`check_auth_error` and :func:`check_context_overflow` are opt-in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from elfhub.errors import AuthFailed, ContextOverflow
from elfhub.llm.protocol import (
    LLM,
    ModelRequest,
    ModelResponse,
    PartDone,
    ReasoningConfig,
    Stop,
    TextDelta,
)
from elfhub.tools.base import ToolAnnotations, ToolSpec
from elfhub.types import Message, ReasoningPart, TextPart, ToolCallPart, ToolResultPart

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city name"}},
        "required": ["city"],
    },
    annotations=ToolAnnotations(read_only=True, destructive=False),
)


async def check_llm(llm: LLM, *, tools: bool = True, usage: bool = True) -> None:
    """Run every check that applies to the adapter's declared capabilities.

    Pass ``usage=False`` for a provider that reports no token usage (token and cost
    budgets then can't work), and ``tools=False`` to skip tool checks.
    """
    await check_stream(llm)
    await check_text_reply(llm)
    if usage:
        await check_usage_reported(llm)
    if tools and llm.capabilities.tools:
        await check_tool_round_trip(llm)
        if llm.capabilities.parallel_tools:
            await check_parallel_tool_calls(llm)
    if llm.capabilities.reasoning:
        await check_reasoning_round_trip(llm)


def _ping() -> ModelRequest:
    return ModelRequest(
        system=[TextPart(text="Reply with exactly one word: pong")],
        messages=[Message.user("ping")],
        max_output_tokens=64,
    )


def _expect(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def check_stream(llm: LLM) -> None:
    """A stream ends with exactly one Stop, and text deltas add up to the final text."""
    events = [event async for event in llm.stream(_ping())]
    stops = [event for event in events if isinstance(event, Stop)]
    _expect(len(stops) == 1, f"a stream must contain exactly one Stop event, got {len(stops)}")
    _expect(isinstance(events[-1], Stop), "Stop must be the last event of a stream")
    deltas = "".join(event.text for event in events if isinstance(event, TextDelta))
    final = "".join(
        event.part.text
        for event in events
        if isinstance(event, PartDone) and isinstance(event.part, TextPart)
    )
    if deltas:
        _expect(
            deltas == final,
            f"text deltas must add up to the finished text parts: {deltas!r} != {final!r}",
        )


async def check_text_reply(llm: LLM) -> None:
    """A plain question gets a text answer, with no tool calls and sane usage."""
    response = await llm.generate(_ping())
    _expect(response.message.role == "assistant", "generate() must return an assistant message")
    _expect(response.message.text.strip(), "a plain question must get a non-empty text reply")
    _expect(not response.message.tool_calls, "a request without tools must not produce tool calls")
    _expect(
        response.stop_reason in ("end", "max_tokens"),
        f"a plain reply must stop with 'end' or 'max_tokens', not {response.stop_reason!r}",
    )
    _expect(
        response.usage.input_tokens >= 0 and response.usage.output_tokens >= 0,
        "usage must not be negative",
    )


async def check_usage_reported(llm: LLM) -> None:
    """The adapter reports token usage, which token and cost budgets depend on."""
    response = await llm.generate(_ping())
    _expect(
        response.usage.input_tokens + response.usage.output_tokens > 0,
        "the adapter must report token usage (UsageReport events); token and cost budgets "
        "depend on it",
    )


def _weather_request(question: str) -> ModelRequest:
    return ModelRequest(
        system=[TextPart(text="Use the tools you are given.")],
        messages=[Message.user(question)],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        max_output_tokens=256,
    )


def _check_weather_calls(response: ModelResponse) -> list[ToolCallPart]:
    calls = response.message.tool_calls
    _expect(calls, "with tool_choice='required' the model must call a tool")
    _expect(
        response.stop_reason == "tool_use",
        f"a reply with tool calls must stop with 'tool_use', not {response.stop_reason!r}",
    )
    _expect(len({c.id for c in calls}) == len(calls), "tool call ids must be unique")
    for c in calls:
        _expect(c.id, "every tool call needs an id")
        _expect(c.name == WEATHER_TOOL.name, f"the model called an unknown tool {c.name!r}")
        _expect(
            isinstance(c.arguments.get("city"), str),
            "tool arguments must be parsed into a dict with the required 'city' string, "
            f"got {c.arguments!r}",
        )
    return calls


def _results_for(calls: Sequence[ToolCallPart]) -> Message:
    return Message(
        role="tool",
        parts=[ToolResultPart(call_id=c.id, content=[TextPart(text="Sunny, 21°C")]) for c in calls],
        trust="untrusted",
    )


async def check_tool_round_trip(llm: LLM) -> None:
    """The model calls a tool with parsed arguments and answers from its result; a
    second turn's call ids never repeat the first turn's."""
    request = _weather_request("What is the weather in Paris right now? Use the get_weather tool.")
    response = await llm.generate(request)
    calls = _check_weather_calls(response)

    history = [*request.messages, response.message, _results_for(calls)]
    final = await llm.generate(
        request.model_copy(update={"tool_choice": "auto", "messages": history})
    )
    _expect(
        final.message.text.strip(), "after receiving tool results the model must answer in text"
    )

    again = request.model_copy(
        update={
            "messages": [
                *history,
                final.message,
                Message.user("And in Tokyo? Use the get_weather tool again."),
            ]
        }
    )
    second = _check_weather_calls(await llm.generate(again))
    reused = {c.id for c in calls} & {c.id for c in second}
    _expect(
        not reused,
        f"tool call ids must be unique across turns, but {sorted(reused)} were reused; "
        "approvals and results are matched by id",
    )


async def check_parallel_tool_calls(llm: LLM) -> None:
    """An adapter declaring ``parallel_tools`` returns several calls in one reply."""
    request = _weather_request(
        "Get the weather for Paris and for Tokyo. Call get_weather once per city, "
        "both in this reply."
    )
    response = await llm.generate(request)
    calls = _check_weather_calls(response)
    _expect(
        len(calls) >= 2,
        f"the adapter declares parallel_tools but returned {len(calls)} call(s) for two cities",
    )
    history = [*request.messages, response.message, _results_for(calls)]
    final = await llm.generate(
        request.model_copy(update={"tool_choice": "auto", "messages": history})
    )
    _expect(final.message.text.strip(), "after parallel tool results the model must answer in text")


async def check_reasoning_round_trip(llm: LLM) -> None:
    """Reasoning parts can be sent back to the same provider, and another provider's
    reasoning data is dropped rather than causing an error."""
    request = ModelRequest(
        messages=[Message.user("What is 17 * 23? Answer with just the number.")],
        reasoning=ReasoningConfig(effort="low"),
        max_output_tokens=2048,
    )
    response = await llm.generate(request)
    for part in response.message.parts:
        if isinstance(part, ReasoningPart):
            _expect(part.provider, "reasoning parts must name the provider they came from")

    foreign = Message(
        role="assistant",
        parts=[
            ReasoningPart(text="notes", provider="elfhub-conformance", opaque={"signature": "x"}),
            TextPart(text="Noted."),
        ],
        trust="self",
    )
    follow_up = request.model_copy(
        update={
            "messages": [
                *request.messages,
                response.message,
                Message.user("Thanks."),
                foreign,
                Message.user("Now reply with the single word: done"),
            ]
        }
    )
    final = await llm.generate(follow_up)
    _expect(
        final.message.text.strip(), "a conversation with reasoning parts must still get a reply"
    )


async def check_auth_error(make_llm_with_bad_credentials: Callable[[], LLM]) -> None:
    """Rejected credentials surface as :class:`~elfhub.errors.AuthFailed`."""
    llm = make_llm_with_bad_credentials()
    try:
        await llm.generate(_ping())
    except AuthFailed:
        return
    except Exception as exc:
        raise AssertionError(
            f"rejected credentials must raise AuthFailed, not {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError("rejected credentials must raise AuthFailed, but the call succeeded")


async def check_context_overflow(llm: LLM) -> None:
    """A request larger than the declared context window raises
    :class:`~elfhub.errors.ContextOverflow`. Needs ``capabilities.context_window``."""
    window = llm.capabilities.context_window
    _expect(window, "set capabilities.context_window to run the context overflow check")
    assert window is not None
    huge = ModelRequest(messages=[Message.user("hello " * (window * 2))], max_output_tokens=16)
    try:
        await llm.generate(huge)
    except ContextOverflow:
        return
    except Exception as exc:
        raise AssertionError(
            f"an oversized request must raise ContextOverflow, not {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError("an oversized request must raise ContextOverflow, but the call succeeded")
