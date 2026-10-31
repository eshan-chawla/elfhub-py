import logging
from typing import Any

import pytest

from elfhub.events import (
    Event,
    EventBus,
    MessageAdded,
    RunFinished,
    ToolStarted,
    Veto,
    parse_event,
)
from elfhub.types import Message, Usage


async def test_observers_get_their_event_type_and_wildcards_get_everything():
    bus = EventBus()
    tool_events: list[str] = []
    all_events: list[str] = []
    bus.on("tool.started", lambda e: tool_events.append(e.type))
    bus.on("*", lambda e: all_events.append(e.type))

    await bus.emit(ToolStarted(call_id="c1", name="add", arguments={}))
    await bus.emit(MessageAdded(message=Message.user("hi")))

    assert tool_events == ["tool.started"]
    assert all_events == ["tool.started", "message.added"]


async def test_async_observers_are_awaited():
    bus = EventBus()
    seen: list[str] = []

    async def observe(event: Event) -> None:
        seen.append(event.type)

    bus.on("*", observe)
    await bus.emit(MessageAdded(message=Message.user("hi")))

    assert seen == ["message.added"]


async def test_a_failing_observer_is_logged_and_does_not_stop_the_others(
    caplog: pytest.LogCaptureFixture,
):
    bus = EventBus()
    seen: list[str] = []

    def broken(_event: Event) -> None:
        raise RuntimeError("observer bug")

    bus.on("*", broken)
    bus.on("*", lambda e: seen.append(e.type))

    with caplog.at_level(logging.ERROR, logger="elfhub.events"):
        await bus.emit(MessageAdded(message=Message.user("hi")))

    assert seen == ["message.added"]
    assert "observer bug" in caplog.text


async def test_unsubscribing_stops_delivery():
    bus = EventBus()
    seen: list[str] = []
    unsubscribe = bus.on("*", lambda e: seen.append(e.type))

    unsubscribe()
    await bus.emit(MessageAdded(message=Message.user("hi")))

    assert seen == []


async def test_interceptors_run_highest_priority_first_and_chain_results():
    bus = EventBus()
    bus.intercept("before_x", lambda value: [*value, "low"], priority=-1)
    bus.intercept("before_x", lambda value: [*value, "high"], priority=10)
    bus.intercept("before_x", lambda value: None)  # None keeps the payload as is

    result = await bus.run_hook("before_x", ["start"])

    assert result == ["start", "high", "low"]


async def test_a_veto_stops_later_interceptors():
    bus = EventBus()
    later: list[Any] = []

    async def block(_value: Any) -> Veto:
        return Veto("not allowed")

    bus.intercept("before_x", block, priority=5)
    bus.intercept("before_x", lambda value: later.append(value))

    result = await bus.run_hook("before_x", "payload")

    assert result == Veto("not allowed")
    assert later == []


async def test_a_hook_without_interceptors_returns_the_payload():
    assert await EventBus().run_hook("nothing", 42) == 42


def test_parse_event_restores_the_concrete_type():
    event = RunFinished(session_id="s", run_id="r", status="completed", output="hi", usage=Usage())

    restored = parse_event(event.model_dump_json())

    assert isinstance(restored, RunFinished)
    assert restored == event


def test_unknown_plugin_events_round_trip_with_their_fields():
    raw = '{"v": 1, "type": "acme.custom", "seq": 3, "ts": 1.0, "session_id": "s", "payload": [1]}'

    restored = parse_event(raw)

    assert type(restored) is Event
    assert restored.type == "acme.custom"
    assert restored.model_dump()["payload"] == [1]


def test_a_second_class_for_an_existing_event_type_is_warned_about():
    with pytest.warns(UserWarning, match="run.finished"):

        class Impostor(Event):  # pyright: ignore[reportUnusedClass]
            type: str = "run.finished"

    # The original registration stays in force for reading logs.
    restored = parse_event({"type": "run.finished", "status": "completed"})
    assert isinstance(restored, RunFinished)
