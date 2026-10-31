"""Events and the event bus.

Everything that happens during a run is an :class:`Event`. Events are persisted to
the session log (so a session can be replayed or resumed), streamed to callers, and
delivered to observers. Interceptors are the other half of the bus: ordered hooks
that may rewrite a payload or veto an action (this is how approvals plug in).

Plugins can define their own events by subclassing :class:`Event` with a new
``type``; unknown types still round-trip through the log as plain :class:`Event`.
"""

from __future__ import annotations

import inspect
import logging
import time
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from elfhub.tools.base import ToolResult
from elfhub.types import Message, Trust, Usage

logger = logging.getLogger("elfhub.events")

T = TypeVar("T")
Observer = Callable[["Event"], Awaitable[None] | None]
Interceptor = Callable[[Any], Any]


_EVENT_TYPES: dict[str, type[Event]] = {}


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")

    v: int = 1
    type: str = "event"
    seq: int = 0
    """Position in the session log, assigned by the store."""
    ts: float = Field(default_factory=time.time)
    session_id: str = ""
    run_id: str = ""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        event_type = cls.model_fields["type"].default
        if not isinstance(event_type, str):
            return
        existing = _EVENT_TYPES.get(event_type)
        redefined = existing is not None and (existing.__module__, existing.__qualname__) == (
            cls.__module__,
            cls.__qualname__,
        )
        if existing is None or redefined:
            _EVENT_TYPES[event_type] = cls
        else:
            warnings.warn(
                f"event type {event_type!r} is already defined by "
                f"{existing.__module__}.{existing.__qualname__}; ignoring {cls.__qualname__}",
                stacklevel=2,
            )


def parse_event(data: str | bytes | dict[str, Any]) -> Event:
    """Rebuild an event, choosing its class from the ``type`` field."""
    generic = (
        Event.model_validate_json(data)
        if isinstance(data, str | bytes)
        else Event.model_validate(data)
    )
    cls = _EVENT_TYPES.get(generic.type, Event)
    return generic if cls is Event else cls.model_validate(generic.model_dump())


class RunStarted(Event):
    type: str = "run.started"


class TurnStarted(Event):
    type: str = "turn.started"
    turn: int


class MessageAdded(Event):
    """A message joined the conversation. Replaying these rebuilds the history."""

    type: str = "message.added"
    message: Message


class ModelCompleted(Event):
    type: str = "model.completed"
    usage: Usage
    stop_reason: str


class ToolStarted(Event):
    type: str = "tool.started"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolCompleted(Event):
    """A tool call finished. The result is kept so a resumed run never repeats the call."""

    type: str = "tool.completed"
    call_id: str
    name: str
    is_error: bool = False
    result: ToolResult | None = None
    trust: Trust = "system"


class ApprovalRequested(Event):
    type: str = "approval.requested"
    call_id: str
    name: str
    arguments: dict[str, Any]


class ApprovalDecided(Event):
    type: str = "approval.decided"
    call_id: str
    approved: bool
    reason: str | None = None


class RunFinished(Event):
    type: str = "run.finished"
    status: str
    output: str | None = None
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)


@dataclass(frozen=True)
class Veto:
    """Returned by an interceptor to stop the action it intercepted."""

    reason: str


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class EventBus:
    """Observers watch events; interceptors can rewrite or veto hook payloads.

    Both run highest ``priority`` first, then in registration order. A failing
    observer is logged and skipped; a failing interceptor propagates, because a
    broken guard must not silently let an action through.
    """

    def __init__(self) -> None:
        self._observers: list[tuple[int, int, str, Observer]] = []
        self._interceptors: list[tuple[int, int, str, Interceptor]] = []
        self._counter = 0

    def on(self, event_type: str, fn: Observer, *, priority: int = 0) -> Callable[[], None]:
        """Observe events of ``event_type`` (``"*"`` for all). Returns an unsubscribe function."""
        return self._add(self._observers, event_type, fn, priority)

    def intercept(self, hook: str, fn: Interceptor, *, priority: int = 0) -> Callable[[], None]:
        """Intercept ``hook``. ``fn`` returns ``None`` (no change), a new payload, or a Veto."""
        return self._add(self._interceptors, hook, fn, priority)

    async def emit(self, event: Event) -> None:
        for _, _, event_type, fn in list(self._observers):
            if event_type not in ("*", event.type):
                continue
            try:
                await _maybe_await(fn(event))
            except Exception:
                logger.exception("observer %r failed on %s event", fn, event.type)

    async def run_hook(self, hook: str, payload: T) -> T | Veto:
        for _, _, name, fn in list(self._interceptors):
            if name != hook:
                continue
            result = await _maybe_await(fn(payload))
            if isinstance(result, Veto):
                return result
            if result is not None:
                payload = result
        return payload

    def _add(
        self, entries: list[tuple[int, int, str, Any]], key: str, fn: Any, priority: int
    ) -> Callable[[], None]:
        self._counter += 1
        entry = (-priority, self._counter, key, fn)
        entries.append(entry)
        entries.sort(key=lambda item: (item[0], item[1]))

        def remove() -> None:
            if entry in entries:
                entries.remove(entry)

        return remove
