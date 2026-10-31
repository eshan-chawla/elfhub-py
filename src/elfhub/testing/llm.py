"""A scripted, offline LLM for tests."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from elfhub.llm.protocol import (
    BaseLLM,
    ModelCapabilities,
    ModelEvent,
    ModelRequest,
    PartDone,
    Stop,
    TextDelta,
    UsageReport,
)
from elfhub.types import Message, Part, TextPart, ToolCallPart, Usage, new_id

Step = (
    str
    | ToolCallPart
    | Sequence[Part | str]
    | Message
    | BaseException
    | Callable[[ModelRequest], Any]
)
"""One scripted model turn: reply text, one tool call, a list of parts (plain strings in
it become text parts), a full message, an exception to raise, or a function of the
request returning any of these."""


def call(name: str, /, *, id: str | None = None, **arguments: Any) -> ToolCallPart:
    """A tool call step, e.g. ``call("add", a=1, b=2)``."""
    return ToolCallPart(id=id or new_id("call"), name=name, arguments=arguments)


class ScriptedLLM(BaseLLM):
    """Answers each request with the next scripted step and records every request."""

    def __init__(
        self,
        *steps: Step,
        ref: str = "scripted/test",
        capabilities: ModelCapabilities | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.ref = ref
        self.capabilities = capabilities or ModelCapabilities(parallel_tools=True)
        self.usage = usage or Usage()
        self.requests: list[ModelRequest] = []
        self._steps: deque[Step] = deque(steps)

    @property
    def remaining(self) -> int:
        return len(self._steps)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self.requests.append(request)
        if not self._steps:
            raise AssertionError(f"ScriptedLLM has no step left for request #{len(self.requests)}")
        step: Any = self._steps.popleft()
        if callable(step) and not isinstance(step, BaseException):
            step = step(request)
        if isinstance(step, BaseException):
            raise step
        for part in _parts(step):
            if isinstance(part, TextPart):
                yield TextDelta(text=part.text)
            yield PartDone(part=part)
        yield UsageReport(usage=self.usage)
        has_calls = any(isinstance(part, ToolCallPart) for part in _parts(step))
        yield Stop(reason="tool_use" if has_calls else "end")


def _parts(step: Any) -> list[Part]:
    if isinstance(step, str):
        return [TextPart(text=step)]
    if isinstance(step, Message):
        return list(step.parts)
    if isinstance(step, ToolCallPart):
        return [step]
    return [TextPart(text=item) if isinstance(item, str) else item for item in step]
