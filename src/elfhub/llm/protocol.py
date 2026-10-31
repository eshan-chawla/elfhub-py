"""The LLM declaration: the contract every model adapter implements.

ElfHub ships no vendor code in its core. An adapter is any object satisfying
:class:`LLM`; most subclass :class:`BaseLLM` and implement only :meth:`~BaseLLM.stream`.

Streaming contract, which ``elfhub.testing`` conformance checks enforce:

* ``TextDelta`` / ``ReasoningDelta`` events are optional and only for live display;
* every finished part (text, tool call, reasoning) is emitted once as ``PartDone``;
* ``UsageReport`` events may arrive any number of times and are summed;
* the stream ends with exactly one ``Stop``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from elfhub.errors import ModelError
from elfhub.tools.base import ToolSpec
from elfhub.types import Message, Part, TextPart, Usage


class ModelCapabilities(BaseModel):
    """What a model can do. Frozen: adapters set a new instance instead of editing one."""

    model_config = ConfigDict(frozen=True)

    tools: bool = True
    parallel_tools: bool = False
    vision: bool = False
    reasoning: bool = False
    json_schema: bool = False
    native_computer_use: bool = False
    context_window: int | None = None


class ReasoningConfig(BaseModel):
    effort: Literal["low", "medium", "high"] | None = None
    budget_tokens: int | None = None


class ModelRequest(BaseModel):
    system: list[TextPart] = Field(default_factory=list[TextPart])
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list[ToolSpec])
    tool_choice: str = "auto"
    """``"auto"``, ``"none"``, ``"required"``, or the name of one tool."""
    max_output_tokens: int | None = None
    temperature: float | None = None
    response_schema: dict[str, Any] | None = None
    reasoning: ReasoningConfig | None = None
    cache_breakpoints: list[int] = Field(default_factory=list[int])
    """Message indexes after which a provider may cache the prompt prefix. Hints only."""
    provider_options: dict[str, dict[str, Any]] = Field(default_factory=dict[str, dict[str, Any]])
    """Escape hatch for vendor-specific settings, namespaced by provider name."""


StopReason = Literal["end", "tool_use", "max_tokens", "refusal"]


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ReasoningDelta(BaseModel):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str


class PartDone(BaseModel):
    type: Literal["part_done"] = "part_done"
    part: Part


class UsageReport(BaseModel):
    type: Literal["usage"] = "usage"
    usage: Usage


class Stop(BaseModel):
    type: Literal["stop"] = "stop"
    reason: StopReason


ModelEvent = Annotated[
    TextDelta | ReasoningDelta | PartDone | UsageReport | Stop,
    Field(discriminator="type"),
]


class ModelResponse(BaseModel):
    message: Message
    usage: Usage
    stop_reason: StopReason


@runtime_checkable
class LLM(Protocol):
    ref: str
    capabilities: ModelCapabilities

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    async def generate(self, request: ModelRequest) -> ModelResponse: ...

    async def count_tokens(self, request: ModelRequest) -> int | None: ...


class BaseLLM(ABC):
    """Convenience base: implement :meth:`stream`, get :meth:`generate` for free."""

    ref: str = ""
    capabilities: ModelCapabilities = ModelCapabilities()

    @abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    async def generate(self, request: ModelRequest) -> ModelResponse:
        parts: list[Part] = []
        usage = Usage()
        stop: StopReason | None = None
        async for event in self.stream(request):
            if isinstance(event, PartDone):
                parts.append(event.part)
            elif isinstance(event, UsageReport):
                usage += event.usage
            elif isinstance(event, Stop):
                stop = event.reason
        if stop is None:
            raise ModelError(
                f"{self.ref or type(self).__name__}: stream ended without a stop event"
            )
        return ModelResponse(
            message=Message(role="assistant", parts=parts, trust="self"),
            usage=usage,
            stop_reason=stop,
        )

    async def count_tokens(self, request: ModelRequest) -> int | None:
        return None
