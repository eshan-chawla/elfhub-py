"""The tool contract: what an elf can call, and what it gets back."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from elfhub.types import ContentPart, TextPart, Trust

if TYPE_CHECKING:
    from elfhub.events import Event


class ToolAnnotations(BaseModel):
    """Behaviour hints, mirroring MCP tool annotations (same pessimistic defaults) so
    MCP tools map onto ElfHub tools one to one. Approval policies match on these."""

    read_only: bool = False
    destructive: bool = True
    idempotent: bool = False
    open_world: bool = True
    parallel_safe: bool = False
    timeout_s: float | None = 120.0


class ToolSpec(BaseModel):
    """What the model sees: a name, a description and a JSON Schema for the arguments."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str = ""
    input_schema: dict[str, Any]
    annotations: ToolAnnotations = Field(default_factory=ToolAnnotations)


class ToolResult(BaseModel):
    content: list[ContentPart]
    is_error: bool = False
    trust: Trust | None = None
    """Overrides the trust level the runner would otherwise derive from the tool's
    annotations (open-world tools produce untrusted content)."""

    @classmethod
    def text(cls, text: str) -> ToolResult:
        return cls(content=[TextPart(text=text)])

    @classmethod
    def error(cls, message: str) -> ToolResult:
        return cls(content=[TextPart(text=message)], is_error=True)


@dataclass(frozen=True)
class ToolContext:
    """Passed to every tool call. Later phases add the runtime, workspace and secrets."""

    session_id: str
    run_id: str
    call_id: str
    emit: Callable[[Event], Awaitable[None]]


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...


def is_tool(obj: object) -> bool:
    """True for tool objects. Classes are excluded: a tool class is a factory, not a tool."""
    return not inspect.isclass(obj) and isinstance(obj, Tool)
