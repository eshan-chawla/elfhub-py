"""Assembling what the model sees at the start of each turn."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from elfhub.types import Message, TextPart


@dataclass(frozen=True)
class RunContext:
    session_id: str
    run_id: str
    messages: Sequence[Message]


ContextProvider = Callable[[RunContext], str | Awaitable[str | None] | None]
"""Returns a piece of the system prompt for this turn, or ``None`` to add nothing."""


async def build_system(providers: Sequence[ContextProvider], ctx: RunContext) -> list[TextPart]:
    parts: list[TextPart] = []
    for provider in providers:
        text = provider(ctx)
        if inspect.isawaitable(text):
            text = await text
        if text:
            parts.append(TextPart(text=text))
    return parts
