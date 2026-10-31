"""Approval policies and brokers.

A :class:`Policy` decides, for each tool call, whether it may run (``allow``), needs a
human (``ask``) or is refused (``deny``). Within one policy the last matching rule
wins; across layers (built-in defaults, ``elf.toml``, the runtime) the most
restrictive answer wins, so an outer layer can tighten but never loosen.

``ask`` goes to an :class:`ApprovalBroker`. A broker that returns ``None`` has no
answer yet: the run suspends and continues later with ``Elf.aresume``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from fnmatch import fnmatchcase
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from elfhub.tools.base import ToolSpec
from elfhub.types import ToolCallPart

Action = Literal["allow", "ask", "deny"]
_STRICTNESS: dict[Action, int] = {"allow": 0, "ask": 1, "deny": 2}


class Rule(BaseModel):
    """Matches tools by name glob and, optionally, by annotation values."""

    model_config = ConfigDict(extra="forbid")

    tool: str = "*"
    read_only: bool | None = None
    destructive: bool | None = None
    open_world: bool | None = None
    action: Action

    def matches(self, spec: ToolSpec) -> bool:
        if not fnmatchcase(spec.name, self.tool):
            return False
        hints = spec.annotations
        return all(
            wanted is None or getattr(hints, field) == wanted
            for field, wanted in (
                ("read_only", self.read_only),
                ("destructive", self.destructive),
                ("open_world", self.open_world),
            )
        )


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(default_factory=list[Rule])
    default: Action = "allow"

    def decide(self, spec: ToolSpec) -> Action:
        for rule in reversed(self.rules):
            if rule.matches(spec):
                return rule.action
        return self.default


def decide(policies: Sequence[Policy], spec: ToolSpec) -> Action:
    """The most restrictive decision across policy layers (``allow`` if there are none)."""
    strictest: Action = "allow"
    for policy in policies:
        action = policy.decide(spec)
        if _STRICTNESS[action] > _STRICTNESS[strictest]:
            strictest = action
    return strictest


class Decision(BaseModel):
    approved: bool
    reason: str | None = None


class ApprovalRequest(BaseModel):
    session_id: str
    run_id: str
    call: ToolCallPart
    spec: ToolSpec


@runtime_checkable
class ApprovalBroker(Protocol):
    async def decide(self, request: ApprovalRequest) -> Decision | None:
        """Approve, refuse, or return ``None`` to suspend the run until a decision arrives."""
        ...


class DenyAll:
    """Refuses everything. The default when nobody is around to ask."""

    async def decide(self, request: ApprovalRequest) -> Decision | None:
        return Decision(approved=False, reason="no approver is configured for this elf")


class ApproveAll:
    """Approves everything. For trusted, supervised setups only."""

    async def decide(self, request: ApprovalRequest) -> Decision | None:
        return Decision(approved=True)


class SuspendAll:
    """Never answers, so every ``ask`` suspends the run for an outside decision."""

    async def decide(self, request: ApprovalRequest) -> Decision | None:
        return None


class CallbackBroker:
    """Asks a function, which returns a bool, a :class:`Decision` or ``None`` (suspend)."""

    def __init__(self, fn: Callable[[ApprovalRequest], Any]) -> None:
        self.fn = fn

    async def decide(self, request: ApprovalRequest) -> Decision | None:
        answer = self.fn(request)
        if inspect.isawaitable(answer):
            answer = await answer
        return as_decision(answer)


def as_decision(answer: bool | Decision | None) -> Decision | None:
    if answer is None or isinstance(answer, Decision):
        return answer
    return Decision(approved=bool(answer))
