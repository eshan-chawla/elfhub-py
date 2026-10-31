"""The turn loop: ask the model, run the tools it calls, repeat.

Every step is written to the session store before it is acted on, so a session can
be replayed, inspected, or resumed from its log after an approval or an
interruption. Each tool call's result is recorded as soon as it finishes, so
resuming never runs a finished call again.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from elfhub.errors import ElfhubError, ModelError, RateLimited, TransientModelError
from elfhub.events import (
    ApprovalDecided,
    ApprovalRequested,
    Event,
    EventBus,
    MessageAdded,
    ModelCompleted,
    RunFinished,
    RunStarted,
    ToolCompleted,
    ToolStarted,
    TurnStarted,
    Veto,
)
from elfhub.hooks import (
    AFTER_TOOL_CALL,
    BEFORE_MODEL_CALL,
    BEFORE_TOOL_CALL,
    ModelCall,
    ToolInvocation,
    ToolOutcome,
)
from elfhub.llm.protocol import LLM, ModelRequest, ModelResponse
from elfhub.loop.budget import Budget, BudgetTracker
from elfhub.loop.context import ContextProvider, RunContext, build_system
from elfhub.persist.store import MemorySessionStore, SessionStore, messages_from
from elfhub.policy.approvals import (
    ApprovalBroker,
    ApprovalRequest,
    Decision,
    DenyAll,
    Policy,
    as_decision,
    decide,
)
from elfhub.tools.base import Tool, ToolContext, ToolResult
from elfhub.types import Message, ToolCallPart, ToolResultPart, Trust, Usage, new_id

__all__ = ["RetryPolicy", "RunContext", "RunResult", "RunStatus", "Runner"]

logger = logging.getLogger("elfhub.loop")

RunStatus = Literal["completed", "suspended", "budget_exceeded", "failed", "cancelled"]
_Payload = TypeVar("_Payload")
_Outcome = tuple[ToolResult, Trust]


class RunResult(BaseModel):
    session_id: str
    run_id: str
    status: RunStatus
    output: str | None = None
    error: str | None = None
    messages: list[Message] = Field(default_factory=list[Message])
    pending_approvals: list[ApprovalRequest] = Field(default_factory=list[ApprovalRequest])
    usage: Usage = Field(default_factory=Usage)


@dataclass(frozen=True)
class RetryPolicy:
    """Retries for rate limits and transient model errors, with exponential backoff.

    A provider's ``retry_after`` is honoured as given. No retry waits past the run's
    wall-clock budget.
    """

    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.1
    """Up to this fraction is added to each backoff, so clients don't retry in lockstep."""

    def delay(self, attempt: int) -> float:
        backoff = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return backoff * (1 + random.uniform(0, self.jitter))


@dataclass
class _Run:
    session_id: str
    run_id: str
    messages: list[Message]
    tracker: BudgetTracker
    logged_decisions: set[str] = field(default_factory=set[str])
    completed: dict[str, _Outcome] = field(default_factory=dict[str, _Outcome])


@dataclass
class _Batch:
    """What the log says about the calls of the last assistant message."""

    decisions: dict[str, Decision] = field(default_factory=dict[str, Decision])
    completed: dict[str, _Outcome] = field(default_factory=dict[str, _Outcome])


def _as_messages(value: str | Message | Sequence[Message]) -> list[Message]:
    if isinstance(value, str):
        return [Message.user(value)]
    if isinstance(value, Message):
        return [value]
    return list(value)


def _unanswered_calls(messages: Sequence[Message]) -> list[ToolCallPart]:
    """Tool calls in the final assistant message that never got results.

    The results of a batch of calls join the conversation together, so only the last
    message can have unanswered calls.
    """
    if messages and messages[-1].role == "assistant":
        return messages[-1].tool_calls
    return []


def _last_batch(events: Sequence[Event]) -> _Batch:
    """Decisions and finished results recorded since the last assistant message.

    Only events after that message count: tool call ids are only unique within one
    model reply, so an older decision must never apply to a newer call.
    """
    start = 0
    for index, event in enumerate(events):
        if isinstance(event, MessageAdded) and event.message.role == "assistant":
            start = index + 1
    batch = _Batch()
    for event in events[start:]:
        if isinstance(event, ApprovalDecided):
            batch.decisions[event.call_id] = Decision(approved=event.approved, reason=event.reason)
        elif isinstance(event, ToolCompleted) and event.result is not None:
            batch.completed[event.call_id] = (event.result, event.trust)
    return batch


def _check_payload(hook: str, value: object, expected: type[_Payload]) -> _Payload | Veto:
    if isinstance(value, Veto | expected):
        return value
    raise TypeError(
        f"{hook} interceptors must return a {expected.__name__}, a Veto or None, "
        f"not {type(value).__name__}"
    )


class Runner:
    def __init__(
        self,
        *,
        llm: LLM,
        tools: Mapping[str, Tool] | None = None,
        store: SessionStore | None = None,
        bus: EventBus | None = None,
        policies: Sequence[Policy] = (),
        broker: ApprovalBroker | None = None,
        context: Sequence[ContextProvider] = (),
        retry: RetryPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.tools = dict(tools or {})
        self.store = store or MemorySessionStore()
        self.bus = bus or EventBus()
        self.policies = list(policies)
        self.broker = broker or DenyAll()
        self.context = list(context)
        self.retry = retry or RetryPolicy()

    async def run(
        self,
        input: str | Message | Sequence[Message],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> RunResult:
        """Add ``input`` to the session (a new one unless ``session_id`` is given) and run."""
        session_id = session_id or new_id("ses")
        history = messages_from(await self.store.events(session_id))
        if _unanswered_calls(history):
            raise ElfhubError(
                f"session {session_id!r} has unanswered tool calls (waiting for approval, or "
                "interrupted); call resume() with any approval decisions to continue it"
            )
        state = _Run(
            session_id, run_id or new_id("run"), history, BudgetTracker(budget or Budget())
        )
        await self._record(state, RunStarted())
        for message in _as_messages(input):
            await self._add_message(state, message)
        return await self._drive(state)

    async def resume(
        self,
        session_id: str,
        decisions: Mapping[str, Decision | bool],
        *,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> RunResult:
        """Continue a session whose last tool calls are unanswered.

        That is a session waiting for approval (``decisions`` maps tool call ids to
        answers) or one that was interrupted. Calls that already finished are not
        run again.
        """
        events = await self.store.events(session_id)
        history = messages_from(events)
        pending = _unanswered_calls(history)
        if not pending:
            raise ElfhubError(f"session {session_id!r} has nothing waiting for approval")
        batch = _last_batch(events)
        state = _Run(
            session_id, run_id or new_id("run"), history, BudgetTracker(budget or Budget())
        )
        state.logged_decisions = set(batch.decisions)
        state.completed = batch.completed
        known = dict(batch.decisions)
        for call_id, answer in decisions.items():
            decision = as_decision(answer)
            if decision is not None:
                known[call_id] = decision
                state.logged_decisions.discard(call_id)
        await self._record(state, RunStarted())
        return await self._drive(state, pending_calls=pending, decisions=known)

    async def _drive(
        self,
        state: _Run,
        *,
        pending_calls: Sequence[ToolCallPart] = (),
        decisions: Mapping[str, Decision] | None = None,
    ) -> RunResult:
        try:
            if pending_calls:
                waiting = await self._handle_calls(state, pending_calls, decisions or {})
                if waiting:
                    return await self._finish(state, "suspended", pending=waiting)
            while True:
                reason = state.tracker.exceeded()
                if reason:
                    return await self._finish(state, "budget_exceeded", error=reason)
                state.tracker.charge_turn()
                await self._record(state, TurnStarted(turn=state.tracker.turns))
                response = await self._ask_model(state)
                if isinstance(response, Veto):
                    return await self._finish(
                        state, "failed", error=f"model call vetoed: {response.reason}"
                    )
                state.tracker.charge_usage(response.usage)
                await self._record(
                    state, ModelCompleted(usage=response.usage, stop_reason=response.stop_reason)
                )
                await self._add_message(state, response.message)
                calls = response.message.tool_calls
                if not calls:
                    return await self._finish(state, "completed", output=response.message.text)
                waiting = await self._handle_calls(state, calls, {})
                if waiting:
                    return await self._finish(state, "suspended", pending=waiting)
        except asyncio.CancelledError:
            await self._finish(state, "cancelled")
            raise
        except Exception as exc:
            if isinstance(exc, ElfhubError):
                logger.warning("run %s failed: %s", state.run_id, exc)
            else:
                logger.exception("run %s failed", state.run_id)
            return await self._finish(state, "failed", error=f"{type(exc).__name__}: {exc}")

    async def _ask_model(self, state: _Run) -> ModelResponse | Veto:
        ctx = RunContext(state.session_id, state.run_id, tuple(state.messages))
        request = ModelRequest(
            system=await build_system(self.context, ctx),
            messages=list(state.messages),
            tools=[t.spec for t in self.tools.values()],
        )
        hooked = await self._hook(
            BEFORE_MODEL_CALL, ModelCall(request, state.session_id, state.run_id), ModelCall
        )
        if isinstance(hooked, Veto):
            return hooked
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self.llm.generate(hooked.request)
            except (RateLimited, TransientModelError) as exc:
                if attempt >= self.retry.max_attempts:
                    raise
                delay = self.retry.delay(attempt)
                if isinstance(exc, RateLimited) and exc.retry_after is not None:
                    delay = exc.retry_after
                remaining = state.tracker.remaining_wall_s()
                if remaining is not None and delay >= remaining:
                    raise
                logger.info("model call failed (%s); retrying in %.2fs", exc, delay)
                await asyncio.sleep(delay)

    async def _handle_calls(
        self, state: _Run, calls: Sequence[ToolCallPart], decisions: Mapping[str, Decision]
    ) -> list[ApprovalRequest]:
        """Run a batch of tool calls. If any still needs approval, run none and return them."""
        ids = [c.id for c in calls]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ModelError(f"the model reused tool call ids within one reply: {duplicates}")

        outcomes: dict[str, _Outcome] = {}
        approved: list[tuple[str, ToolCallPart, Tool]] = []
        waiting: list[ApprovalRequest] = []
        for original in calls:
            if original.id in state.completed:
                outcomes[original.id] = state.completed[original.id]
                continue
            tool = self.tools.get(original.name)
            if tool is None:
                available = ", ".join(sorted(self.tools)) or "none"
                outcomes[original.id] = (
                    ToolResult.error(f"unknown tool {original.name!r} (available: {available})"),
                    "system",
                )
                continue
            hooked = await self._hook(
                BEFORE_TOOL_CALL,
                ToolInvocation(original, tool.spec, state.session_id, state.run_id),
                ToolInvocation,
            )
            if isinstance(hooked, Veto):
                outcomes[original.id] = (ToolResult.error(f"blocked: {hooked.reason}"), "system")
                continue
            call = hooked.call
            action = decide(self.policies, tool.spec)
            if action == "deny":
                outcomes[original.id] = (
                    ToolResult.error(f"denied by policy: {tool.spec.name} may not run"),
                    "system",
                )
                continue
            if action == "ask":
                request = ApprovalRequest(
                    session_id=state.session_id, run_id=state.run_id, call=call, spec=tool.spec
                )
                decision = decisions.get(original.id)
                if decision is None:
                    decision = await self.broker.decide(request)
                    if decision is None:
                        waiting.append(request)
                        continue
                if original.id not in state.logged_decisions:
                    await self._record(
                        state,
                        ApprovalDecided(
                            call_id=original.id, approved=decision.approved, reason=decision.reason
                        ),
                    )
                if not decision.approved:
                    reason = decision.reason or "the approver said no"
                    outcomes[original.id] = (ToolResult.error(f"not approved: {reason}"), "system")
                    continue
            approved.append((original.id, call, tool))

        if waiting:
            for request in waiting:
                await self._record(
                    state,
                    ApprovalRequested(
                        call_id=request.call.id,
                        name=request.call.name,
                        arguments=request.call.arguments,
                    ),
                )
            return waiting

        limit = state.tracker.budget.max_tool_calls
        if limit is not None:
            allowed = max(0, limit - state.tracker.tool_calls)
            for call_id, _, _ in approved[allowed:]:
                outcomes[call_id] = (
                    ToolResult.error(f"not run: the tool call budget ({limit}) is used up"),
                    "system",
                )
            approved = approved[:allowed]

        outcomes.update(await self._execute_all(state, approved))
        parts = [
            ToolResultPart(
                call_id=c.id, content=outcomes[c.id][0].content, is_error=outcomes[c.id][0].is_error
            )
            for c in calls
        ]
        trust: Trust = (
            "untrusted" if any(t == "untrusted" for _, t in outcomes.values()) else "system"
        )
        await self._add_message(state, Message(role="tool", parts=list(parts), trust=trust))
        return []

    async def _execute_all(
        self, state: _Run, approved: Sequence[tuple[str, ToolCallPart, Tool]]
    ) -> dict[str, _Outcome]:
        if len(approved) > 1 and all(t.spec.annotations.parallel_safe for _, _, t in approved):
            async with asyncio.TaskGroup() as group:
                tasks = {
                    call_id: group.create_task(self._execute(state, call_id, call, tool))
                    for call_id, call, tool in approved
                }
            return {call_id: task.result() for call_id, task in tasks.items()}
        return {
            call_id: await self._execute(state, call_id, call, tool)
            for call_id, call, tool in approved
        }

    async def _execute(self, state: _Run, call_id: str, call: ToolCallPart, tool: Tool) -> _Outcome:
        await self._record(
            state, ToolStarted(call_id=call_id, name=call.name, arguments=call.arguments)
        )
        state.tracker.charge_tool_calls(1)

        async def emit(event: Event) -> None:
            await self._record(state, event)

        ctx = ToolContext(
            session_id=state.session_id, run_id=state.run_id, call_id=call_id, emit=emit
        )
        timeout = tool.spec.annotations.timeout_s
        try:
            async with asyncio.timeout(timeout):
                result = await tool(call.arguments, ctx)
        except TimeoutError:
            result = ToolResult.error(f"{call.name} timed out after {timeout}s")
        except Exception as exc:
            logger.warning("tool %s failed", call.name, exc_info=True)
            result = ToolResult.error(f"{call.name} failed: {type(exc).__name__}: {exc}")
        hooked = await self._hook(
            AFTER_TOOL_CALL,
            ToolOutcome(call, tool.spec, result, state.session_id, state.run_id),
            ToolOutcome,
        )
        result = (
            ToolResult.error(f"result withheld: {hooked.reason}")
            if isinstance(hooked, Veto)
            else hooked.result
        )
        trust: Trust = result.trust or (
            "untrusted" if tool.spec.annotations.open_world else "system"
        )
        await self._record(
            state,
            ToolCompleted(
                call_id=call_id,
                name=call.name,
                is_error=result.is_error,
                result=result,
                trust=trust,
            ),
        )
        return result, trust

    async def _hook(
        self, hook: str, payload: _Payload, expected: type[_Payload]
    ) -> _Payload | Veto:
        return _check_payload(hook, await self.bus.run_hook(hook, payload), expected)

    async def _add_message(self, state: _Run, message: Message) -> None:
        state.messages.append(message)
        await self._record(state, MessageAdded(message=message))

    async def _record(self, state: _Run, event: Event) -> Event:
        stamped = event.model_copy(update={"session_id": state.session_id, "run_id": state.run_id})
        stored = await self.store.append(stamped)
        await self.bus.emit(stored)
        return stored

    async def _finish(
        self,
        state: _Run,
        status: RunStatus,
        *,
        output: str | None = None,
        error: str | None = None,
        pending: Sequence[ApprovalRequest] = (),
    ) -> RunResult:
        usage = state.tracker.usage
        await self._record(
            state, RunFinished(status=status, output=output, error=error, usage=usage)
        )
        return RunResult(
            session_id=state.session_id,
            run_id=state.run_id,
            status=status,
            output=output,
            error=error,
            messages=list(state.messages),
            pending_approvals=list(pending),
            usage=usage,
        )
