import asyncio
from dataclasses import replace
from typing import Any

import pytest

from elfhub.errors import AuthFailed, ElfhubError, RateLimited
from elfhub.events import ApprovalRequested, Event, RunFinished, ToolStarted, Veto
from elfhub.hooks import ModelCall, ToolInvocation, ToolOutcome
from elfhub.loop.budget import Budget
from elfhub.loop.runner import RetryPolicy, RunContext, Runner
from elfhub.persist import MemorySessionStore, messages_from
from elfhub.policy import ApprovalRequest, CallbackBroker, Policy, Rule, SuspendAll
from elfhub.testing import ScriptedLLM, call
from elfhub.tools import Tool, ToolContext, ToolResult, tool
from elfhub.types import TextPart, ToolResultPart, Usage


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def make_runner(
    llm: ScriptedLLM,
    *tools: Tool,
    store: MemorySessionStore | None = None,
    **kwargs: Any,
) -> Runner:
    kwargs.setdefault("retry", RetryPolicy(base_delay=0))
    return Runner(
        llm=llm,
        tools={t.spec.name: t for t in tools},
        store=store or MemorySessionStore(),
        **kwargs,
    )


def tool_results(llm: ScriptedLLM, request_index: int) -> list[ToolResultPart]:
    return llm.requests[request_index].messages[-1].tool_results


async def test_a_plain_reply_completes_the_run():
    llm = ScriptedLLM("Hello!")

    result = await make_runner(llm).run("hi")

    assert result.status == "completed"
    assert result.output == "Hello!"
    assert [m.role for m in result.messages] == ["user", "assistant"]


async def test_tool_calls_run_and_their_results_go_back_to_the_model():
    llm = ScriptedLLM(call("add", a=2, b=3), "The answer is 5.")

    result = await make_runner(llm, add).run("what is 2+3?")

    assert result.output == "The answer is 5."
    [result_part] = tool_results(llm, 1)
    assert result_part.content == [TextPart(text="5")]
    assert not result_part.is_error


async def test_tools_are_advertised_to_the_model():
    llm = ScriptedLLM("ok")

    await make_runner(llm, add).run("hi")

    assert [t.name for t in llm.requests[0].tools] == ["add"]


async def test_context_providers_build_the_system_prompt_in_order():
    seen: list[str] = []

    def soul(ctx: RunContext) -> str:
        seen.append(ctx.session_id)
        return "You are Ada."

    async def rules(_ctx: RunContext) -> str:
        return "Be brief."

    llm = ScriptedLLM("ok")
    result = await make_runner(llm, context=[soul, lambda _ctx: None, rules]).run("hi")

    assert llm.requests[0].system == [TextPart(text="You are Ada."), TextPart(text="Be brief.")]
    assert seen == [result.session_id]


async def test_an_unknown_tool_becomes_an_error_result_and_the_run_continues():
    llm = ScriptedLLM(call("nope"), "sorry")

    result = await make_runner(llm, add).run("hi")

    assert result.status == "completed"
    [part] = tool_results(llm, 1)
    assert part.is_error and "nope" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_crashing_tool_becomes_an_error_result():
    @tool
    def explode() -> str:
        raise RuntimeError("kaboom")

    llm = ScriptedLLM(call("explode"), "it broke")

    result = await make_runner(llm, explode).run("hi")

    assert result.status == "completed"
    [part] = tool_results(llm, 1)
    assert part.is_error and "kaboom" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_slow_tool_times_out():
    @tool(timeout_s=0.01)
    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    llm = ScriptedLLM(call("slow"), "gave up")

    await make_runner(llm, slow).run("hi")

    [part] = tool_results(llm, 1)
    assert part.is_error and "timed out" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


async def test_parallel_safe_tools_run_concurrently_and_results_keep_call_order():
    both_running = asyncio.Event()
    running = 0

    @tool(parallel_safe=True)
    async def wait_for_peer(label: str) -> str:
        nonlocal running
        running += 1
        if running == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), timeout=1)  # deadlocks if run one at a time
        if label == "first":
            await asyncio.sleep(0.01)  # finish last
        return label

    llm = ScriptedLLM(
        [call("wait_for_peer", label="first"), call("wait_for_peer", label="second")], "done"
    )

    result = await make_runner(llm, wait_for_peer).run("hi")

    assert result.status == "completed"
    texts = [p.content[0].text for p in tool_results(llm, 1)]  # pyright: ignore[reportAttributeAccessIssue]
    assert texts == ["first", "second"]


async def test_tools_that_are_not_parallel_safe_run_one_after_another():
    log: list[str] = []

    @tool
    async def step(label: str) -> str:
        log.append(f"{label}-start")
        await asyncio.sleep(0)
        log.append(f"{label}-end")
        return label

    llm = ScriptedLLM([call("step", label="a"), call("step", label="b")], "done")

    await make_runner(llm, step).run("hi")

    assert log == ["a-start", "a-end", "b-start", "b-end"]


async def test_a_before_tool_call_veto_blocks_the_tool():
    runs: list[int] = []

    @tool
    def risky() -> str:
        runs.append(1)
        return "done"

    llm = ScriptedLLM(call("risky"), "ok")
    runner = make_runner(llm, risky)
    runner.bus.intercept("before_tool_call", lambda _call: Veto("too risky"))

    await runner.run("hi")

    assert runs == []
    [part] = tool_results(llm, 1)
    assert part.is_error and "too risky" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_before_tool_call_interceptor_can_rewrite_arguments():
    seen: list[ToolInvocation] = []

    def rewrite(invocation: ToolInvocation) -> ToolInvocation:
        seen.append(invocation)
        return replace(
            invocation, call=invocation.call.model_copy(update={"arguments": {"a": 10, "b": 20}})
        )

    llm = ScriptedLLM(call("add", a=1, b=1), "ok")
    runner = make_runner(llm, add)
    runner.bus.intercept("before_tool_call", rewrite)

    result = await runner.run("hi")

    assert tool_results(llm, 1)[0].content == [TextPart(text="30")]
    [invocation] = seen
    assert (invocation.spec.name, invocation.session_id, invocation.run_id) == (
        "add",
        result.session_id,
        result.run_id,
    )


async def test_an_after_tool_call_interceptor_can_redact_results():
    seen: list[ToolOutcome] = []

    def redact(outcome: ToolOutcome) -> ToolOutcome:
        seen.append(outcome)
        return replace(outcome, result=ToolResult.text("[redacted]"))

    llm = ScriptedLLM(call("add", a=1, b=1), "ok")
    runner = make_runner(llm, add)
    runner.bus.intercept("after_tool_call", redact)

    await runner.run("hi")

    assert tool_results(llm, 1)[0].content == [TextPart(text="[redacted]")]
    assert seen[0].result.content == [TextPart(text="2")]
    assert seen[0].call.arguments == {"a": 1, "b": 1}


async def test_a_before_model_call_interceptor_sees_and_can_change_the_request():
    def add_note(model_call: ModelCall) -> ModelCall:
        request = model_call.request
        extra = [*request.system, TextPart(text="Injected note.")]
        return replace(model_call, request=request.model_copy(update={"system": extra}))

    llm = ScriptedLLM("ok")
    runner = make_runner(llm)
    runner.bus.intercept("before_model_call", add_note)

    await runner.run("hi")

    assert llm.requests[0].system == [TextPart(text="Injected note.")]


async def test_an_interceptor_returning_the_wrong_type_fails_the_run_clearly():
    llm = ScriptedLLM(call("add", a=1, b=1), "ok")
    runner = make_runner(llm, add)
    runner.bus.intercept("before_tool_call", lambda invocation: invocation.call)  # old style

    result = await runner.run("hi")

    assert result.status == "failed"
    assert result.error is not None and "ToolInvocation" in result.error


async def test_retries_never_wait_past_the_wall_clock_budget():
    llm = ScriptedLLM(RateLimited(retry_after=60), "too late")

    result = await asyncio.wait_for(
        make_runner(llm).run("hi", budget=Budget(max_wall_s=5)), timeout=2
    )

    assert result.status == "failed"
    assert result.error is not None and "RateLimited" in result.error


async def test_a_policy_deny_refuses_the_call():
    llm = ScriptedLLM(call("add", a=1, b=1), "ok")

    await make_runner(llm, add, policies=[Policy(rules=[Rule(tool="add", action="deny")])]).run(
        "hi"
    )

    [part] = tool_results(llm, 1)
    assert part.is_error and "denied" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


ASK_ADD = [Policy(rules=[Rule(tool="add", action="ask")])]


async def test_ask_runs_the_tool_when_the_broker_approves():
    asked: list[str] = []

    def approve(request: ApprovalRequest) -> bool:
        asked.append(request.call.name)
        return True

    llm = ScriptedLLM(call("add", a=1, b=2), "ok")

    await make_runner(llm, add, policies=ASK_ADD, broker=CallbackBroker(approve)).run("hi")

    assert asked == ["add"]
    assert tool_results(llm, 1)[0].content == [TextPart(text="3")]


async def test_ask_refused_by_the_broker_tells_the_model():
    llm = ScriptedLLM(call("add", a=1, b=2), "ok")

    await make_runner(llm, add, policies=ASK_ADD, broker=CallbackBroker(lambda _r: False)).run("hi")

    [part] = tool_results(llm, 1)
    assert part.is_error and "not approved" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


async def test_ask_without_an_answer_suspends_and_resume_finishes_the_run():
    store = MemorySessionStore()
    llm = ScriptedLLM(call("add", a=1, b=2, id="c1"), "It is 3.")
    runner = make_runner(llm, add, store=store, policies=ASK_ADD, broker=SuspendAll())

    suspended = await runner.run("what is 1+2?")

    assert suspended.status == "suspended"
    assert [p.call.id for p in suspended.pending_approvals] == ["c1"]
    assert len(llm.requests) == 1
    events = await store.events(suspended.session_id)
    assert any(isinstance(e, ApprovalRequested) for e in events)
    assert not any(isinstance(e, ToolStarted) for e in events)

    resumed = await runner.resume(suspended.session_id, {"c1": True})

    assert resumed.status == "completed"
    assert resumed.output == "It is 3."
    assert resumed.session_id == suspended.session_id
    assert tool_results(llm, 1)[0].content == [TextPart(text="3")]


async def test_resume_with_a_refusal_tells_the_model():
    llm = ScriptedLLM(call("add", a=1, b=2, id="c1"), "ok, I won't")
    runner = make_runner(llm, add, policies=ASK_ADD, broker=SuspendAll())
    suspended = await runner.run("hi")

    resumed = await runner.resume(suspended.session_id, {"c1": False})

    assert resumed.status == "completed"
    [part] = tool_results(llm, 1)
    assert part.is_error and "not approved" in part.content[0].text  # pyright: ignore[reportAttributeAccessIssue]


async def test_nothing_in_a_suspended_batch_runs_until_resume_and_then_runs_once():
    runs: list[str] = []

    @tool
    def free() -> str:
        runs.append("free")
        return "free"

    llm = ScriptedLLM([call("free"), call("add", a=1, b=1, id="c2")], "done")
    runner = make_runner(llm, free, add, policies=ASK_ADD, broker=SuspendAll())

    suspended = await runner.run("hi")
    assert runs == []

    await runner.resume(suspended.session_id, {"c2": True})
    assert runs == ["free"]


async def test_decisions_made_before_suspending_are_not_asked_again():
    asked: list[str] = []

    def broker(request: ApprovalRequest) -> bool | None:
        asked.append(request.call.id)
        return True if request.call.id == "c1" else None

    llm = ScriptedLLM([call("add", a=1, b=1, id="c1"), call("add", a=2, b=2, id="c2")], "done")
    runner = make_runner(llm, add, policies=ASK_ADD, broker=CallbackBroker(broker))

    suspended = await runner.run("hi")
    resumed = await runner.resume(suspended.session_id, {"c2": True})

    assert asked == ["c1", "c2"]
    assert resumed.status == "completed"
    texts = [p.content[0].text for p in tool_results(llm, 1)]  # pyright: ignore[reportAttributeAccessIssue]
    assert texts == ["2", "4"]


async def test_new_input_is_refused_while_a_session_waits_for_approval():
    runner = make_runner(
        ScriptedLLM(call("add", a=1, b=1)), add, policies=ASK_ADD, broker=SuspendAll()
    )
    suspended = await runner.run("hi")

    with pytest.raises(ElfhubError, match="approval"):
        await runner.run("something else", session_id=suspended.session_id)


async def test_resuming_a_session_with_nothing_pending_is_an_error():
    runner = make_runner(ScriptedLLM("hi"))
    finished = await runner.run("hello")

    with pytest.raises(ElfhubError, match="nothing"):
        await runner.resume(finished.session_id, {})


async def test_the_turn_budget_stops_a_looping_model():
    llm = ScriptedLLM(*[call("add", a=1, b=1) for _ in range(10)])

    result = await make_runner(llm, add).run("loop", budget=Budget(max_turns=3))

    assert result.status == "budget_exceeded"
    assert result.error is not None and "max_turns" in result.error
    assert len(llm.requests) == 3


async def test_the_token_budget_counts_model_usage():
    llm = ScriptedLLM(
        *[call("add", a=1, b=1) for _ in range(10)], usage=Usage(input_tokens=40, output_tokens=10)
    )

    result = await make_runner(llm, add).run("loop", budget=Budget(max_tokens=120))

    assert result.status == "budget_exceeded"
    assert len(llm.requests) == 3
    assert result.usage == Usage(input_tokens=120, output_tokens=30)


async def test_rate_limits_are_retried():
    llm = ScriptedLLM(RateLimited(retry_after=0), RateLimited(retry_after=0), "finally")

    result = await make_runner(llm).run("hi")

    assert (result.status, result.output) == ("completed", "finally")


async def test_the_run_fails_when_retries_run_out():
    llm = ScriptedLLM(*[RateLimited(retry_after=0) for _ in range(3)])

    result = await make_runner(llm, retry=RetryPolicy(max_attempts=3, base_delay=0)).run("hi")

    assert result.status == "failed"
    assert result.error is not None and "RateLimited" in result.error
    assert llm.remaining == 0


async def test_errors_that_retrying_cannot_fix_fail_immediately():
    llm = ScriptedLLM(AuthFailed("bad key"), "never reached")

    result = await make_runner(llm).run("hi")

    assert result.status == "failed"
    assert result.error is not None and "bad key" in result.error
    assert len(llm.requests) == 1


async def test_the_event_log_replays_to_the_same_messages():
    store = MemorySessionStore()
    llm = ScriptedLLM(call("add", a=2, b=3), "5")

    result = await make_runner(llm, add, store=store).run("2+3?")

    assert messages_from(await store.events(result.session_id)) == result.messages
    assert isinstance((await store.events(result.session_id))[-1], RunFinished)


async def test_a_later_run_continues_the_same_session():
    llm = ScriptedLLM("first answer", "second answer")
    runner = make_runner(llm)

    first = await runner.run("one")
    second = await runner.run("two", session_id=first.session_id)

    assert [m.text for m in llm.requests[1].messages] == ["one", "first answer", "two"]
    assert len(second.messages) == 4


async def test_open_world_tool_output_is_marked_untrusted():
    @tool(open_world=False)
    def local() -> str:
        return "mine"

    @tool
    def web() -> str:
        return "from the internet"

    llm = ScriptedLLM(call("local"), call("web"), "done")

    result = await make_runner(llm, local, web).run("hi")

    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert [m.trust for m in tool_messages] == ["system", "untrusted"]


async def test_tools_can_emit_their_own_events():
    class Progress(Event):
        type: str = "test.progress"
        percent: int

    @tool
    async def work(ctx: ToolContext) -> str:
        await ctx.emit(Progress(percent=50))
        return "done"

    store = MemorySessionStore()
    result = await make_runner(ScriptedLLM(call("work"), "ok"), work, store=store).run("hi")

    [progress] = [e for e in await store.events(result.session_id) if isinstance(e, Progress)]
    assert (progress.session_id, progress.run_id, progress.percent) == (
        result.session_id,
        result.run_id,
        50,
    )


ASK_SEND = [Policy(rules=[Rule(tool="send", action="ask")])]


async def test_an_approval_never_carries_over_to_a_later_call_with_the_same_id():
    sent: list[str] = []

    @tool
    def send(to: str) -> str:
        sent.append(to)
        return "sent"

    # Some adapters number calls per turn, so ids repeat across turns.
    llm = ScriptedLLM(
        call("send", to="bob", id="call_0"),
        "sent to bob",
        call("send", to="everyone@company.com", id="call_0"),
        "done",
    )
    runner = make_runner(llm, send, policies=ASK_SEND, broker=SuspendAll())
    first = await runner.run("email bob")
    await runner.resume(first.session_id, {"call_0": True})
    await runner.run("email everyone", session_id=first.session_id)

    again = await runner.resume(first.session_id, {})

    assert again.status == "suspended"
    assert sent == ["bob"]


async def test_resume_after_an_interruption_does_not_repeat_finished_tools():
    sent: list[str] = []
    hanging = asyncio.Event()

    @tool
    def send_email(to: str) -> str:
        sent.append(to)
        return "sent"

    @tool
    async def slow() -> str:
        if not hanging.is_set():
            hanging.set()
            await asyncio.sleep(10)  # interrupted here the first time
        return "slow done"

    llm = ScriptedLLM([call("send_email", to="bob"), call("slow")], "all done")
    runner = make_runner(llm, send_email, slow)
    task = asyncio.create_task(runner.run("go", session_id="s1"))
    await asyncio.wait_for(hanging.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    result = await runner.resume("s1", {})

    assert result.status == "completed"
    assert sent == ["bob"]
    texts = [p.content[0].text for p in tool_results(llm, 1)]  # pyright: ignore[reportAttributeAccessIssue]
    assert texts == ["sent", "slow done"]


async def test_duplicate_tool_call_ids_in_one_turn_fail_the_run():
    runs: list[int] = []

    @tool
    def once(n: int) -> str:
        runs.append(n)
        return "ok"

    llm = ScriptedLLM([call("once", n=1, id="dup"), call("once", n=2, id="dup")])

    result = await make_runner(llm, once).run("hi")

    assert result.status == "failed"
    assert result.error is not None and "dup" in result.error
    assert runs == []


async def test_the_tool_call_budget_also_caps_calls_within_a_turn():
    runs: list[int] = []

    @tool
    def count(n: int) -> str:
        runs.append(n)
        return "ok"

    llm = ScriptedLLM([call("count", n=i) for i in range(3)], "never reached")

    result = await make_runner(llm, count).run("hi", budget=Budget(max_tool_calls=2))

    assert runs == [0, 1]
    assert tool_results(llm, 0) == []  # first request has no tool results yet
    assert result.status == "budget_exceeded"
    last_tool_message = [m for m in result.messages if m.role == "tool"][-1]
    assert last_tool_message.tool_results[2].is_error
