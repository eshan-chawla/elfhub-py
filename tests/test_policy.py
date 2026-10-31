import pytest

from elfhub.loop.budget import Budget, BudgetTracker
from elfhub.policy import (
    ApprovalRequest,
    ApproveAll,
    CallbackBroker,
    Decision,
    DenyAll,
    Policy,
    Rule,
    decide,
)
from elfhub.tools import ToolAnnotations, ToolSpec
from elfhub.types import ToolCallPart, Usage


def spec(name: str, **annotations: bool) -> ToolSpec:
    return ToolSpec(name=name, input_schema={}, annotations=ToolAnnotations(**annotations))


def test_no_matching_rule_falls_back_to_the_default():
    assert Policy(default="ask").decide(spec("anything")) == "ask"


def test_rules_match_tool_name_globs():
    policy = Policy(rules=[Rule(tool="fs_*", action="deny")])

    assert policy.decide(spec("fs_write")) == "deny"
    assert policy.decide(spec("web_fetch")) == "allow"


def test_rules_match_annotations():
    policy = Policy(rules=[Rule(read_only=False, action="ask")])

    assert policy.decide(spec("look", read_only=True)) == "allow"
    assert policy.decide(spec("touch", read_only=False)) == "ask"


def test_within_one_policy_the_last_matching_rule_wins():
    policy = Policy(rules=[Rule(tool="*", action="ask"), Rule(tool="read_file", action="allow")])

    assert policy.decide(spec("read_file")) == "allow"
    assert policy.decide(spec("write_file")) == "ask"


def test_across_layers_the_most_restrictive_decision_wins():
    owner = Policy(rules=[Rule(tool="send_email", action="allow")])
    runtime = Policy(rules=[Rule(tool="send_*", action="ask")])
    lockdown = Policy(rules=[Rule(tool="delete_*", action="deny")])

    assert decide([owner, runtime, lockdown], spec("send_email")) == "ask"
    assert decide([owner, runtime, lockdown], spec("delete_all")) == "deny"
    assert decide([owner, runtime, lockdown], spec("read")) == "allow"


def test_no_policy_layers_allow_everything():
    assert decide([], spec("anything")) == "allow"


def approval() -> ApprovalRequest:
    return ApprovalRequest(
        session_id="s", run_id="r", call=ToolCallPart(id="c", name="x"), spec=spec("x")
    )


async def test_deny_all_broker_refuses_with_a_reason():
    decision = await DenyAll().decide(approval())

    assert decision is not None and not decision.approved and decision.reason


async def test_approve_all_broker_approves():
    assert await ApproveAll().decide(approval()) == Decision(approved=True)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (True, Decision(approved=True)),
        (False, Decision(approved=False)),
        (None, None),
        (Decision(approved=False, reason="busy"), Decision(approved=False, reason="busy")),
    ],
)
async def test_callback_broker_accepts_bools_decisions_and_none(
    answer: bool | Decision | None, expected: Decision | None
):
    assert await CallbackBroker(lambda _req: answer).decide(approval()) == expected


async def test_callback_broker_awaits_async_callbacks():
    async def ask(_req: ApprovalRequest) -> bool:
        return True

    assert await CallbackBroker(ask).decide(approval()) == Decision(approved=True)


def test_budget_tracker_reports_the_first_limit_reached():
    tracker = BudgetTracker(Budget(max_turns=2, max_tokens=100))

    tracker.charge_turn()
    assert tracker.exceeded() is None
    tracker.charge_turn()
    assert "max_turns" in (tracker.exceeded() or "")


def test_budget_tracker_counts_tokens_tool_calls_and_cost():
    tracker = BudgetTracker(Budget(max_turns=None, max_tokens=10))
    tracker.charge_usage(Usage(input_tokens=6, output_tokens=4))
    assert "max_tokens" in (tracker.exceeded() or "")

    tracker = BudgetTracker(Budget(max_turns=None, max_tool_calls=2))
    tracker.charge_tool_calls(2)
    assert "max_tool_calls" in (tracker.exceeded() or "")

    tracker = BudgetTracker(Budget(max_turns=None, max_cost_usd=0.5))
    tracker.charge_usage(Usage(cost_usd=0.5))
    assert "max_cost_usd" in (tracker.exceeded() or "")


def test_budget_tracker_measures_wall_time_with_the_given_clock():
    now = [100.0]
    tracker = BudgetTracker(Budget(max_turns=None, max_wall_s=5), clock=lambda: now[0])

    now[0] = 104.9
    assert tracker.exceeded() is None
    now[0] = 105.0
    assert "max_wall_s" in (tracker.exceeded() or "")


def test_default_budget_stops_runaway_loops():
    tracker = BudgetTracker(Budget())
    for _ in range(1000):
        if tracker.exceeded():
            break
        tracker.charge_turn()

    assert tracker.turns < 1000
