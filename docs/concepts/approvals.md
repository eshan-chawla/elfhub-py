# Approvals and budgets

## Policies

A [`Policy`][elfhub.Policy] decides, for each tool call, whether it may run
(`allow`), needs a human (`ask`), or is refused (`deny`). Rules match tool names with
glob patterns and can also match [annotations](tools.md#annotations):

```python
from elfhub import Policy, Rule

policy = Policy(
    rules=[
        Rule(tool="*", destructive=True, action="ask"),  # anything destructive needs a yes
        Rule(tool="delete_*", action="deny"),  # never delete
        Rule(tool="read_file", action="allow"),  # but reading is always fine
    ]
)
```

- **Within one policy, the last matching rule wins.** Put broad rules first and
  exceptions after them.
- **Across policy layers, the most restrictive answer wins.** Layers include your
  elf's policy, policies added by modules, and later the runtime's policy. An outer
  layer can therefore tighten what an elf may do, but never loosen it.

With no policy at all, every tool may run.

## Approval brokers

An `ask` goes to the elf's broker:

| Broker | Behaviour |
|---|---|
| `DenyAll` (default) | refuses, so headless elves never act without permission |
| `ApproveAll` | approves everything; for trusted, supervised setups |
| `CallbackBroker(fn)` | asks your function, which returns `True`, `False`, a `Decision`, or `None` |
| `SuspendAll` | never answers, so the run suspends and waits for a decision |

When a broker returns `None`, the run **suspends**. None of that turn's tool calls
run. You continue the run later with the decisions:

```python
import asyncio

from elfhub import Elf, Policy, Rule, tool
from elfhub.policy import SuspendAll
from elfhub.testing import ScriptedLLM, call


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"sent to {to}"


elf = Elf(
    llm=ScriptedLLM(call("send_email", to="bob@example.com", body="Hi!"), "Email sent."),
    tools=[send_email],
    policy=Policy(rules=[Rule(tool="send_*", action="ask")]),
    broker=SuspendAll(),
)


async def main() -> None:
    result = await elf.arun("Email Bob to say hi")
    print(result.status)  # suspended

    request = result.pending_approvals[0]
    print(request.call.name, request.call.arguments)

    result = await elf.aresume(result.session_id, {request.call.id: True})
    print(result.output)  # Email sent.


asyncio.run(main())
```

A refusal is reported to the model as an error result, so it can explain or try
something else. Decisions are recorded in the session log, and a decision made
before the run suspended is not asked again.

## Budgets

A [`Budget`][elfhub.Budget] limits each run. Limits are checked between steps, so the
step under way finishes first:

| Limit | Default |
|---|---|
| `max_turns` | 50 model calls |
| `max_tool_calls` | none (calls beyond it in a turn are refused, not run) |
| `max_tokens` | none (input + output) |
| `max_cost_usd` | none (when the adapter reports cost) |
| `max_wall_s` | none |

```python
from elfhub import Budget, Elf
from elfhub.testing import ScriptedLLM, call

looping = ScriptedLLM(*[call("nothing") for _ in range(10)])
result = Elf(llm=looping).run("go", budget=Budget(max_turns=3))

print(result.status)  # budget_exceeded
print(result.error)  # max_turns (3) reached
```
