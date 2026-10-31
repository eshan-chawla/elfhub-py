# Elves and runs

An [`Elf`][elfhub.Elf] combines a model, an identity, tools and modules.

## Declaring an elf

Pass settings to the constructor, or subclass and set class attributes. Constructor
arguments win over class attributes, so one class can serve as a template:

```python
from elfhub import Elf, tool
from elfhub.testing import ScriptedLLM


@tool(read_only=True)
def lookup_order(order_id: str) -> str:
    """Look up an order's status."""
    return f"Order {order_id} has shipped."


class SupportElf(Elf):
    name = "support"
    soul = "You are a friendly support agent for an online shop."
    instructions = "Look things up before answering. Never guess an order's status."
    tools = (lookup_order,)


elf = SupportElf(llm=ScriptedLLM("How can I help?"))
print(elf.run("hi").output)
```

| Setting | Meaning |
|---|---|
| `name` | the elf's name |
| `llm` | a `"provider/model"` string or any object implementing the LLM contract |
| `soul` | identity and personality, sent first in every system prompt |
| `instructions` | operating rules, sent after the soul |
| `tools` | tools or plain functions the elf can call |
| `modules` | modules that add abilities (see [Modules](modules.md)) |
| `policy`, `broker` | what needs approval and who approves (see [Approvals](approvals.md)) |
| `budget` | limits for each run |
| `home` | a folder for the elf's local state; sessions are saved there |

## Running

| Call | Use it from |
|---|---|
| `elf.run(task)` | sync code (it also works inside a running event loop by using a thread) |
| `await elf.arun(task)` | async code |
| `async for event in elf.astream(task)` | async code that wants live progress |
| `await elf.aresume(session_id, decisions)` | continuing a run that waits for approval |
| `elf(task)` | same as `run`; override `invoke` to change what calling an elf does |

Every run returns a [`RunResult`][elfhub.RunResult]. Its `status` is one of:

| Status | Meaning |
|---|---|
| `completed` | the model gave a final answer, in `output` |
| `suspended` | a tool call waits for approval, listed in `pending_approvals` |
| `budget_exceeded` | a budget limit was reached; `error` says which |
| `failed` | something went wrong; `error` says what |
| `cancelled` | the run's task was cancelled |

Runs report problems through `status` instead of raising. Configuration mistakes,
such as a missing model or two tools with the same name, raise
[`ConfigError`][elfhub.errors.ConfigError] when the elf starts.

## Sessions

Each run belongs to a session, which is a conversation. Pass `session=` to continue
one:

```python
from elfhub import Elf
from elfhub.testing import ScriptedLLM

elf = Elf(llm=ScriptedLLM("Noted: 42.", "You said 42."))

first = elf.run("Remember the number 42.")
second = elf.run("What number did I say?", session=first.session_id)

print(second.output)  # You said 42.
print([m.role for m in second.messages])  # ['user', 'assistant', 'user', 'assistant']
```

Without a `home` folder, sessions live in memory. With one, or when the elf is
loaded with `Elf.from_dir`, they are saved under `.elf/sessions/` and survive
restarts.

## Starting and stopping

Modules are started before a run and stopped after it. Runs on the same elf may
overlap (for example with `asyncio.gather`): modules start once and stop after the
last of those runs finishes. To keep them running across separate runs, use the elf
as an async context manager:

```python
import asyncio

from elfhub import Elf
from elfhub.testing import ScriptedLLM


async def main() -> None:
    elf = Elf(llm=ScriptedLLM("one", "two"))
    async with elf:
        await elf.arun("first task")
        await elf.arun("second task")


asyncio.run(main())
```
