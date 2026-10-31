# Events and sessions

Everything that happens in a run is an **event**. Every event is:

- **recorded** in the session log, before anything acts on it;
- **delivered** to observers;
- **streamed** to callers of `astream`.

## Event types

| Type | When |
|---|---|
| `run.started` | a run begins (or resumes) |
| `turn.started` | the elf is about to ask the model |
| `message.added` | a message joins the conversation: user input, model reply, or tool results |
| `model.completed` | the model answered, with token usage |
| `tool.started` / `tool.completed` | a tool call runs |
| `approval.requested` / `approval.decided` | a tool call needs, or got, a decision |
| `run.finished` | a run ends, with its status, output and usage |

Every event carries `session_id`, `run_id`, a `seq` number (its position in the
session log), a timestamp and a version `v`. Plugins can define their own event types
by subclassing [`Event`][elfhub.events.Event].

## Streaming a run

```python
import asyncio

from elfhub import Elf, tool
from elfhub.testing import ScriptedLLM, call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


async def main() -> None:
    elf = Elf(llm=ScriptedLLM(call("add", a=1, b=2), "3"), tools=[add])
    async for event in elf.astream("1 + 2?"):
        print(event.seq, event.type)


asyncio.run(main())
```

## Observers and interceptors

**Observers** watch. **Interceptors** can change what happens:

```python
from elfhub import Elf, tool
from elfhub.events import Veto
from elfhub.testing import ScriptedLLM, call

deleted: list[str] = []


@tool
def rm_rf(path: str) -> str:
    """Delete a folder and everything in it."""
    deleted.append(path)
    return "deleted"


elf = Elf(llm=ScriptedLLM(call("rm_rf", path="/"), "I wasn't allowed to."), tools=[rm_rf])

elf.on("tool.started", lambda event: print("running", event.name))
elf.intercept(
    "before_tool_call",
    lambda invocation: Veto("not today") if invocation.call.name.startswith("rm") else None,
)

result = elf.run("clean up")
print(result.output)  # I wasn't allowed to.
assert result.status == "completed"
assert deleted == []  # the veto stopped the call before it ran
```

An interceptor receives a small, frozen payload from [`elfhub.hooks`][elfhub.hooks]
that also says which session and run it belongs to. It returns one of three things:

- `None` to leave the payload unchanged;
- a modified copy, made with `dataclasses.replace(payload, ...)`;
- a [`Veto`][elfhub.events.Veto] to stop the action.

The available hooks are:

| Hook | Payload | Use it to |
|---|---|---|
| `before_model_call` | `ModelCall(request, ...)` | change the request; a veto fails the run |
| `before_tool_call` | `ToolInvocation(call, spec, ...)` | rewrite the call's arguments, or veto it (the model is told) |
| `after_tool_call` | `ToolOutcome(call, spec, result, ...)` | change or redact the result the model sees |

This rewrites the result of every tool before the model sees it:

```python
from dataclasses import replace

from elfhub import Elf, tool
from elfhub.hooks import ToolOutcome
from elfhub.testing import ScriptedLLM, call
from elfhub.tools import ToolResult


@tool
def read_secret() -> str:
    """Read the secret."""
    return "the password is hunter2"


def redact(outcome: ToolOutcome) -> ToolOutcome:
    return replace(outcome, result=ToolResult.text("[redacted]"))


model = ScriptedLLM(call("read_secret"), "I can't see it.")
elf = Elf(llm=model, tools=[read_secret])
elf.intercept("after_tool_call", redact)
elf.run("what is the secret?")

assert model.requests[1].messages[-1].tool_results[0].content[0].text == "[redacted]"
```

**Ordering:** both observers and interceptors run highest `priority` first, then in
the order they were registered.

**Errors:** a failing observer is logged and skipped. A failing interceptor fails the
run, because a broken safety check must never let an action through.

## The session log

The conversation is rebuilt from `message.added` events, so a session can always be
inspected, replayed, or resumed. Each `tool.completed` event also carries the call's
result. If a run is interrupted part-way through a batch of tool calls, `aresume` skips
the calls that already finished: an email is never sent twice. With a `home` folder, logs are JSON Lines files in
`.elf/sessions/<session_id>.jsonl`, one event per line.

Session stores are pluggable. Implement [`SessionStore`][elfhub.persist.SessionStore]
to keep logs anywhere else.
