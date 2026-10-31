# Writing plugins

ElfHub discovers plugins through Python
[entry points](https://packaging.python.org/en/latest/specifications/entry-points/).
To add a plugin, publish a package that declares one; ElfHub needs no changes.

| Entry-point group | Plugin kind | Points at |
|---|---|---|
| `elfhub.llms` | LLM adapter | a class or factory called as `factory(model=..., **options)` |
| `elfhub.modules` | module | a `Module` subclass |
| `elfhub.tools` | tool | a `@tool` function, a tool class, or a factory returning a tool |

```toml
# pyproject.toml of your plugin package
[project.entry-points."elfhub.llms"]
acme = "acme_elfhub:AcmeLLM"

[project.entry-points."elfhub.modules"]
timezone = "acme_elfhub:Timezone"

[project.entry-points."elfhub.tools"]
web_search = "acme_elfhub:web_search"
```

**When plugins load:** entry points are listed lazily and imported only when used,
so installing many plugins costs nothing until an elf needs one.

**Name clashes:** if two packages claim the same name, ElfHub refuses to guess and
raises an error naming both.

## LLM adapters

An adapter translates between ElfHub's provider-neutral types and one vendor's API.
Subclass [`BaseLLM`][elfhub.llm.BaseLLM] and implement `stream`. You get `generate`
for free:

```python
from collections.abc import AsyncIterator
from typing import Any

from elfhub.llm import BaseLLM, ModelEvent, ModelRequest, PartDone, Stop, TextDelta, UsageReport
from elfhub.types import TextPart, Usage


class ShoutLLM(BaseLLM):
    """A toy adapter that shouts the last message back."""

    def __init__(self, model: str, **options: Any) -> None:
        self.ref = f"shout/{model}"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        text = request.messages[-1].text.upper() or "PONG"
        yield TextDelta(text=text)
        yield PartDone(part=TextPart(text=text))
        yield UsageReport(usage=Usage(input_tokens=1, output_tokens=1))
        yield Stop(reason="end")
```

`capabilities` describes what the model can do. It is frozen, so set it as a whole,
for example `capabilities = ModelCapabilities(vision=True)`.

The streaming contract:

1. `TextDelta` and `ReasoningDelta` events are optional and only for live display.
   If you send text deltas, they must add up to the finished text.
2. Every finished part (text, tool call, reasoning) is sent exactly once as
   `PartDone`.
3. `UsageReport` events may come any number of times; they are added together.
4. The stream ends with exactly one `Stop`. Its reason is `end`, `tool_use`,
   `max_tokens` or `refusal`.

A few more rules:

- **Tool call ids:** they must be unique across the whole conversation, not just
  within one reply. Approvals and results are matched by id. If the provider sends
  no ids, generate them, for example with `elfhub.types.new_id("call")`.
- **Errors:** map vendor errors onto `RateLimited`, `TransientModelError`,
  `ContextOverflow`, `AuthFailed` and `Refused` from `elfhub.errors`. The loop
  retries rate limits and transient errors.
- **Reasoning:** put vendor reasoning data that must be sent back (such as
  signatures) in `ReasoningPart.opaque`.
- **Vendor-specific settings:** read them from
  `request.provider_options["<your provider>"]`.

### Checking an adapter

Run the conformance checks against the real model in your test suite:

```python
import asyncio

from elfhub.testing import check_llm

asyncio.run(check_llm(ShoutLLM("loud"), tools=False))
```

`check_llm` runs every check that fits the adapter's declared capabilities:

| Check | What it proves |
|---|---|
| `check_stream` | one final `Stop`, and text deltas that add up to the final text |
| `check_text_reply` | a plain question gets a text answer |
| `check_usage_reported` | token usage is reported, which budgets need (skip with `usage=False`) |
| `check_tool_round_trip` | tool calls with parsed arguments, answers from tool results, ids unique across turns |
| `check_parallel_tool_calls` | several calls in one reply (when `parallel_tools` is declared) |
| `check_reasoning_round_trip` | reasoning parts can be sent back, and another provider's are dropped (when `reasoning` is declared) |

Two error-mapping checks are opt-in, because they have to provoke a failure:

- `check_auth_error(factory)` takes a function that builds the adapter with bad
  credentials.
- `check_context_overflow(llm)` sends a request larger than
  `capabilities.context_window`.

## Testing elves

`ScriptedLLM` replays a script of model turns and records every request, so you can
test an elf without a network:

```python
from elfhub import Elf, tool
from elfhub.testing import ScriptedLLM, call


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


model = ScriptedLLM(call("add", a=2, b=2), "4")
result = Elf(llm=model, tools=[add]).run("2+2?")

assert result.output == "4"
assert model.requests[1].messages[-1].tool_results[0].content[0].text == "4"
```

A script step can be any of these:

- reply text;
- a `call(...)`;
- a list of parts;
- a `Message`;
- an exception to raise;
- a function of the request that returns one of the above.
