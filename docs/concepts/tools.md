# Tools

Tools are the actions an elf can take. The simplest way to make one is the
[`@tool`][elfhub.tools.tool] decorator:

```python
from typing import Annotated

from pydantic import Field

from elfhub import tool


@tool(read_only=True, open_world=False)
def word_count(text: Annotated[str, Field(description="The text to count")]) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


print(word_count.spec.name)  # word_count
print(word_count.spec.input_schema)  # JSON Schema built from the type hints
```

- **The name** is the function name, unless you pass `name=`.
- **The description** is the docstring, unless you pass `description=`.
- **The arguments** are validated against the type hints before the function runs.
  If the model sends bad arguments, it gets an error result explaining what was
  wrong and can try again.
- **The function can be sync or async.** Sync functions run in a worker thread, so
  they never block the elf.
- **The return value** is passed back as text: strings as they are, anything else
  as JSON. Return a [`ToolResult`][elfhub.tools.ToolResult] for full control.

Plain functions passed to `Elf(tools=[...])` are wrapped automatically.

## Annotations

Annotations tell ElfHub how a tool behaves. Approval [policies](approvals.md) match on
them. They mirror MCP's tool annotations and use the same cautious defaults:

| Annotation | Default | Meaning |
|---|---|---|
| `read_only` | `False` | the tool changes nothing |
| `destructive` | `not read_only` | the tool may destroy or overwrite things |
| `idempotent` | `False` | calling it twice has the same effect as once |
| `open_world` | `True` | the tool touches the outside world; its output is treated as untrusted |
| `parallel_safe` | `False` | the tool can run at the same time as other calls in the same turn |
| `timeout_s` | `120` | how long a call may take before it fails |

When every call in a turn is `parallel_safe`, the calls run concurrently. Otherwise
they run one after another, in the order the model asked for them.

## Failures

A tool that raises, times out, or is refused never crashes the run. The model gets an
error result and decides what to do next.

## The tool context

Add a parameter annotated as [`ToolContext`][elfhub.tools.ToolContext] to get
information about the current call. It is filled in automatically and hidden from
the model:

```python
from elfhub import tool
from elfhub.tools import ToolContext


@tool
def where_am_i(ctx: ToolContext) -> str:
    """Report the current session."""
    return f"session {ctx.session_id}, call {ctx.call_id}"
```

`ctx.emit(event)` records a custom [event](events.md) in the session log, for
example to report progress.

## Tool classes

For tools that hold state, such as a client or credentials, write a class. A tool
is any object with a `spec` and an async `__call__`:

```python
from typing import Any

from elfhub import Elf
from elfhub.testing import ScriptedLLM, call
from elfhub.tools import ToolContext, ToolResult, ToolSpec


class Search:
    spec = ToolSpec(
        name="search",
        description="Search the knowledge base.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    def __init__(self, index: dict[str, str]) -> None:
        self.index = index

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.text(self.index.get(args["query"], "nothing found"))


model = ScriptedLLM(call("search", query="refunds"), "Refunds take 5 days.")
Elf(llm=model, tools=[Search({"refunds": "5 business days"})]).run("How long do refunds take?")
```

Pass an instance, not the class. Passing the class raises an error explaining this.

## Tools in an elf folder

Every `tools/*.py` file in an elf folder is imported when the elf loads, and every
tool object it defines or imports is enabled. Tools from installed plugins are enabled by name in
`elf.toml`:

```toml
tools = ["web_search"]
```

!!! danger "Tool files are code"
    Loading an elf folder runs its `tools/*.py` files. Only load elves you trust.
