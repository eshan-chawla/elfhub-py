# ElfHub

An open-source Python framework for building **elves**: agents that work like an
individual person, each with its own model, identity, tools and modules.

> **Alpha.** The core framework works: elves, tools, modules, approvals, budgets,
> session logs, plugins and the `elfhub` command. Self-updating souls, skills,
> workspaces, LLM adapters and coding agents are on the [roadmap](docs/roadmap.md).
> APIs may change before 1.0.

## Install

```bash
pip install elfhub
```

Requires Python 3.11+.

## Example

```python
from elfhub import Elf, tool
from elfhub.testing import ScriptedLLM, call


@tool(read_only=True, open_world=False)
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ScriptedLLM stands in for a real model, so this runs offline.
model = ScriptedLLM(call("add", a=2, b=3), "2 + 3 = 5")
elf = Elf(name="ada", llm=model, soul="You are Ada, a careful assistant.", tools=[add])

print(elf.run("What is 2 + 3?").output)  # 2 + 3 = 5
```

With a real model, declare it as `llm="provider/model"`. ElfHub finds the adapter for
that provider among installed plugins.

Or start from a folder:

```bash
elfhub new my-elf       # elf.toml, SOUL.md, AGENTS.md, tools/
elfhub run my-elf "hello"
```

## Documentation

Start with [Getting started](docs/getting-started.md). The full docs, including a
generated API reference, live in [`docs/`](docs/). Preview them with `zensical serve`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
pip install ./tests/fixtures/plugin_pkg   # a stand-in plugin used by the tests
pytest && ruff check . && pyright && zensical build --strict
```

## License

MIT
