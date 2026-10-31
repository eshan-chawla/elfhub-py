# ElfHub

ElfHub is an open-source Python framework for building **elves**: agents that work
like an individual person. An elf has its own model, identity, tools and modules.
It can run on your machine today, and its design lets it move to containers and a
managed runtime later.

!!! warning "Alpha"
    ElfHub is in early development. The core framework (Phases 0–1 of the
    [roadmap](roadmap.md)) works: elves, tools, modules, approvals, budgets,
    sessions, plugins and the `elfhub` command. Self-updating souls, skills,
    workspaces, LLM adapters and coding agents are on the roadmap. APIs may change
    before 1.0.

## Install

```bash
pip install elfhub
```

ElfHub needs Python 3.11 or newer.

## A first elf

This runs offline: `ScriptedLLM` plays the model's part, which is also how you test
elves.

```python
from elfhub import Elf, tool
from elfhub.testing import ScriptedLLM, call


@tool(read_only=True, open_world=False)
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


model = ScriptedLLM(call("add", a=2, b=3), "2 + 3 = 5")
elf = Elf(name="ada", llm=model, soul="You are Ada, a careful assistant.", tools=[add])

result = elf.run("What is 2 + 3?")
print(result.output)  # 2 + 3 = 5
```

With a real model you declare it as `"provider/model"`, and ElfHub finds the adapter
for that provider among installed plugins:

<!-- skip-test -->
```python
elf = Elf(name="ada", llm="anthropic/claude-sonnet-5-5", tools=[add])
```

## Design ideas

- **An elf is a folder.** `elf.toml`, `SOUL.md`, `AGENTS.md` and `tools/` describe it,
  so it can be versioned, shared, and moved between machines and runtimes.
- **The LLM layer is a contract, not a vendor SDK.** Any provider can be supported
  through an adapter, and every adapter is checked by the same conformance suite.
- **Everything optional is a plugin.** LLM adapters, modules and tools are found
  through Python entry points, so you extend ElfHub by installing packages, not
  by forking it.
- **Safe by default.** Tool calls go through approval policies, runs have budgets,
  and every step is recorded in a session log you can inspect, replay and resume.
