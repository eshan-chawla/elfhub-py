---
type: architecture
title: The Elf agent class
description: How the `Elf` base agent class defines the invoke contract that all ElfHub agents share and how to subclass it.
tags: [agents, architecture, elfhub]
verified:
  - by: openwiki/0.6.1
    at: 2026-10-17T22:47:17.598Z
sources:
  - id: openwiki-source-93e81318eca0e930c9cb2e3b
    resource: repo://src/elfhub/__init__.py
  - id: openwiki-source-556b637139fef4f91392f644
    resource: repo://src/elfhub/agent.py
generated: { by: "opencode", at: "2026-10-17T22:47:17.598Z" }
---

`Elf` (src/elfhub/agent.py) is the foundation of ElfHub's claim to be "a minimal agent framework base class": every agent in the framework derives from it.

## Entry points

- src/elfhub/agent.py defines `class Elf` with two behaviors:
  - `invoke(*args, **kwargs)` returns the literal string `"Hi !"`. The docstring ("Respond to an invocation") states its contract, but the shipped base implementation ignores all input and always answers with the greeting.
  - `__call__(*args, **kwargs)` forwards to `self.invoke(...)`, so an `Elf` instance is directly callable: `Elf()() == "Hi !"`.
- src/elfhub/__init__.py imports `Elf` from `elfhub.agent` and re-exports it as the package's sole public symbol: `__all__ == ["Elf"]`, alongside `__version__ = "0.1.0"`.

## The agent contract

The framework's core mechanism is call-time polymorphism: callers invoke an agent by calling it, and the actual behavior lives in `invoke`. Since `__call__` delegates, a subclass only needs to override `invoke`, not `__call__`, to change what the agent does:

```python
from elfhub import Elf


class GreetElf(Elf):
    def invoke(self, name="world"):
        return f"Hello, {name}!"
```

Instances of `GreetElf` remain drop-in callable the same way `Elf` is, because the inherited `__call__` keeps routing through the overridden `invoke`.

## Invariants and seams

- The base `invoke` takes `*args, **kwargs` and rejects nothing; downstream behavior is entirely defined by overrides.
- The package surface is deliberately restricted to `Elf`; nothing else is re-exported, so subclassing is the intended extension seam rather than new top-level exports.
- There is no state, persistence, async, or error handling in the shipped class; lifecycle questions fall to agent authors.
