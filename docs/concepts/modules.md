# Modules

Modules give an elf new abilities. A module can add tools, pieces of the system
prompt, approval policies and event hooks. Future features such as the
self-updating soul, memory, skills, workspaces and coding agents will all be
modules, built on the same API that is available to you.

## Writing a module

```python
from typing import ClassVar

from pydantic import BaseModel

from elfhub import Elf, ElfContext, Module
from elfhub.testing import ScriptedLLM


class TimezoneConfig(BaseModel):
    zone: str = "UTC"


class Timezone(Module[TimezoneConfig]):
    name: ClassVar[str] = "timezone"

    def setup(self, ctx: ElfContext) -> None:
        zone = self.config.zone
        ctx.add_context(lambda _run: f"The user's timezone is {zone}.")


model = ScriptedLLM("Noted.")
Elf(llm=model, modules=[Timezone(zone="Europe/Paris")]).run("hi")

print(model.requests[0].system[0].text)  # The user's timezone is Europe/Paris.
```

- `name` identifies the module. It is how `elf.toml` and `requires` refer to it.
- The config is a pydantic model, given as the type argument (`Module[TimezoneConfig]`)
  so `self.config` is fully typed. The constructor validates its keyword arguments,
  or a mapping, against it. Invalid config raises
  [`ConfigError`][elfhub.errors.ConfigError] naming the module. A nested
  `class Config(BaseModel)` also works, but leaves `self.config` untyped.
- `requires` lists modules that must also be enabled. They are set up and started
  first.

## The elf context

A module never receives the `Elf` object itself. It gets an
[`ElfContext`][elfhub.ElfContext] with only what modules need, so the elf's
internals can change without breaking modules:

| Method | What it does |
|---|---|
| `add_tool(tool)` | give the elf a tool (or plain function) |
| `add_context(provider)` | add a piece of the system prompt; `provider(run_context)` returns text or `None` |
| `add_policy(policy)` | add an approval policy layer (layers can only restrict) |
| `on(event_type, fn)` | observe events |
| `intercept(hook, fn)` | rewrite or veto at a hook, such as `before_tool_call` |
| `llm(role)` | the elf's model |
| `name`, `home`, `store` | the elf's name, folder and session store |

Hooks a module registers are removed when the elf stops, so restarting never
duplicates them.

## Lifecycle

Each time an elf starts:

1. **`setup(ctx)`** runs for every module, dependencies first.
2. **`await start()`** runs for every module, in the same order.

When the elf stops, **`await stop()`** runs in reverse order, even if the run failed.
If a module fails to start, the modules already started are stopped again.

## Enabling modules by name

Installed module plugins are enabled in `elf.toml`, each with its config table:

```toml
[modules.timezone]
zone = "Europe/Paris"
```

The same works in Python with `Elf(modules=["timezone"], module_config={"timezone":
{"zone": "Europe/Paris"}})`. See [Writing plugins](../plugins/writing-plugins.md)
for how to publish a module.
