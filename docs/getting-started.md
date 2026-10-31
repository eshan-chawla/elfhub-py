# Getting started

## Create an elf

```bash
elfhub new my-elf
```

This creates a folder that holds everything about the elf:

```text
my-elf/
  elf.toml          configuration: model, modules, policies, budget
  SOUL.md           who the elf is (identity, personality, values)
  AGENTS.md         how the elf should work (operating rules)
  tools/            custom tools, one Python file each
  .gitignore        keeps .elf/ (local state such as session logs) out of git
```

## Give it a model

Set `llm` in `elf.toml` to a `"provider/model"` reference:

```toml
llm = "acme/large-model"
```

ElfHub looks the provider up among the installed LLM adapter plugins. Official
adapters are planned for Phase 5 of the [roadmap](roadmap.md). Until then, writing
an adapter for your provider takes a few dozen lines; see
[Writing plugins](plugins/writing-plugins.md).

## Run it

```bash
elfhub run my-elf "Summarise what you can do"
```

Useful options:

| Option | What it does |
|---|---|
| `--session ID` | continue an earlier conversation |
| `--yes` | approve every tool call that a policy says needs approval |
| `--json` | print the full run result as JSON |

The exit code is `0` when the run completes, `3` when it is waiting for an approval,
and `1` for anything else.

When a [policy](concepts/approvals.md) asks for approval and you are at a terminal,
`elfhub run` asks you. Otherwise the call is refused and the elf is told why.

## From Python

Load an elf folder:

<!-- skip-test -->
```python
from elfhub import Elf

elf = Elf.from_dir("my-elf")
print(elf.run("hello").output)
```

Or build one in code. Here `ScriptedLLM` stands in for a real model:

```python
from elfhub import Elf
from elfhub.testing import ScriptedLLM

elf = Elf(
    name="scribe",
    llm=ScriptedLLM("Hello! I take notes."),
    soul="You are Scribe. You take careful notes.",
    instructions="Answer in one sentence.",
)

result = elf.run("Who are you?")
assert result.status == "completed"
print(result.output)
```

Next, read about [elves and runs](concepts/elves.md), then [tools](concepts/tools.md).
