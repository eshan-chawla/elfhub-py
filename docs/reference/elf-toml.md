# elf.toml

`elf.toml` is an elf's configuration. `elfhub schema` prints its full JSON Schema,
which editors and other tools can use for validation and autocompletion.

```toml
format_version = 1
name = "ada"
version = "0.1.0"
description = "A careful research assistant"

llm = "acme/large-model"            # or a table: see below
tools = ["web_search"]              # tools from installed plugins, by name

[modules.timezone]                  # a module from an installed plugin, with its config
zone = "Europe/Paris"

[[policy.rules]]                    # approval rules (see Concepts → Approvals)
tool = "*"
destructive = true
action = "ask"

[budget]
max_turns = 30
max_cost_usd = 2.0

[extensions."elfhub-cloud"]         # settings owned by other programs, kept as-is
```

| Key | Type | Meaning |
|---|---|---|
| `format_version` | int | the file format; an elfhub too old to read it refuses to load the elf |
| `name` | string | the elf's name (required) |
| `version` | string | the elf's own version |
| `description` | string | one line about the elf |
| `llm` | string or table | `"provider/model"`, or `{ main = "provider/model", options = { ... } }`; `options` are passed to the adapter |
| `tools` | list of strings | plugin tools to enable; `tools/*.py` files load automatically |
| `modules.<name>` | table | enables a module plugin; the table is its config |
| `policy.rules` | list of tables | `tool` (glob), optional `read_only` / `destructive` / `open_world`, and `action` (`allow`, `ask`, `deny`) |
| `policy.default` | string | the action when no rule matches (default `allow`) |
| `budget` | table | `max_turns`, `max_tool_calls`, `max_tokens`, `max_cost_usd`, `max_wall_s` |
| `extensions.<owner>` | table | settings for other programs, such as a hosted runtime |

## Unknown keys

ElfHub handles keys it doesn't know by one rule:

- **In `[budget]` and `[policy]` they are errors.** Ignoring a misspelt limit or rule
  would silently loosen it. For example, `tol = "safe_tool"` with `action = "allow"`
  would become "allow everything".
- **Anywhere else they are warnings.** The warning includes a "did you mean" hint, and
  the key is ignored, so an elf written for a newer ElfHub still loads.
- **`[extensions.<owner>]` is kept as-is,** for settings that belong to other
  programs, such as a hosted runtime.

ElfHub never rewrites `elf.toml` itself, so nothing in the file is ever lost.

Alongside `elf.toml`, an elf folder holds `SOUL.md` (identity), `AGENTS.md`
(operating rules), `tools/*.py` (custom tools) and `.elf/` (local state, never
shared).
