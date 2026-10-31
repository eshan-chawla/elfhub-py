# Roadmap

ElfHub is built in phases. Each phase ends with tests that prove it works.

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation: packaging, CI on Python 3.11–3.14, the typed plugin registry | done |
| 1 | Core: the LLM contract and conformance checks, tools, modules, events, approvals, budgets, session logs, `elf.toml`, the `elfhub` command | done |
| 2 | Identity: a `SOUL.md` the elf updates itself (locked sections, version history, rollback), memory, post-run reflection, trust tracking | next |
| 3 | Skills: agentskills.io `SKILL.md` skills the elf writes itself, MCP skill sources | planned |
| 4 | Environment: local and Docker runtimes, a confined workspace, secrets, computer use | planned |
| 5 | Inference and coding: official LLM adapters, delegating to coding agents (opencode, Claude Code, Codex) | planned |
| 6 | Always-on and portability: scheduled wake-ups, channels, `elfhub serve` (HTTP session API), portable bundles, publishing elves, the managed-runtime contract, 1.0 | planned |

## Stability

Until 1.0, the plugin API and file formats may change. Each carries a version so
changes can be detected:

- the module API: `Module.api_version`;
- `elf.toml`: `format_version`;
- events: the `v` field on every event.

A plugin built for a different module API version is refused with a clear error
instead of misbehaving.
