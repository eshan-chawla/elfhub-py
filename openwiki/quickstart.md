---
type: quickstart
title: Quickstart
description: What the elfhub-py repository contains, how to install and use the published package, and where each wiki page lives.
tags: [quickstart, elfhub, onboarding]
verified:
  - by: openwiki/0.6.1
    at: 2026-10-17T22:47:17.598Z
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-93e81318eca0e930c9cb2e3b
    resource: repo://src/elfhub/__init__.py
generated: { by: "opencode", at: "2026-10-17T22:47:17.598Z" }
---

ElfHub (`elfhub` on PyPI, v0.1.0) is a minimal agent framework: its entire public surface is one base class, `Elf`, which any custom agent subclasses.

## What the repository contains

- `src/elfhub/` — the package itself: `agent.py` (the `Elf` class) and `__init__.py` (exports). This is the only code.
- `pyproject.toml` — hatchling packaging config, metadata, and the `dev` extra (`pytest`, `build`, `twine`).
- `.github/workflows/ci.yml` — the Python 3.9-13 test matrix and the tag-triggered PyPI publish job.
- `README.md`, `LICENSE`, and this `openwiki/` index.

## Install and use

```bash
pip install elfhub
```

```python
from elfhub import Elf

agent = Elf()
print(agent())  # "Hi !", because __call__ routes to invoke
```

## Build your own agent

Subclass `Elf` and override `invoke`; instances stay directly callable because the inherited `__call__` keeps delegating.

Reference: The Elf agent class.

## Ship a release

Commit any version bump in `pyproject.toml`, then push a `v*` tag — CI runs the smoke test matrix and, on success, publishes to PyPI via trusted publishing. Details: Packaging and CI.

Reference: Packaging and CI.
