---
type: operations
title: Packaging and CI
description: How elfhub-py is built with hatchling, checked in CI, and published to PyPI by version tag pushes.
tags: [packaging, ci, publishing, pypi, elfhub]
verified:
  - by: openwiki/0.6.1
    at: 2026-10-17T22:47:17.598Z
sources:
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
generated: { by: "opencode", at: "2026-10-17T22:47:17.598Z" }
---

The repository ships a single-distribution library (`elfhub`) with an automated CI-and-publish pipeline driven entirely by GitHub Actions.

## Packaging layout (pyproject.toml)

- Build backend is `hatchling`; the wheel build target packages only `src/elfhub` (`[tool.hatch.build.targets.wheel] packages = ["src/elfhub"]`), giving the repo a `src`-layout.
- The project is declared as `elfhub` v0.1.0, an MIT-licensed library requiring Python >= 3.9, with `dependencies = []` — no runtime dependencies.
- Dev tooling lives in optional extras: `dev = ["pytest", "build", "twine"]`.
- URLs point at the GitHub repository `eshan-chawla/elfhub-py`.

## CI workflow (.github/workflows/ci.yml)

- Triggers: pushes to `main`, any push of a `v*` tag, and pull requests.
- The `test` job runs a matrix over Python 3.9-3.13, installs the package with `python -m pip install .[dev]`, then runs the smoke test `python demo/demo.py`. A pytest suite (`tests/`) previously filled this slot; it was replaced by the demo script, whose job creates its own venv, installs the *published* `elfhub` from PyPI, and exercises `Elf()` — see the agent-class page.

## Publish flow

- The `publish` job runs only when the ref is a `v*` tag (`if: startsWith(github.ref, 'refs/tags/v')`) and only after `test` succeeds (`needs: test`) on Python 3.12.
- It builds with `python -m build` and publishes using `pypa/gh-action-pypi-publish@release/v1` with `id-token: write` permission, i.e. PyPI trusted publishing via OIDC — no API token in the workflow.
- Consequence: cutting a release is just creating an `v*` tag; version bumps must land in `pyproject.toml` (version 0.1.0) before tagging, since nothing syncs that version automatically.

## Failure semantics

- A failed test job blocks publishing for that tag; the release path never bypasses CI.
