"""The elf folder: the on-disk form of an elf that you edit, version, share and run.

::

    my-elf/
      elf.toml      configuration (see elfhub.config)
      SOUL.md       identity and personality
      AGENTS.md     operating rules
      tools/*.py    custom tools
      .elf/         local state (sessions...), never shared
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from elfhub.config import load_config
from elfhub.errors import ConfigError
from elfhub.tools.base import Tool, is_tool

CONFIG_FILE = "elf.toml"
SOUL_FILE = "SOUL.md"
INSTRUCTIONS_FILE = "AGENTS.md"
TOOLS_DIR = "tools"
STATE_DIR = ".elf"


def load_elf_kwargs(root: Path) -> dict[str, Any]:
    """Read an elf folder into keyword arguments for :class:`~elfhub.Elf`.

    Only what the folder actually declares is included, so anything it leaves out
    falls back to the elf class's own declarations.
    """
    from elfhub.plugins import tools as tool_registry

    config = load_config(root / CONFIG_FILE)
    kwargs: dict[str, Any] = {"name": config.name, "home": root, "config": config}
    if config.llm.main is not None:
        kwargs["llm"] = config.llm.main
    if config.llm.options:
        kwargs["llm_options"] = config.llm.options
    soul = _read_optional(root / SOUL_FILE)
    if soul is not None:
        kwargs["soul"] = soul
    instructions = _read_optional(root / INSTRUCTIONS_FILE)
    if instructions is not None:
        kwargs["instructions"] = instructions
    tools: list[Tool] = [tool_registry.create(name) for name in config.tools]
    tools += load_tool_files(root / TOOLS_DIR)
    if tools:
        kwargs["tools"] = tools
    if config.modules:
        kwargs["modules"] = list(config.modules)
        kwargs["module_config"] = config.modules
    if "policy" in config.model_fields_set:
        kwargs["policy"] = config.policy
    if "budget" in config.model_fields_set:
        kwargs["budget"] = config.budget
    return kwargs


def load_tool_files(directory: Path) -> list[Tool]:
    """Import every ``*.py`` file in ``directory`` and collect the tools it defines.

    This runs code from the elf folder: only load elves you trust.
    """
    if not directory.is_dir():
        return []
    found: dict[int, Tool] = {}
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        module_name = f"_elfhub_tools_{digest}_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"cannot load tool file {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            del sys.modules[module_name]
            raise ConfigError(
                f"tool file {path} failed to import: {type(exc).__name__}: {exc}"
            ) from exc
        for value in vars(module).values():
            if is_tool(value):
                found.setdefault(id(value), value)
    return list(found.values())


def scaffold(path: str | Path, *, name: str) -> Path:
    """Create a new elf folder. Refuses to touch a folder that already has files."""
    root = Path(path)
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"{root} exists and is not a folder")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} already exists and is not empty")
    files = {
        # A JSON string with non-ASCII kept literal is also a valid TOML basic string.
        CONFIG_FILE: _CONFIG_TEMPLATE.format(name=json.dumps(name, ensure_ascii=False)),
        SOUL_FILE: _SOUL_TEMPLATE.format(name=name),
        INSTRUCTIONS_FILE: _INSTRUCTIONS_TEMPLATE,
        f"{TOOLS_DIR}/word_count.py": _TOOL_TEMPLATE,
        ".gitignore": f"{STATE_DIR}/\n",
    }
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root


def _read_optional(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


_CONFIG_TEMPLATE = """\
format_version = 1
name = {name}
version = "0.1.0"
description = ""

# The model this elf thinks with, as "provider/model". Install an adapter package
# for the provider first; `elfhub run` lists the adapters it can find.
# llm = "provider/model"

[budget]
max_turns = 50

# Ask a human before tools that change things run:
# [[policy.rules]]
# tool = "*"
# destructive = true
# action = "ask"
"""

_SOUL_TEMPLATE = """\
# {name}

<!-- elf:lock -->
## Core values
- Be honest, including about what you don't know.
- Act in your owner's interest and ask when unsure.
<!-- /elf:lock -->

## Personality
You are {name}. Describe who this elf is, how it talks and what it cares about.
"""

_INSTRUCTIONS_TEMPLATE = """\
# Operating rules

- Think before acting, and check your work.
- Prefer tools over guessing when a tool can answer.
"""

_TOOL_TEMPLATE = '''\
from elfhub.tools import tool


@tool(read_only=True, open_world=False)
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())
'''
