"""Plugin registries.

Every pluggable kind has one registry, backed by a Python entry-point group, so a
third-party package can add plugins just by declaring entry points::

    [project.entry-points."elfhub.llms"]
    acme = "acme_elfhub:AcmeLLM"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elfhub.plugins.registry import Registry
from elfhub.tools.base import Tool, is_tool

if TYPE_CHECKING:
    from elfhub.llm.protocol import LLM
    from elfhub.plugins.module import Module


def _tool_factory(obj: Any) -> Any:
    # Tools are stateless, so an entry point may point straight at a @tool function.
    return (lambda: obj) if is_tool(obj) else obj


llms: Registry[LLM] = Registry("llm", group="elfhub.llms")
modules: Registry[Module] = Registry("module", group="elfhub.modules")
tools: Registry[Tool] = Registry("tool", group="elfhub.tools", coerce=_tool_factory)

__all__ = ["Registry", "llms", "modules", "tools"]
