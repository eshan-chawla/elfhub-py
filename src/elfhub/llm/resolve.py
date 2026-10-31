"""Turn a ``"provider/model"`` declaration into an LLM adapter."""

from __future__ import annotations

from typing import Any

from elfhub.errors import ConfigError, PluginNotFoundError
from elfhub.llm.protocol import LLM
from elfhub.plugins.registry import Registry


def parse_ref(ref: str) -> tuple[str, str]:
    """Split ``"provider/model"``. The model id may itself contain slashes."""
    provider, _, model = ref.partition("/")
    if not provider or not model:
        raise ConfigError(f"model reference {ref!r} must look like 'provider/model'")
    return provider, model


def resolve_llm(ref: str | LLM, *, registry: Registry[Any] | None = None, **options: Any) -> LLM:
    """Return ``ref`` unchanged if it is already an LLM, else build its adapter.

    Adapters are looked up by provider name in the ``elfhub.llms`` registry and
    called as ``factory(model=<model id>, **options)``.
    """
    if not isinstance(ref, str):
        return ref
    provider, model = parse_ref(ref)
    if registry is None:
        from elfhub.plugins import llms as registry
    if provider not in registry:
        installed = ", ".join(sorted(registry.names())) or "none"
        raise PluginNotFoundError(
            f"no LLM adapter for provider {provider!r} (installed: {installed}). "
            f"Install a package that provides a {provider!r} adapter with `pip install <package>`; "
            "the ElfHub docs list known adapters."
        )
    return registry.create(provider, model=model, **options)
