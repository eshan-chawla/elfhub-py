"""Modules: the main way to give an elf new abilities.

A module is set up with an :class:`ElfContext`, a narrow facade through which it can
add tools, system-prompt context, policies and event hooks. Modules never receive
the :class:`~elfhub.Elf` itself, so the elf's internals can change without breaking
them. Soul, memory, skills, workspaces, coding agents... are all modules.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic, cast, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import TypeVar

from elfhub.errors import ConfigError, PluginError
from elfhub.events import EventBus, Interceptor, Observer
from elfhub.llm.protocol import LLM
from elfhub.loop.context import ContextProvider
from elfhub.persist.store import SessionStore
from elfhub.policy.approvals import Policy
from elfhub.tools.base import Tool
from elfhub.tools.decorator import as_tool

API_VERSION = 1
"""The module plugin API this version of ElfHub implements."""


class NoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


ConfigT = TypeVar("ConfigT", bound=BaseModel, default=NoConfig)


class Module(Generic[ConfigT]):
    """Base class for modules.

    Subclasses set ``name``, optionally ``requires`` (names of modules that must be
    enabled and are started first), and declare their config as a pydantic model,
    preferably as a type argument so ``self.config`` is typed::

        class Timezone(Module[TimezoneConfig]):
            name = "timezone"

    The config is validated on construction and available as ``self.config``.
    """

    name: ClassVar[str]
    api_version: ClassVar[int] = API_VERSION
    Config: ClassVar[type[BaseModel]] = NoConfig
    requires: ClassVar[tuple[str, ...]] = ()
    config: ConfigT

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "Config" in cls.__dict__:
            return
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Module:
                [argument] = get_args(base)
                if isinstance(argument, type) and issubclass(argument, BaseModel):
                    cls.Config = argument
                return

    def __init__(self, config: Mapping[str, Any] | None = None, /, **options: Any) -> None:
        try:
            validated = self.Config.model_validate({**(config or {}), **options})
        except ValidationError as exc:
            raise ConfigError(f"invalid config for module {module_name(self)!r}: {exc}") from exc
        self.config = cast(ConfigT, validated)

    def setup(self, ctx: ElfContext) -> None:
        """Register tools, context, policies and hooks. Called on every elf start."""

    async def start(self) -> None:
        """Acquire resources. Dependencies are started first."""

    async def stop(self) -> None:
        """Release resources. Called in reverse start order, even after failures."""


def module_name(module: Module[Any]) -> str:
    return getattr(type(module), "name", type(module).__name__)


def check_api_version(cls: type[Any]) -> None:
    """Refuse a module class built for another version of the module plugin API."""
    version = getattr(cls, "api_version", API_VERSION)
    if version != API_VERSION:
        name = getattr(cls, "name", cls.__name__)
        raise PluginError(
            f"module {name!r} targets plugin API v{version}, "
            f"but this version of elfhub provides v{API_VERSION}"
        )


@dataclass
class Assembly:
    """What modules contribute while an elf starts."""

    tools: list[Tool] = field(default_factory=list[Tool])
    context: list[ContextProvider] = field(default_factory=list[ContextProvider])
    policies: list[Policy] = field(default_factory=list[Policy])


class ElfContext:
    """The view of an elf that a module gets during :meth:`Module.setup`."""

    def __init__(
        self,
        *,
        name: str,
        home: Path | None,
        store: SessionStore,
        bus: EventBus,
        llms: Mapping[str, LLM],
        assembly: Assembly,
    ) -> None:
        self.name = name
        self.home = home
        self.store = store
        self._bus = bus
        self._llms = dict(llms)
        self._assembly = assembly
        self._disposers: list[Callable[[], None]] = []

    def llm(self, role: str = "main") -> LLM:
        """The elf's model for ``role`` (e.g. ``"reflect"``), falling back to ``"main"``."""
        return self._llms.get(role) or self._llms["main"]

    def add_tool(self, tool: Tool | Callable[..., Any]) -> None:
        self._assembly.tools.append(as_tool(tool))

    def add_context(self, provider: ContextProvider) -> None:
        self._assembly.context.append(provider)

    def add_policy(self, policy: Policy) -> None:
        """Add a policy layer. Layers can only make the elf more restricted."""
        self._assembly.policies.append(policy)

    def on(self, event_type: str, fn: Observer, *, priority: int = 0) -> Callable[[], None]:
        return self._keep(self._bus.on(event_type, fn, priority=priority))

    def intercept(self, hook: str, fn: Interceptor, *, priority: int = 0) -> Callable[[], None]:
        return self._keep(self._bus.intercept(hook, fn, priority=priority))

    def dispose(self) -> None:
        """Remove every hook this module registered. Called when the elf stops."""
        for remove in self._disposers:
            remove()
        self._disposers.clear()

    def _keep(self, remove: Callable[[], None]) -> Callable[[], None]:
        self._disposers.append(remove)
        return remove


def order_modules(modules: Sequence[Module]) -> list[Module]:
    """Order modules so each comes after everything it requires (stable otherwise)."""
    by_name: dict[str, Module] = {}
    for module in modules:
        name = module_name(module)
        if name in by_name:
            raise ConfigError(f"module {name!r} is enabled twice")
        by_name[name] = module

    ordered: list[Module] = []
    state: dict[str, str] = {}

    def visit(module: Module, chain: list[str]) -> None:
        name = module_name(module)
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise ConfigError(
                f"modules require each other in a cycle: {' -> '.join([*chain, name])}"
            )
        state[name] = "visiting"
        for dependency in type(module).requires:
            if dependency not in by_name:
                raise ConfigError(f"module {name!r} requires {dependency!r}, which is not enabled")
            visit(by_name[dependency], [*chain, name])
        state[name] = "done"
        ordered.append(module)

    for module in modules:
        visit(module, [])
    return ordered
