"""Typed, entry-point backed registries for ElfHub plugins.

A :class:`Registry` maps names to *factories* (classes or functions) for one kind
of plugin. Third-party packages contribute plugins through a Python entry-point
group; those are listed lazily and only imported when first used.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Generic, TypeVar, overload

from elfhub.errors import DuplicatePluginError, PluginError, PluginNotFoundError

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _is_factory(obj: object) -> bool:
    return inspect.isclass(obj) or inspect.isroutine(obj) or isinstance(obj, functools.partial)


def _origin(obj: object) -> str:
    module = getattr(obj, "__module__", None) or "?"
    qualname = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return f"{module}:{qualname}"


def _entry_point_origin(ep: EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    return f"{ep.value} (from {dist.name if dist is not None else 'an unknown distribution'})"


class Registry(Generic[T]):
    """Name -> factory registry for one kind of plugin.

    Only classes and factory functions can be registered, never instances, so every
    :meth:`create` call returns a fresh, independently configured object. A kind whose
    objects are stateless (like tools) can pass ``coerce`` to turn such an object into
    a factory before it is checked.
    """

    def __init__(
        self,
        kind: str,
        group: str | None = None,
        *,
        coerce: Callable[[Any], Any] | None = None,
    ) -> None:
        self.kind = kind
        self.group = group
        self._coerce = coerce
        self._factories: dict[str, Callable[..., T]] = {}
        self._entry_points: dict[str, EntryPoint] = {}
        self._conflicts: dict[str, str] = {}
        self._discovered = group is None

    @overload
    def register(self, obj: F, /, *, name: str | None = None, replace: bool = False) -> F: ...

    @overload
    def register(
        self, obj: None = None, /, *, name: str | None = None, replace: bool = False
    ) -> Callable[[F], F]: ...

    def register(
        self,
        obj: Callable[..., Any] | None = None,
        /,
        *,
        name: str | None = None,
        replace: bool = False,
    ) -> Any:
        """Register a class or factory, directly or as a decorator.

        The name comes from ``name=``, else from the object's ``name`` attribute.
        """
        if obj is None:

            def decorator(target: F) -> F:
                return self.register(target, name=name, replace=replace)

            return decorator

        factory = self._coerce(obj) if self._coerce is not None else obj
        if not _is_factory(factory):
            raise TypeError(
                f"register a class or factory function as a {self.kind} plugin, "
                f"not an instance of {type(obj).__name__}"
            )
        resolved = name or getattr(obj, "name", None)
        if not isinstance(resolved, str) or not resolved:
            raise PluginError(
                f"{self.kind} plugin {_origin(obj)} has no name: "
                "pass name=... or give it a string `name` attribute"
            )

        self._discover()
        if replace:
            self._entry_points.pop(resolved, None)
            self._conflicts.pop(resolved, None)
        elif resolved in self._conflicts:
            raise DuplicatePluginError(self._conflicts[resolved])
        elif resolved in self._factories:
            raise DuplicatePluginError(
                f"{self.kind} plugin {resolved!r} is already registered by "
                f"{_origin(self._factories[resolved])}; cannot also register {_origin(obj)} "
                "(pass replace=True to override)"
            )
        elif resolved in self._entry_points:
            raise DuplicatePluginError(
                f"{self.kind} plugin {resolved!r} is already provided by "
                f"{_entry_point_origin(self._entry_points[resolved])}; cannot also register "
                f"{_origin(obj)} (pass replace=True to override)"
            )
        self._factories[resolved] = factory
        return obj

    def unregister(self, name: str) -> None:
        """Remove a registration. Raises :class:`PluginNotFoundError` if absent."""
        self._discover()
        if self._factories.pop(name, None) is None and self._entry_points.pop(name, None) is None:
            raise PluginNotFoundError(self._not_found(name))

    def get(self, name: str) -> Callable[..., T]:
        """Return the factory registered under ``name``, importing it if needed."""
        self._discover()
        factory = self._factories.get(name)
        if factory is not None:
            return factory
        if name in self._conflicts:
            raise DuplicatePluginError(self._conflicts[name])
        ep = self._entry_points.get(name)
        if ep is None:
            raise PluginNotFoundError(self._not_found(name))
        try:
            loaded = ep.load()
        except Exception as exc:
            raise PluginError(
                f"could not load {self.kind} plugin {name!r} from {_entry_point_origin(ep)}: {exc}"
            ) from exc
        if self._coerce is not None:
            loaded = self._coerce(loaded)
        if not _is_factory(loaded):
            raise PluginError(
                f"{self.kind} plugin {name!r} from {_entry_point_origin(ep)} "
                "must point at a class or factory function"
            )
        del self._entry_points[name]
        self._factories[name] = loaded
        return loaded

    def __getitem__(self, name: str) -> Callable[..., T]:
        return self.get(name)

    def create(self, name: str, /, *args: Any, **kwargs: Any) -> T:
        """Build a new plugin object by calling its factory with the given arguments."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        """Names of every registered or installed plugin of this kind."""
        self._discover()
        return [*self._factories, *self._entry_points]

    def __contains__(self, name: object) -> bool:
        return name in self.names()

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def _discover(self) -> None:
        if self._discovered or self.group is None:
            return
        found: dict[str, EntryPoint] = {}
        clashing: dict[str, list[EntryPoint]] = {}
        for ep in entry_points(group=self.group):
            if ep.name in clashing:
                clashing[ep.name].append(ep)
                continue
            previous = found.get(ep.name)
            if previous is not None and previous.value != ep.value:
                # Refuse to guess, but only for this name: other plugins keep working.
                clashing[ep.name] = [found.pop(ep.name), ep]
                continue
            found[ep.name] = ep
        self._entry_points = found
        self._conflicts = {
            name: f"{self.kind} plugin {name!r} is provided more than once: by "
            + " and by ".join(_entry_point_origin(ep) for ep in eps)
            for name, eps in clashing.items()
        }
        self._discovered = True

    def _not_found(self, name: str) -> str:
        available = ", ".join(sorted(self.names())) or "none installed"
        return f"no {self.kind} plugin named {name!r} (available: {available})"
