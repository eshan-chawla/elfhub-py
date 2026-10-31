"""Turn plain Python functions into tools."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast, get_type_hints, overload

import pydantic_core
from pydantic import BaseModel, ValidationError, create_model

from elfhub.tools.base import Tool, ToolAnnotations, ToolContext, ToolResult, ToolSpec, is_tool

_SCHEMA_MAPS = frozenset({"properties", "$defs", "definitions", "patternProperties"})


def _strip_titles(node: Any, *, in_map: bool = False) -> Any:
    """Drop pydantic's auto-generated ``title`` keys, which only add noise for the model.

    Keys of ``properties``-like maps are argument names, so a parameter that is
    literally called ``title`` survives.
    """
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():  # pyright: ignore[reportUnknownVariableType]
            if not in_map and key == "title" and isinstance(value, str):
                continue
            cleaned[key] = _strip_titles(value, in_map=not in_map and key in _SCHEMA_MAPS)
        return cleaned
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]  # pyright: ignore[reportUnknownVariableType]
    return node


def _to_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, str):
        return ToolResult.text(value)
    return ToolResult.text(pydantic_core.to_json(value).decode())


class FunctionTool:
    """A :class:`Tool` backed by a Python function (sync or async).

    Arguments are validated against the function's type hints before the call. A
    parameter annotated as :class:`ToolContext` is injected and hidden from the model.
    Sync functions run in a worker thread so they cannot block the event loop.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        annotations: ToolAnnotations | None = None,
    ) -> None:
        self.fn = fn
        tool_name = name or fn.__name__
        hints = get_type_hints(fn, include_extras=True)
        fields: dict[str, Any] = {}
        self._ctx_param: str | None = None
        for param in inspect.signature(fn).parameters.values():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(f"tool {tool_name!r} cannot take variadic parameter {param.name!r}")
            hint = hints.get(param.name, Any)
            if hint is ToolContext:
                self._ctx_param = param.name
                continue
            fields[param.name] = (hint, ... if param.default is param.empty else param.default)
        self._arguments: type[BaseModel] = create_model(f"{tool_name}_arguments", **fields)
        self.spec = ToolSpec(
            name=tool_name,
            description=description if description is not None else inspect.getdoc(fn) or "",
            input_schema=_strip_titles(self._arguments.model_json_schema()),
            annotations=annotations or ToolAnnotations(),
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            validated = self._arguments.model_validate(args)
        except ValidationError as exc:
            return ToolResult.error(f"invalid arguments for {self.spec.name}: {exc}")
        kwargs = {field: getattr(validated, field) for field in type(validated).model_fields}
        if self._ctx_param is not None:
            kwargs[self._ctx_param] = ctx
        if inspect.iscoroutinefunction(self.fn):
            value = await self.fn(**kwargs)
        else:
            value = await asyncio.to_thread(self.fn, **kwargs)
            if inspect.isawaitable(value):
                value = await value
        return _to_result(value)

    @property
    def name(self) -> str:
        return self.spec.name

    def __repr__(self) -> str:
        return f"FunctionTool({self.spec.name!r})"


@overload
def tool(fn: Callable[..., Any], /) -> FunctionTool: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
    destructive: bool | None = None,
    idempotent: bool = False,
    open_world: bool = True,
    parallel_safe: bool = False,
    timeout_s: float | None = 120.0,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = False,
    destructive: bool | None = None,
    idempotent: bool = False,
    open_world: bool = True,
    parallel_safe: bool = False,
    timeout_s: float | None = 120.0,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Make a tool from a function, as ``@tool`` or ``@tool(read_only=True, ...)``.

    ``destructive`` defaults to the opposite of ``read_only``.
    """
    annotations = ToolAnnotations(
        read_only=read_only,
        destructive=(not read_only) if destructive is None else destructive,
        idempotent=idempotent,
        open_world=open_world,
        parallel_safe=parallel_safe,
        timeout_s=timeout_s,
    )

    def wrap(target: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(target, name=name, description=description, annotations=annotations)

    return wrap(fn) if fn is not None else wrap


def as_tool(obj: Tool | Callable[..., Any]) -> Tool:
    """Return ``obj`` if it already is a :class:`Tool`, else wrap the function."""
    if is_tool(obj):
        return cast(Tool, obj)
    if inspect.isclass(obj) and hasattr(obj, "spec"):
        raise TypeError(f"{obj.__name__} is a tool class; pass an instance, e.g. {obj.__name__}()")
    if callable(obj):
        return FunctionTool(obj)
    raise TypeError(f"expected a Tool or a function, got {type(obj).__name__}")
