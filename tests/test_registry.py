from collections import OrderedDict, deque
from importlib.metadata import EntryPoint

import pytest

from elfhub.errors import DuplicatePluginError, PluginError, PluginNotFoundError
from elfhub.plugins import registry as registry_module
from elfhub.plugins.registry import Registry


class Widget:
    name = "widget"

    def __init__(self, size: int = 1) -> None:
        self.size = size


def test_bare_decorator_registers_class_under_its_name():
    reg: Registry[Widget] = Registry("widget")

    reg.register(Widget)

    assert reg.create("widget", size=3).size == 3


def test_decorator_with_explicit_name():
    reg: Registry[str] = Registry("thing")

    @reg.register(name="hello")
    def make_hello() -> str:
        return "hi"

    assert reg.create("hello") == "hi"
    assert make_hello() == "hi"  # decorator returns the original object


def test_plain_function_without_name_is_a_plugin_error():
    reg: Registry[str] = Registry("thing")

    def anonymous() -> str:
        return "x"

    with pytest.raises(PluginError, match="name"):
        reg.register(anonymous)


def test_registering_an_instance_is_rejected():
    reg: Registry[Widget] = Registry("widget")

    with pytest.raises(TypeError, match="instance"):
        reg.register(Widget())  # pyright: ignore[reportArgumentType]


def test_create_returns_a_fresh_object_each_time():
    reg: Registry[Widget] = Registry("widget")
    reg.register(Widget)

    first = reg.create("widget", size=5)
    second = reg.create("widget")

    assert first is not second
    assert (first.size, second.size) == (5, 1)


def test_factories_need_not_accept_config():
    reg: Registry[int] = Registry("number")
    reg.register(lambda: 42, name="answer")

    assert reg.create("answer") == 42


def test_duplicate_name_is_an_error_naming_both_origins():
    reg: Registry[Widget] = Registry("widget")
    reg.register(Widget)

    class OtherWidget(Widget):
        name = "widget"

    with pytest.raises(DuplicatePluginError) as info:
        reg.register(OtherWidget)

    message = str(info.value)
    assert "Widget" in message and "OtherWidget" in message


def test_replace_true_overrides_an_existing_registration():
    reg: Registry[Widget] = Registry("widget")
    reg.register(Widget)

    class BigWidget(Widget):
        name = "widget"

        def __init__(self) -> None:
            super().__init__(size=100)

    reg.register(BigWidget, replace=True)

    assert reg.create("widget").size == 100


def test_getitem_and_get_return_the_same_factory():
    reg: Registry[Widget] = Registry("widget")
    reg.register(Widget)

    assert reg["widget"] is reg.get("widget") is Widget


def test_unknown_name_lists_what_is_available():
    reg: Registry[Widget] = Registry("widget")
    reg.register(Widget)

    with pytest.raises(PluginNotFoundError, match="widget") as info:
        reg.get("gadget")

    assert isinstance(info.value, KeyError)
    assert "gadget" in str(info.value)


def test_unregister_removes_the_entry():
    reg: Registry[Widget] = Registry("widget")
    reg.register(Widget)

    reg.unregister("widget")

    assert "widget" not in reg
    with pytest.raises(PluginNotFoundError):
        reg.unregister("widget")


def _fake_entry_points(monkeypatch: pytest.MonkeyPatch, *eps: EntryPoint) -> list[str]:
    requested: list[str] = []

    def fake(*, group: str) -> list[EntryPoint]:
        requested.append(group)
        return [ep for ep in eps if ep.group == group]

    monkeypatch.setattr(registry_module, "entry_points", fake)
    return requested


def test_entry_points_are_listed_without_being_imported(monkeypatch: pytest.MonkeyPatch):
    _fake_entry_points(
        monkeypatch,
        EntryPoint(name="ordered", value="collections:OrderedDict", group="elfhub.test"),
        EntryPoint(name="broken", value="elfhub_no_such_module:Thing", group="elfhub.test"),
    )
    reg: Registry[object] = Registry("test", group="elfhub.test")

    assert set(reg.names()) == {"ordered", "broken"}
    assert reg.get("ordered") is OrderedDict


def test_entry_point_import_failure_is_a_plugin_error(monkeypatch: pytest.MonkeyPatch):
    _fake_entry_points(
        monkeypatch,
        EntryPoint(name="broken", value="elfhub_no_such_module:Thing", group="elfhub.test"),
    )
    reg: Registry[object] = Registry("test", group="elfhub.test")

    with pytest.raises(PluginError, match="broken"):
        reg.get("broken")


def test_discovery_is_lazy_and_happens_once(monkeypatch: pytest.MonkeyPatch):
    requested = _fake_entry_points(
        monkeypatch,
        EntryPoint(name="ordered", value="collections:OrderedDict", group="elfhub.test"),
    )
    reg: Registry[object] = Registry("test", group="elfhub.test")
    assert requested == []

    reg.names()
    reg.get("ordered")
    "ordered" in reg  # noqa: B015

    assert requested == ["elfhub.test"]


def test_two_entry_points_with_the_same_name_conflict(monkeypatch: pytest.MonkeyPatch):
    _fake_entry_points(
        monkeypatch,
        EntryPoint(name="dup", value="collections:OrderedDict", group="elfhub.test"),
        EntryPoint(name="dup", value="collections:Counter", group="elfhub.test"),
        EntryPoint(name="fine", value="collections:deque", group="elfhub.test"),
    )
    reg: Registry[object] = Registry("test", group="elfhub.test")

    with pytest.raises(DuplicatePluginError, match="dup"):
        reg.get("dup")
    # One clash must not break every other plugin of the same kind.
    assert reg.names() == ["fine"]
    assert reg.get("fine") is deque


def test_explicit_registration_conflicts_with_an_entry_point(monkeypatch: pytest.MonkeyPatch):
    _fake_entry_points(
        monkeypatch,
        EntryPoint(name="widget", value="collections:OrderedDict", group="elfhub.test"),
    )
    reg: Registry[object] = Registry("test", group="elfhub.test")

    with pytest.raises(DuplicatePluginError):
        reg.register(Widget)


def test_the_same_entry_point_seen_twice_is_not_a_conflict(monkeypatch: pytest.MonkeyPatch):
    # importlib.metadata reports a distribution twice when it is on sys.path twice.
    ep = EntryPoint(name="ordered", value="collections:OrderedDict", group="elfhub.test")
    _fake_entry_points(monkeypatch, ep, ep)
    reg: Registry[object] = Registry("test", group="elfhub.test")

    assert reg.names() == ["ordered"]


class Gadget:
    """A stateless object that a registry may hold as-is (like a tool)."""

    name = "gadget"


def test_coerce_lets_a_registry_hold_stateless_instances(monkeypatch: pytest.MonkeyPatch):
    def constant(obj: object) -> object:
        return (lambda: obj) if isinstance(obj, Gadget) else obj

    gadget = Gadget()
    reg: Registry[Gadget] = Registry("gadget", coerce=constant)

    returned = reg.register(gadget)

    assert returned is gadget
    assert reg.create("gadget") is gadget


def test_coerce_applies_to_entry_points(monkeypatch: pytest.MonkeyPatch):
    _fake_entry_points(
        monkeypatch,
        EntryPoint(name="cwd", value="os:curdir", group="elfhub.test"),  # a str instance
    )
    reg: Registry[str] = Registry("test", group="elfhub.test", coerce=lambda o: lambda: o)

    assert reg.create("cwd") == "."


def test_the_tools_registry_builds_class_based_tools():
    from elfhub.plugins import tools
    from elfhub.tools import ToolResult, ToolSpec

    class Lookup:
        spec = ToolSpec(name="lookup_test", input_schema={"type": "object", "properties": {}})
        name = "lookup_test"

        async def __call__(self, args: object, ctx: object) -> ToolResult:
            return ToolResult.text("x")

    tools.register(Lookup)
    try:
        assert isinstance(tools.create("lookup_test"), Lookup)
    finally:
        tools.unregister("lookup_test")
