"""End-to-end: an installed third-party package plugs in through entry points alone.

Needs the fixture package: ``pip install ./tests/fixtures/plugin_pkg``.
"""

import textwrap
from pathlib import Path

import pytest

from elfhub import Elf
from elfhub.plugins import llms, modules, tools

try:
    import elfhub_fixture_plugin  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: no cover
    pytest.fail(
        "the fixture plugin is not installed: pip install ./tests/fixtures/plugin_pkg",
        pytrace=False,
    )


def test_installed_plugins_are_discovered():
    assert "fixture" in llms
    assert "fixture-greeter" in modules
    assert "fixture_shout" in tools


@pytest.fixture
def elf_dir(tmp_path: Path) -> Path:
    (tmp_path / "elf.toml").write_text(
        textwrap.dedent(
            """
            name = "plugged"
            llm = "fixture/model-a"
            tools = ["fixture_shout"]

            [modules.fixture-greeter]
            greeting = "Howdy"
            """
        )
    )
    return tmp_path


EXPECTED = "model-a | Howdy from the fixture plugin. | HELLO THERE"


def test_an_elf_built_only_from_plugins_runs_sync(elf_dir: Path):
    result = Elf.from_dir(elf_dir).run("hello there")

    assert (result.status, result.output) == ("completed", EXPECTED)


async def test_an_elf_built_only_from_plugins_runs_async(elf_dir: Path):
    result = await Elf.from_dir(elf_dir).arun("hello there")

    assert (result.status, result.output) == ("completed", EXPECTED)
