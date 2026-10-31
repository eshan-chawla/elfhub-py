"""Every Python example in the docs must run.

Mark an illustrative block that cannot run (for example, one that needs a real
model) with ``<!-- skip-test -->`` on the line before its opening fence.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
BLOCK = re.compile(r"(?P<skip><!-- skip-test -->\s*\n)?```python\n(?P<code>.*?)```", re.S)
PAGES = sorted(DOCS.rglob("*.md"))


def test_the_docs_have_pages():
    assert PAGES


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_python_examples_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {"__name__": "docs_example"}
    for match in BLOCK.finditer(page.read_text(encoding="utf-8")):
        if match["skip"]:
            continue
        exec(compile(match["code"], str(page), "exec"), namespace)
