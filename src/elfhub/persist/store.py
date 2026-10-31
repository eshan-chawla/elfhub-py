"""Session stores: append-only event logs, one per session."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from elfhub.errors import InvalidSessionId
from elfhub.events import Event, MessageAdded, parse_event
from elfhub.types import Message

_SESSION_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*")


@runtime_checkable
class SessionStore(Protocol):
    async def append(self, event: Event) -> Event:
        """Store ``event``, assigning the next ``seq`` for its session. Returns the stored event."""
        ...

    async def events(self, session_id: str) -> list[Event]: ...

    async def sessions(self) -> list[str]: ...


def messages_from(events: Iterable[Event]) -> list[Message]:
    """Rebuild a conversation from its event log."""
    return [event.message for event in events if isinstance(event, MessageAdded)]


class MemorySessionStore:
    """Keeps logs in memory. The default for elves without a home directory."""

    def __init__(self) -> None:
        self._logs: dict[str, list[Event]] = {}

    async def append(self, event: Event) -> Event:
        log = self._logs.setdefault(event.session_id, [])
        stored = event.model_copy(update={"seq": len(log) + 1})
        log.append(stored)
        return stored

    async def events(self, session_id: str) -> list[Event]:
        return list(self._logs.get(session_id, []))

    async def sessions(self) -> list[str]:
        return list(self._logs)


class JsonlSessionStore:
    """One ``<session_id>.jsonl`` file per session under ``root``.

    Several store objects (in one process) can share a folder: the next sequence
    number is re-read whenever the file changed behind this store's back. A line
    left half-written by a crash is skipped with a warning. Writers in separate
    processes must not share a session.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._known: dict[str, tuple[int, int]] = {}  # session -> (file size, next seq)

    async def append(self, event: Event) -> Event:
        path = self._path(event.session_id)
        self.root.mkdir(parents=True, exist_ok=True)
        size = path.stat().st_size if path.exists() else 0
        known = self._known.get(event.session_id)
        if known is not None and known[0] == size:
            seq = known[1]
        else:
            existing = self._read(path)
            seq = existing[-1].seq + 1 if existing else 1
        stored = event.model_copy(update={"seq": seq})
        line = stored.model_dump_json() + "\n"
        with path.open("ab") as fh:
            if size and not self._ends_with_newline(path):
                fh.write(b"\n")  # never glue a new event onto a half-written line
            fh.write(line.encode("utf-8"))
        self._known[event.session_id] = (path.stat().st_size, seq + 1)
        return stored

    async def events(self, session_id: str) -> list[Event]:
        return self._read(self._path(session_id))

    async def sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        return [path.stem for path in sorted(self.root.glob("*.jsonl"))]

    def _read(self, path: Path) -> list[Event]:
        if not path.exists():
            return []
        events: list[Event] = []
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(parse_event(line))
                except ValueError:
                    warnings.warn(
                        f"skipping an incomplete or corrupt line {number} in {path}",
                        stacklevel=3,
                    )
        return events

    @staticmethod
    def _ends_with_newline(path: Path) -> bool:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            return fh.read(1) == b"\n"

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise InvalidSessionId(
                f"session id {session_id!r} must be a plain name (letters, digits, _ . -)"
            )
        return self.root / f"{session_id}.jsonl"
