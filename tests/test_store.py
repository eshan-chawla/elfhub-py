from pathlib import Path

import pytest

from elfhub.events import MessageAdded, RunStarted
from elfhub.persist import JsonlSessionStore, MemorySessionStore, SessionStore, messages_from
from elfhub.types import Message


@pytest.fixture(params=["memory", "jsonl"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> SessionStore:
    if request.param == "memory":
        return MemorySessionStore()
    return JsonlSessionStore(tmp_path)


async def test_sequence_numbers_increase_per_session(store: SessionStore):
    a1 = await store.append(RunStarted(session_id="a", run_id="r"))
    b1 = await store.append(RunStarted(session_id="b", run_id="r"))
    a2 = await store.append(RunStarted(session_id="a", run_id="r"))

    assert (a1.seq, a2.seq, b1.seq) == (1, 2, 1)


async def test_events_come_back_in_order_with_their_types(store: SessionStore):
    hello = Message.user("hello")
    await store.append(RunStarted(session_id="s", run_id="r"))
    await store.append(MessageAdded(session_id="s", run_id="r", message=hello))

    events = await store.events("s")

    assert [type(e) for e in events] == [RunStarted, MessageAdded]
    assert isinstance(events[1], MessageAdded) and events[1].message == hello


async def test_unknown_sessions_have_no_events(store: SessionStore):
    assert await store.events("nobody") == []


async def test_sessions_are_listed(store: SessionStore):
    await store.append(RunStarted(session_id="one", run_id="r"))
    await store.append(RunStarted(session_id="two", run_id="r"))

    assert sorted(await store.sessions()) == ["one", "two"]


async def test_jsonl_store_survives_a_restart(tmp_path: Path):
    first = JsonlSessionStore(tmp_path)
    await first.append(RunStarted(session_id="s", run_id="r"))
    await first.append(MessageAdded(session_id="s", run_id="r", message=Message.user("hi")))

    second = JsonlSessionStore(tmp_path)
    appended = await second.append(RunStarted(session_id="s", run_id="r2"))

    assert appended.seq == 3
    assert [e.seq for e in await second.events("s")] == [1, 2, 3]


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".hidden", "", "abc\n"])
async def test_jsonl_store_rejects_session_ids_that_are_not_plain_names(tmp_path: Path, bad: str):
    store = JsonlSessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError, match="session id"):
        await store.append(RunStarted(session_id=bad, run_id="r"))


def test_messages_are_rebuilt_from_the_log_in_order():
    first, second = Message.user("one"), Message.user("two")
    events = [
        RunStarted(session_id="s", run_id="r"),
        MessageAdded(session_id="s", run_id="r", message=first),
        MessageAdded(session_id="s", run_id="r", message=second),
    ]

    assert messages_from(events) == [first, second]


async def test_two_stores_on_one_folder_never_reuse_sequence_numbers(tmp_path: Path):
    first, second = JsonlSessionStore(tmp_path), JsonlSessionStore(tmp_path)

    await first.append(RunStarted(session_id="s", run_id="r"))
    await second.append(RunStarted(session_id="s", run_id="r"))
    await first.append(RunStarted(session_id="s", run_id="r"))

    assert [e.seq for e in await first.events("s")] == [1, 2, 3]


async def test_a_half_written_last_line_is_skipped_and_appends_continue(tmp_path: Path):
    store = JsonlSessionStore(tmp_path)
    await store.append(RunStarted(session_id="s", run_id="r"))
    with (tmp_path / "s.jsonl").open("a") as fh:
        fh.write('{"v": 1, "type": "run.sta')  # the process died mid-write

    with pytest.warns(UserWarning, match="incomplete"):
        appended = await JsonlSessionStore(tmp_path).append(RunStarted(session_id="s", run_id="r2"))

    assert appended.seq == 2
    with pytest.warns(UserWarning, match="incomplete"):
        assert [e.seq for e in await store.events("s")] == [1, 2]
