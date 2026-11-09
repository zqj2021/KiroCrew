"""A chat can switch one steering file off where Crew writes that file's text.

The store (``kiro_crew.steering_off``), the two writers that honour it (the Claude
Code ``[Steering resources]`` load and the folder steering section), the turn text
with and without an exclusion, and the switch route.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from conftest import plant_day_link
from kiro_crew import pinned_fs
from kiro_crew import steering_off as store
from kiro_crew.agent_sdk.provider_identity import PROVIDER_CLAUDE_CODE
from kiro_crew.context import ContextBuilder
from kiro_crew.context_assembly import inclusion
from kiro_crew.dashboard import chat_handlers
from kiro_crew.dashboard.handlers import chat_steering
from kiro_crew.dashboard.state import DashboardState, _ChatSlot
from kiro_crew.learn import LessonStore
from kiro_crew.memory import MemoryStore
from kiro_crew.skills import SkillsLoader

RULE = "GLOBAL-STEERING-RULE-3b7e"
FOLDER_RULE = "FOLDER-STEERING-RULE-9d1c"


@pytest.fixture(autouse=True)
def _close_skills_loaders(close_skills_loaders):
    """Every test here may build a ``ContextBuilder`` (``test/conftest.py``)."""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home with one global ``always`` guide, the shipped agent, and a data home."""
    steering = tmp_path / ".kiro" / "steering"
    steering.mkdir(parents=True)
    (steering / "global.md").write_text(f"# Global\n{RULE}\n", encoding="utf-8")
    agents = tmp_path / ".kiro" / "agents"
    agents.mkdir(parents=True)
    (agents / "kirocrew.json").write_text(
        json.dumps({"resources": ["file://.kiro/steering/**/*.md"]}), encoding="utf-8"
    )
    monkeypatch.setattr(store, "_store_path", lambda: tmp_path / "data" / store._FILE_NAME)
    with patch("pathlib.Path.home", return_value=tmp_path):
        yield tmp_path


@pytest.fixture
def standards(tmp_path):
    root = tmp_path / "standards"
    root.mkdir()
    (root / "coding.md").write_text(f"# Coding\n{FOLDER_RULE}\n", encoding="utf-8")
    return root


def _builder(tmp_path) -> ContextBuilder:
    return ContextBuilder(
        memory=MemoryStore(workspace=tmp_path / "ws"),
        skills=SkillsLoader(skills_path=tmp_path / "skills", install_builtins=False),
        lessons=LessonStore(base_dir=tmp_path),
    )


def _resolved(path: Path) -> str:
    return str(path.resolve())


def test_an_agent_cannot_write_the_store():
    """The record sits on both fences that keep it agent-unwritable.

    The file tool refuses the path, and the sandbox mounts the leaf read-only
    (pre-created, so an absent file cannot be created from inside either). Only the
    gateway's dashboard route, which runs outside the sandbox, writes it.
    """
    from kiro_crew import sandbox, security
    from kiro_crew.config.paths import data_home

    leaf = store._FILE_NAME
    assert store._store_path() == data_home() / leaf
    assert leaf in security._CREW_SECRET_LEAVES
    assert security.is_sensitive_path(f"~/.kiro/crew/{leaf}") is True
    assert security.is_sensitive_write_path(f"~/.kiro/crew/{leaf}") is True
    assert leaf in sandbox._CREW_READONLY_LEAVES
    assert leaf in sandbox._CREW_PRECREATE_READONLY_FILE_LEAVES


def test_the_store_keeps_paths_per_chat(home):
    assert store.steering_off_for("main") == frozenset()
    assert store.set_steering_off("main", "/a.md", True) == {"/a.md"}
    store.set_steering_off("other", "/b.md", True)
    assert store.steering_off_for("main") == {"/a.md"}
    assert store.set_steering_off("main", "/a.md", False) == frozenset()
    assert store.steering_off_for("main") == frozenset()
    assert store.steering_off_for("other") == {"/b.md"}


def test_a_write_keeps_the_record_on_its_inode(home):
    """A rename would detach a live sandbox's read-only bind; the rewrite does not."""
    store.set_steering_off("main", "/a.md", True)
    inode = store._store_path().stat().st_ino
    store.set_steering_off("main", "/b.md", True)
    assert store._store_path().stat().st_ino == inode


def test_a_planted_link_at_the_record_is_refused(home, tmp_path):
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}", encoding="utf-8")
    path = store._store_path()
    path.parent.mkdir(parents=True)
    plant_day_link(path, elsewhere)
    with pytest.raises(OSError):
        store.set_steering_off("main", "/a.md", True)
    assert elsewhere.read_text(encoding="utf-8") == "{}"


def test_a_switch_never_overwrites_an_unreadable_store(home):
    """Every other chat's record survives a broken file; the write refuses instead."""
    path = store._store_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"other": ["/b.md"]', encoding="utf-8")
    with pytest.raises(store.SteeringOffStoreUnreadable):
        store.set_steering_off("main", "/a.md", True)
    assert path.read_text(encoding="utf-8") == '{"other": ["/b.md"]'


def test_an_empty_pre_created_store_is_a_valid_empty_one(home):
    path = store._store_path()
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    assert store.set_steering_off("main", "/a.md", True) == {"/a.md"}


def test_every_caller_of_the_builder_honours_the_store(home):
    """A channel or messaging turn passes no set; the builder reads the chat's own."""
    store.set_steering_off("slack:1700", _resolved(home / ".kiro" / "steering" / "global.md"), True)
    msg, _ = _builder(home).build_message(
        "hello", True, "slack:1700", provider_type=PROVIDER_CLAUDE_CODE
    )
    assert RULE not in msg
    msg, _ = _builder(home).build_message(
        "hello", True, "slack:1701", provider_type=PROVIDER_CLAUDE_CODE
    )
    assert RULE in msg


def test_an_unreadable_store_never_reads_as_nothing_switched_off(home):
    """Garbage with no path in it switches nothing on that the store had off."""
    path = store._store_path()
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    assert store.steering_off_for("main") == frozenset()
    store._store_path().write_text('{"main": ["/a.md"]}', encoding="utf-8")
    assert store.steering_off_for("main") == {"/a.md"}
    path.write_text("not json", encoding="utf-8")
    assert store.steering_off_for("main") == {"/a.md"}


def test_a_save_cut_short_at_any_byte_keeps_every_chat_s_record(home, monkeypatch):
    """A power loss after any number of bytes leaves a whole record, after a restart."""
    store.set_steering_off("a", "/standards/a.md", True)
    store.set_steering_off("b", "/standards/b.md", True)
    path = store._store_path()
    whole = path.read_bytes()
    real_write = store.os.write
    before = {"a": {"/standards/a.md"}, "b": {"/standards/b.md"}}
    after = {"a": {"/standards/a.md", "/standards/long-one.md"}, "b": {"/standards/b.md"}}
    lost = 0
    for cut_at in range(1, 200):
        path.write_bytes(whole)
        budget = [cut_at]

        def power_loss(fd, data, budget=budget):
            n = min(len(data), budget[0])
            budget[0] -= n
            if n:
                real_write(fd, data[:n])
            if n < len(data):
                raise OSError(5, "power lost")
            return n

        monkeypatch.setattr(store.os, "write", power_loss)
        try:
            store.set_steering_off("a", "/standards/long-one.md", True)
        except OSError:
            lost += 1
        monkeypatch.setattr(store.os, "write", real_write)
        monkeypatch.setattr(store, "_LAST_GOOD", {})  # a restart: nothing held in memory
        seen = {key: set(store.steering_off_for(key)) for key in ("a", "b")}
        assert seen in (before, after), (cut_at, seen)
    assert lost  # some cut points did land inside the save
    # A save after a torn one starts a fresh line, so the torn bytes never join it.
    path.write_bytes(whole + b'\n{"a":["/standards/a.md","/x')
    monkeypatch.setattr(store, "_LAST_GOOD", {})
    assert store.set_steering_off("a", "/standards/c.md", True) == {
        "/standards/a.md",
        "/standards/c.md",
    }
    assert store.steering_off_for("b") == {"/standards/b.md"}


def test_the_claude_code_load_leaves_out_a_switched_off_file(home):
    off = frozenset({_resolved(home / ".kiro" / "steering" / "global.md")})
    msg, _ = _builder(home).build_message(
        "hello", True, "main", provider_type=PROVIDER_CLAUDE_CODE, steering_off=off
    )
    assert RULE not in msg
    msg, _ = _builder(home).build_message("hello", True, "main", provider_type=PROVIDER_CLAUDE_CODE)
    assert RULE in msg


def test_compaction_reinjection_leaves_it_out_too(home):
    off = frozenset({_resolved(home / ".kiro" / "steering" / "global.md")})
    msg, _ = _builder(home).build_message(
        "carry on",
        is_new_session=False,
        needs_reinjection=True,
        provider_type=PROVIDER_CLAUDE_CODE,
        steering_off=off,
    )
    assert RULE not in msg


@pytest.mark.skipif(
    not pinned_fs.supports_pinned_tree_walk(),
    reason="folder steering refuses to walk by name on this platform",
)
def test_the_folder_section_leaves_out_a_switched_off_file(home, standards):
    off = frozenset({_resolved(standards / "coding.md")})
    msg, _ = _builder(home).build_message(
        "hello",
        True,
        "main",
        provider_type="codex",
        steering_dirs=(str(standards),),
        steering_off=off,
    )
    assert FOLDER_RULE not in msg
    msg, _ = _builder(home).build_message(
        "hello", True, "main", provider_type="codex", steering_dirs=(str(standards),)
    )
    assert FOLDER_RULE in msg


@pytest.mark.skipif(
    not pinned_fs.supports_pinned_tree_walk(),
    reason="folder steering refuses to walk by name on this platform",
)
def test_with_nothing_switched_off_both_writers_give_the_old_text(home, standards):
    """Golden: an empty set takes exactly the call main makes."""
    from kiro_crew import context as ctx

    cap = ctx._resolve_caps(None).steering
    assert inclusion._render_folder_steering_section(
        (str(standards),), None, cap, steering_off=frozenset()
    ) == ctx.render_folder_steering(
        ctx.collect_folder_steering((str(standards),), project=None),
        max_chars=cap,
        scrub=ctx._neutralize_structural_markers,
    )
    kwargs = dict(agent=None, is_cc=True, essentials="", context_groups=None, project=None)
    assert inclusion.steering_resources_part(
        **kwargs, steering_off=frozenset()
    ) == ctx._load_steering_resources(None)


def _state(slot: _ChatSlot) -> DashboardState:
    state = MagicMock(spec=DashboardState)
    state._slots = {slot.key: slot}
    state.sessions = MagicMock()
    state.sessions.get_provider = MagicMock(return_value=None)
    state.owner_id = ""
    return state


async def _call(
    state: DashboardState, method: str, body: object = None, *, user: str = "local-app"
) -> tuple[int, dict]:
    """Call the route as *user*; the default is the machine-local dashboard owner."""

    @web.middleware
    async def identity(request, handler):
        request["app"] = ""
        if user:
            request["user"] = user
        return await handler(request)

    app = web.Application(middlewares=[identity])
    app["state"] = state
    app.router.add_get("/api/chat/slots/{slot}/steering", chat_steering.api_chat_slot_steering)
    app.router.add_put(
        "/api/chat/slots/{slot}/steering", chat_steering.api_chat_slot_steering_switch
    )
    async with TestClient(TestServer(app)) as client:
        resp = await client.request(method, "/api/chat/slots/main/steering", json=body)
        return resp.status, await resp.json()


@pytest.fixture
def claude_chat(home, monkeypatch):
    """A cold chat on the Claude Code harness whose turns run the default agent."""
    import kiro_crew.session_agent_selection as selection

    monkeypatch.setattr(
        chat_handlers.KiroCrewConfig,
        "load",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(acp_backend="claude", member_acp_backend=""),
            agents={},
            degraded_sections=frozenset(),
            workspaces={},
        ),
    )
    monkeypatch.setattr(
        selection,
        "resolve_session_agent_bindings",
        lambda *_a, **_k: SimpleNamespace(kiro_agent="kirocrew"),
    )
    return _state(_ChatSlot("main"))


@pytest.mark.asyncio
async def test_switching_a_file_off_is_stored_and_listed(claude_chat, home):
    status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    assert (status, row["reaches"], row["switch"]) == (200, "yes", "on")
    assert "resolved" not in row
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True})
    row = next(r for r in data["files"] if r["name"] == "global.md")
    assert (status, row["reaches"], row["reason"], row["switch"]) == (
        200,
        "no",
        "switched_off",
        "off",
    )
    assert store.steering_off_for("dashboard:main") == {
        _resolved(home / ".kiro" / "steering" / "global.md")
    }
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": False})
    assert store.steering_off_for("dashboard:main") == frozenset()


@pytest.mark.asyncio
async def test_a_file_crew_does_not_write_cannot_be_switched(home, monkeypatch):
    import kiro_crew.session_agent_selection as selection

    monkeypatch.setattr(
        chat_handlers.KiroCrewConfig,
        "load",
        lambda: SimpleNamespace(
            agent=SimpleNamespace(acp_backend="codex", member_acp_backend=""), agents={}
        ),
    )
    monkeypatch.setattr(
        selection,
        "resolve_session_agent_bindings",
        lambda *_a, **_k: SimpleNamespace(kiro_agent="kirocrew"),
    )
    state = _state(_ChatSlot("main"))
    _status, data = await _call(state, "GET")
    path = next(r for r in data["files"] if r["name"] == "global.md")["path"]
    status, data = await _call(state, "PUT", {"path": path, "off": True})
    assert (status, data["code"]) == (409, "not_switchable")
    assert store.steering_off_for("dashboard:main") == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"path": "", "off": True}, {"path": "x.md", "off": "yes"}, []])
async def test_a_malformed_switch_is_refused(claude_chat, body):
    status, _data = await _call(claude_chat, "PUT", body)
    assert status == 400


@pytest.mark.asyncio
async def test_an_unreadable_store_refuses_the_switch(claude_chat):
    path = store._store_path()
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True})
    assert (status, data["code"]) == (409, "store_unreadable")
    assert path.read_text(encoding="utf-8") == "not json"


@pytest.mark.asyncio
async def test_the_switch_takes_the_session_aware_app_gate(claude_chat, monkeypatch):
    """An app must own the session as well as the slot, as other per-slot writes do."""
    seen: list[str] = []

    def gate(_app, _slot, _name, operation):
        seen.append(operation)
        return web.json_response({"error": "nope"}, status=404)

    from kiro_crew.dashboard import slot_ownership

    monkeypatch.setattr(slot_ownership, "deny_app_slot_session_access", gate)
    status, _data = await _call(claude_chat, "PUT", {"path": "x.md", "off": True})
    assert (status, seen) == (404, ["slot_steering_switch"])


@pytest.mark.asyncio
@pytest.mark.parametrize("user", ["", "agent-internal"])
async def test_only_the_owner_can_switch(claude_chat, user):
    """A caller with no owner claim, such as an agent's own credential, is refused."""
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    status, _data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True}, user=user)
    assert status in (401, 403)
    assert store.steering_off_for("dashboard:main") == frozenset()


def test_a_full_disk_keeps_the_old_record(home, monkeypatch):
    """A save that runs out of space part way leaves the record before it standing."""
    store.set_steering_off("main", "/a.md", True)
    real_write = store.os.write

    def full(fd, data):
        real_write(fd, data[: len(data) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store.os, "write", full)
    with pytest.raises(OSError):
        store.set_steering_off("other", "/b.md", True)
    monkeypatch.setattr(store, "_LAST_GOOD", {})
    assert store.steering_off_for("main") == {"/a.md"}
    assert store.steering_off_for("other") == frozenset()


def test_a_read_waits_for_a_rewrite_in_flight(home, monkeypatch):
    """The reader takes the writer's lock, so it never sees a half-rewritten record."""
    held: list[bool] = []
    real = store._read_all

    def spy(**kwargs):
        held.append(store._WRITE_LOCK.locked())
        return real(**kwargs)

    monkeypatch.setattr(store, "_read_all", spy)
    store.steering_off_for("main")
    assert held == [True]


def test_the_log_folds_to_its_newest_record_on_the_same_inode(home, monkeypatch):
    monkeypatch.setattr(store, "_COMPACT_AT", 200)
    expected: set[str] = set()
    store.set_steering_off("main", "/start.md", True)
    expected.add("/start.md")
    inode = store._store_path().stat().st_ino
    for n in range(20):
        path, off = f"/f{n % 3}.md", n % 2 == 0
        store.set_steering_off("main", path, off)
        (expected.add if off else expected.discard)(path)
    assert store._store_path().stat().st_ino == inode
    assert store._store_path().stat().st_size < 400  # folded, not 21 records long
    monkeypatch.setattr(store, "_LAST_GOOD", {})
    assert store.steering_off_for("main") == expected


def test_a_fold_cut_short_before_the_trim_keeps_the_newest_record(home, monkeypatch):
    monkeypatch.setattr(store, "_COMPACT_AT", 1)
    store.set_steering_off("a", "/a.md", True)
    store.set_steering_off("b", "/b.md", True)
    real_trim = store.os.ftruncate
    kill = [True]

    def killed(fd, length):
        if kill.pop() if kill else False:
            raise KeyboardInterrupt  # the gateway dies between the fold and the trim
        return real_trim(fd, length)

    monkeypatch.setattr(store.os, "ftruncate", killed)
    with pytest.raises(KeyboardInterrupt):
        store.set_steering_off("a", "/a.md", False)
    monkeypatch.setattr(store, "_LAST_GOOD", {})
    assert store.steering_off_for("a") == frozenset()
    assert store.steering_off_for("b") == {"/b.md"}
    assert store.set_steering_off("a", "/c.md", True) == {"/c.md"}
    assert store.steering_off_for("b") == {"/b.md"}


@pytest.mark.skipif(
    not pinned_fs.supports_pinned_tree_walk(),
    reason="folder steering refuses to walk by name on this platform",
)
def test_compaction_reinjection_leaves_a_switched_off_folder_file_out(home, standards):
    off = frozenset({_resolved(standards / "coding.md")})
    kwargs = dict(
        is_new_session=False,
        needs_reinjection=True,
        provider_type="codex",
        steering_dirs=(str(standards),),
    )
    msg, _ = _builder(home).build_message("carry on", **kwargs, steering_off=off)
    assert FOLDER_RULE not in msg
    msg, _ = _builder(home).build_message("carry on", **kwargs, steering_off=frozenset())
    assert FOLDER_RULE in msg


@pytest.mark.asyncio
@pytest.mark.skipif(
    not pinned_fs.supports_pinned_tree_walk(),
    reason="folder steering refuses to walk by name on this platform",
)
async def test_a_folder_file_switches_off_through_the_route(claude_chat, standards, monkeypatch):
    """The route lists a folder row, stores its resolved path, and lists it as off."""
    from kiro_crew.dashboard.handlers import _shared

    async def dirs(_state, _key):
        return (str(standards),)

    monkeypatch.setattr(_shared, "session_folder_steering_dirs", dirs)
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "coding.md")
    assert (row["source"], row["switch"]) == ("folder", "on")
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True})
    row = next(r for r in data["files"] if r["name"] == "coding.md")
    assert (status, row["reaches"], row["reason"], row["switch"]) == (
        200,
        "no",
        "switched_off",
        "off",
    )
    assert store.steering_off_for("dashboard:main") == {_resolved(standards / "coding.md")}


@pytest.mark.asyncio
async def test_a_failed_write_answers_a_coded_error(claude_chat, monkeypatch):
    """A writer error (a full disk, a refused link) is a coded 500, and nothing changed."""

    def refuse(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store, "_append_record", refuse)
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True})
    assert (status, data["code"]) == (500, "write_failed")
    assert store.steering_off_for("dashboard:main") == frozenset()


@pytest.mark.asyncio
async def test_a_rebind_while_the_rows_are_built_refuses_the_switch(claude_chat, monkeypatch):
    """A channel rebind mid-request would store the switch under a key no turn reads."""
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    real_rows = chat_steering._chat_rows

    async def rows_then_rebind(*args, **kwargs):
        found = await real_rows(*args, **kwargs)
        claude_chat._slots["main"].linked_session_key = "slack:C1:T1"
        return found

    monkeypatch.setattr(chat_steering, "_chat_rows", rows_then_rebind)
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True})
    assert (status, data["code"]) == (409, "session_changed")
    assert store.steering_off_for("dashboard:main") == frozenset()
    assert store.steering_off_for("slack:C1:T1") == frozenset()


@pytest.mark.asyncio
async def test_a_save_that_fails_after_its_record_landed_claims_neither_outcome(
    claude_chat, monkeypatch
):
    """The record reached the file but the sync failed: the reply must not say nothing changed."""
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    real_fsync = store.os.fsync

    def sync_fails(fd):
        real_fsync(fd)
        raise OSError(5, "I/O error")

    monkeypatch.setattr(store.os, "fsync", sync_fails)
    status, data = await _call(claude_chat, "PUT", {"path": row["path"], "off": True})
    monkeypatch.setattr(store.os, "fsync", real_fsync)
    assert (status, data["code"]) == (500, "write_failed")
    assert "nothing changed" not in data["error"]
    # The list the client reads again shows what is stored: here, the switch applied.
    _status, data = await _call(claude_chat, "GET")
    row = next(r for r in data["files"] if r["name"] == "global.md")
    assert (row["switch"], row["reason"]) == ("off", "switched_off")


@pytest.mark.skipif(
    not pinned_fs.supports_pinned_tree_walk(),
    reason="folder steering refuses to walk by name on this platform",
)
def test_a_stored_switch_off_only_acts_where_crew_is_the_only_sender(home, standards):
    """kiro-cli may load the file itself, so the turn there keeps Crew's copy as well."""
    from kiro_crew.agent_sdk.provider_identity import PROVIDER_ACP

    store.set_steering_off("main", _resolved(standards / "coding.md"), True)
    kwargs = dict(steering_dirs=(str(standards),))
    msg, _ = _builder(home).build_message("hi", True, "main", provider_type="codex", **kwargs)
    assert FOLDER_RULE not in msg
    msg, _ = _builder(home).build_message("hi", True, "main", provider_type=PROVIDER_ACP, **kwargs)
    assert FOLDER_RULE in msg
