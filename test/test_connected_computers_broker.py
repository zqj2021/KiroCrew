"""Connected computers: the remote gateway's connections and channels (B07, B08, B10, B11).

* B07 -- revoking one of two computers ends only its channels, at once.
* B08 -- a connection lost while a tool call runs reports the outcome unknown;
  nothing is resent, and nothing moves to another connection or computer.
* B10 -- an offline computer refuses new work; the binding is kept.
* B11 -- an older credential cannot replace a live connection.
* R11 -- a stream that ends with calls waiting answers each of them unknown.
* G9 -- past 64 waiting calls the next is refused "not sent", none is forgotten.
Plus the bounds: channels per chat and per connection, the credit window, and the wire shape;
cancellation before and after the computer starts a tool server; and connection
generations (a replaced socket's late end never touches its replacement).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kiro_crew.connected_computers import broker as broker_mod
from kiro_crew.connected_computers import protocol
from kiro_crew.connected_computers.broker import (
    STATE_INTERRUPTED,
    STATE_OPEN,
    ChannelInterrupted,
    ConnectionBroker,
    DeviceConnection,
    DeviceOffline,
)
from kiro_crew.connected_computers.connector_token import CredentialRejected
from kiro_crew.connected_computers.devices import STATUS_ACTIVE, DeviceRecord
from kiro_crew.connected_computers.launch import ConnectedComputerLaunch, PendingCalls
from kiro_crew.remote_session.link import LinkUnavailable
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS, async_wait_until, await_until

pytestmark = pytest.mark.asyncio

CALL = b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"run_command"}}\n'


def _record(device_id: str = "laptop", cgen: int = 1) -> DeviceRecord:
    return DeviceRecord(
        device_id=device_id,
        label=f"{device_id} label",
        owner="me",
        fingerprint="a" * 64,
        cgen=cgen,
        status=STATUS_ACTIVE,
        created=1.0,
        updated=1.0,
    )


class FakeSocket:
    """Records every frame the gateway sends to one computer."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.closed = False

    async def send(self, text: str) -> None:
        self.frames.append(json.loads(text))

    async def close(self) -> None:
        self.closed = True

    def ops(self, op: str) -> list[dict]:
        return [f for f in self.frames if f["op"] == op]


def _connect(broker: ConnectionBroker, device_id: str = "laptop", *, cgen: int = 1, mcp=()):
    sock = FakeSocket()
    conn = DeviceConnection(
        _record(device_id, cgen), sock.send, root="/work", mcp=tuple(mcp), closer=sock.close
    )
    broker.attach(conn)
    return conn, sock


async def _open(
    conn: DeviceConnection,
    sock: FakeSocket,
    kind: str = "workspace",
    session_key: str = "session-1",
):
    """Open a channel and answer it as the computer would."""
    before = len(sock.ops("open"))
    task = asyncio.ensure_future(conn.open(kind, session_key))
    await async_wait_until(lambda: len(sock.ops("open")) > before)
    ch = sock.ops("open")[-1]["ch"]
    conn.handle({"op": "opened", "ch": ch})
    return await asyncio.wait_for(task, 30)


async def _linked(broker: ConnectionBroker, sock: FakeSocket, conn: DeviceConnection):
    launch = ConnectedComputerLaunch("laptop", "Laptop", session_key="session-1", broker=broker)
    port = await launch.start_worker("w1", [protocol.KIND_WORKSPACE], b"unused")
    link = launch.make_link("w1", b"unused", launch.dial(port), **launch.link_kwargs())
    link.start()
    await async_wait_until(lambda: sock.ops("open"))
    conn.handle({"op": "opened", "ch": sock.ops("open")[-1]["ch"]})
    await asyncio.wait_for(link.wait_connected(30), 30)
    return launch, link


async def test_b08_a_lost_connection_reports_the_call_unknown_and_never_resends():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    await link.write(CALL)
    assert len(sock.ops("data")) == 1

    broker.detach(conn, "the socket dropped")

    answer = await asyncio.wait_for(link.stdout.read(1 << 20), 30)
    reply = json.loads(answer.decode().strip())
    assert reply["id"] == 7
    assert reply["result"]["isError"] is True
    assert "unknown" in reply["result"]["content"][0]["text"]
    assert "not retried" in reply["result"]["content"][0]["text"]
    with pytest.raises(LinkUnavailable):
        await asyncio.wait_for(link.stdout.read(1 << 20), 30)
    assert link.state == "lost"
    # The one call was sent exactly once.
    assert len(sock.ops("data")) == 1


async def test_b08_a_reconnect_does_not_revive_or_replay_an_interrupted_channel():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    await link.write(CALL)
    broker.detach(conn, "the socket dropped")
    _new, new_sock = _connect(broker)
    await asyncio.wait_for(link.stdout.read(1 << 20), 30)
    with pytest.raises(LinkUnavailable):
        await link.write(CALL)
    assert new_sock.frames == [], "nothing may be resent on the new connection"


async def test_b08_an_answered_call_is_not_reported_unknown():
    pending = PendingCalls()
    assert pending.admit(CALL) == (CALL, b"")
    pending.observe_response(b'{"jsonrpc":"2.0","id":7,"result":{"content":[]}}\n')
    assert pending.close_out("lost") == b""


async def test_b10_an_offline_computer_refuses_new_work():
    broker = ConnectionBroker()
    with pytest.raises(DeviceOffline):
        await broker.open_channel("laptop", protocol.KIND_WORKSPACE, session_key="s")


async def test_b07_revoking_one_computer_leaves_the_other_running():
    broker = ConnectionBroker()
    conn_a, sock_a = _connect(broker, "alpha")
    conn_b, sock_b = _connect(broker, "beta")
    chan_a = await _open(conn_a, sock_a)
    chan_b = await _open(conn_b, sock_b)

    assert await broker.disconnect("alpha", "access revoked") is True

    assert chan_a.state == STATE_INTERRUPTED
    assert sock_a.ops("stop") and sock_a.closed
    assert chan_b.state == STATE_OPEN
    assert broker.live("beta") is conn_b and not sock_b.ops("stop")
    with pytest.raises(DeviceOffline):
        await broker.open_channel("alpha", protocol.KIND_WORKSPACE, session_key="s")


async def test_b11_an_older_credential_cannot_replace_a_live_connection():
    broker = ConnectionBroker()
    live, _sock = _connect(broker, cgen=2)
    stale = DeviceConnection(_record(cgen=1), FakeSocket().send, root="/work")
    with pytest.raises(CredentialRejected):
        broker.attach(stale)
    assert broker.live("laptop") is live


async def test_a_reconnect_of_the_same_generation_replaces_and_interrupts_the_old():
    broker = ConnectionBroker()
    old, old_sock = _connect(broker)
    channel = await _open(old, old_sock)
    new, _ = _connect(broker)
    assert broker.live("laptop") is new
    assert channel.state == STATE_INTERRUPTED
    await async_wait_until(lambda: old_sock.closed)


async def _fill_chat(conn: DeviceConnection, sock: FakeSocket, session_key: str) -> None:
    """Open one chat's full allowance: its workspace and every exposed server."""
    await _open(conn, sock, protocol.KIND_WORKSPACE, session_key)
    for name in conn.mcp:
        channel = await _open(conn, sock, f"mcp:{name}", session_key)
        assert channel.state == STATE_OPEN


def _all_servers() -> tuple[str, ...]:
    return tuple(f"server-{i}" for i in range(protocol.MAX_EXPOSED_MCP))


async def test_channels_per_chat_are_bounded():
    assert protocol.MAX_CHANNELS == protocol.MAX_EXPOSED_MCP + 1
    broker = ConnectionBroker()
    conn, sock = _connect(broker, mcp=_all_servers())
    await _fill_chat(conn, sock, "session-1")
    assert len(conn.channels) == protocol.MAX_CHANNELS
    opens = len(sock.ops("open"))
    with pytest.raises(ChannelInterrupted, match="per chat"):
        await conn.open(protocol.KIND_WORKSPACE, "session-1")
    # Refused before anything reached the computer.
    assert len(sock.ops("open")) == opens


async def test_a_full_chat_does_not_use_up_another_chats_room():
    """The per-chat limit is per chat: a second chat on the same computer still opens."""
    broker = ConnectionBroker()
    conn, sock = _connect(broker, mcp=_all_servers())
    await _fill_chat(conn, sock, "session-1")
    second = await _open(conn, sock, protocol.KIND_WORKSPACE, "session-2")
    assert second.state == STATE_OPEN
    assert second.session_key == "session-2"
    await _fill_chat(conn, sock, "session-3")
    assert len(conn.channels) == 2 * protocol.MAX_CHANNELS + 1


async def test_a_closed_channel_gives_its_chat_the_room_back():
    broker = ConnectionBroker()
    conn, sock = _connect(broker, mcp=_all_servers())
    await _fill_chat(conn, sock, "session-1")
    first = next(iter(conn.channels.values()))
    await first.aclose()
    again = await _open(conn, sock, protocol.KIND_WORKSPACE, "session-1")
    assert again.state == STATE_OPEN


async def test_channels_per_connection_stay_bounded_across_chats():
    assert protocol.MAX_CONNECTION_CHANNELS % protocol.MAX_CHANNELS == 0
    broker = ConnectionBroker()
    conn, sock = _connect(broker, mcp=_all_servers())
    chats = protocol.MAX_CONNECTION_CHANNELS // protocol.MAX_CHANNELS
    for n in range(chats):
        await _fill_chat(conn, sock, f"session-{n}")
    assert len(conn.channels) == protocol.MAX_CONNECTION_CHANNELS
    opens = len(sock.ops("open"))
    with pytest.raises(ChannelInterrupted, match="across all chats"):
        await conn.open(protocol.KIND_WORKSPACE, "a-new-chat")
    assert len(sock.ops("open")) == opens


async def test_b14_only_an_exposed_local_mcp_server_can_be_opened():
    broker = ConnectionBroker()
    conn, sock = _connect(broker, mcp=("notes",))
    with pytest.raises(ChannelInterrupted):
        await conn.open("mcp:browser", "s")
    channel = await _open(conn, sock, "mcp:notes")
    assert channel.kind == "mcp:notes"


async def test_a_computer_sending_past_its_window_breaks_the_protocol():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    channel = await _open(conn, sock)
    piece = b"x" * protocol.MAX_DATA_BYTES
    with pytest.raises(protocol.ProtocolError):
        for _ in range(protocol.WINDOW_BYTES // len(piece) + 1):
            conn.handle(protocol.data_frames(channel.id, piece)[0])


async def test_reading_grants_the_computer_credit():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    channel = await _open(conn, sock)
    conn.handle(protocol.data_frames(channel.id, b"hello")[0])
    assert await asyncio.wait_for(channel.stdout.read(100), 30) == b"hello"
    await async_wait_until(lambda: sock.ops("credit"))
    assert sock.ops("credit")[-1] == {"op": "credit", "ch": channel.id, "bytes": 5}


def _call(rid: int) -> bytes:
    return (
        json.dumps(
            {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": "run_command"}}
        ).encode()
        + b"\n"
    )


def _sent_lines(sock: FakeSocket, ch: int) -> list[dict]:
    """Every JSON-RPC line the gateway sent on channel ``ch``, in order."""
    raw = b"".join(protocol.data_of(f) for f in sock.ops("data") if f["ch"] == ch)
    return [json.loads(line) for line in raw.split(b"\n") if line.strip()]


async def _drain(link, limit: int = 64) -> bytes:
    """Read the link to its end (``b""`` or ``LinkUnavailable``), bounded."""
    out = bytearray()
    for _ in range(limit):
        try:
            data = await asyncio.wait_for(link.stdout.read(1 << 20), 30)
        except LinkUnavailable:
            break
        if not data:
            break
        out += data
    return bytes(out)


def _replies(raw: bytes) -> dict:
    return {
        msg["id"]: msg
        for msg in (json.loads(line) for line in raw.split(b"\n") if line.strip().startswith(b"{"))
        if "id" in msg
    }


async def test_g9_b08_calls_past_the_limit_are_refused_and_never_sent():
    """R11/G9: the 65th waiting call is answered "not sent"; none is forgotten."""
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    ch = sock.ops("open")[-1]["ch"]
    for rid in range(1, 66):
        await link.write(_call(rid))

    sent = [m["id"] for m in _sent_lines(sock, ch)]
    assert sent == list(range(1, 65)), "the call past the limit must not reach the computer"
    refusal = _replies(await asyncio.wait_for(link.stdout.read(1 << 20), 30))
    assert set(refusal) == {65}
    assert refusal[65]["result"]["isError"] is True
    assert "not sent" in refusal[65]["result"]["content"][0]["text"]

    broker.detach(conn, "the socket dropped")
    unknown = _replies(await _drain(link))
    # Every one of the 64 sent calls is answered unknown -- the oldest included.
    assert set(unknown) == set(range(1, 65))
    assert all("unknown" in r["result"]["content"][0]["text"] for r in unknown.values())


async def test_r11_b08_a_channel_ending_mid_reply_answers_every_waiting_call_unknown():
    """R11: EOF with calls waiting is never a silent end of the stream."""
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    ch = sock.ops("open")[-1]["ch"]
    await link.write(_call(7))
    await link.write(_call(8))
    half = b'{"jsonrpc":"2.0","id":7,"res'
    conn.handle(protocol.data_frames(ch, half)[0])
    conn.handle({"op": "closed", "ch": ch, "reason": "exited with status 1"})

    raw = await _drain(link)
    assert raw.startswith(half + b"\n"), "a half-delivered reply is ended before any answer"
    replies = _replies(raw[len(half) :])
    assert set(replies) == {7, 8}
    for reply in replies.values():
        text = reply["result"]["content"][0]["text"]
        assert reply["result"]["isError"] is True
        assert "unknown" in text and "not retried" in text
    assert link.state == "finished"
    assert await asyncio.wait_for(link.stdout.read(1 << 20), 30) == b""


async def test_r11_a_clean_end_with_nothing_waiting_adds_nothing():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    ch = sock.ops("open")[-1]["ch"]
    await link.write(_call(7))
    conn.handle(protocol.data_frames(ch, b'{"jsonrpc":"2.0","id":7,"result":{}}\n')[0])
    conn.handle({"op": "closed", "ch": ch, "reason": "exited with status 0"})
    assert await _drain(link) == b'{"jsonrpc":"2.0","id":7,"result":{}}\n'


async def test_g9_a_request_line_is_sent_only_once_it_is_whole():
    """A call split across writes is admitted (and counted) as one line."""
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    ch = sock.ops("open")[-1]["ch"]
    line = _call(9)
    await link.write(line[:10])
    assert not sock.ops("data"), "half a request is held, not sent"
    await link.write(line[10:])
    assert [m["id"] for m in _sent_lines(sock, ch)] == [9]


async def test_g9_a_partial_last_request_is_sent_before_the_input_ends():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    ch = sock.ops("open")[-1]["ch"]
    unterminated = _call(5).rstrip(b"\n")
    await link.write(unterminated)
    link.close_input()
    await async_wait_until(lambda: sock.ops("eof"))
    ops = [f["op"] for f in sock.frames if f.get("ch") == ch and f["op"] in ("data", "eof")]
    assert ops == ["data", "eof"]
    assert b"".join(protocol.data_of(f) for f in sock.ops("data")) == unterminated


async def test_g9_an_over_long_request_is_refused_explicitly():
    from kiro_crew.connected_computers import launch as launch_mod

    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    with pytest.raises(LinkUnavailable, match="too large"):
        await link.write(b"x" * (launch_mod._MAX_TRACKED_LINE + 1))
    assert not sock.ops("data")


async def test_b08_cancelling_an_open_before_the_computer_answers_closes_it_there():
    """Cancelled before spawn: the computer is told to close; a late opened is dropped."""
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    task = asyncio.ensure_future(conn.open(protocol.KIND_WORKSPACE, "session-1"))
    await async_wait_until(lambda: sock.ops("open"))
    ch = sock.ops("open")[-1]["ch"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await async_wait_until(lambda: sock.ops("close"))
    assert sock.ops("close")[-1] == {"op": "close", "ch": ch}
    assert conn.channels == {}
    conn.handle({"op": "opened", "ch": ch})  # late answer: no owner, dropped
    assert conn.channels == {}


async def test_b08_closing_a_link_while_it_connects_closes_the_channel_there():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    launch = ConnectedComputerLaunch("laptop", "Laptop", session_key="session-1", broker=broker)
    port = await launch.start_worker("w1", [protocol.KIND_WORKSPACE], b"unused")
    link = launch.make_link("w1", b"unused", launch.dial(port), **launch.link_kwargs())
    link.start()
    await async_wait_until(lambda: sock.ops("open"))
    await asyncio.wait_for(link.aclose(), 30)
    await async_wait_until(lambda: sock.ops("close"))
    assert conn.channels == {}


async def test_b08_closing_a_link_after_spawn_stops_its_tree_there():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    _launch, link = await _linked(broker, sock, conn)
    ch = sock.ops("open")[-1]["ch"]
    await link.write(CALL)
    await asyncio.wait_for(link.aclose(), 30)
    assert sock.ops("close")[-1] == {"op": "close", "ch": ch}
    assert len(sock.ops("data")) == 1, "closing never resends"


class _FakeRelay:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_an_ended_runtime_closes_only_the_relays_it_attached_not_a_respawns():
    from kiro_crew.connected_computers import launch as launch_mod

    launch = ConnectedComputerLaunch("laptop", "Laptop", session_key="session-1")
    first_ws, first_mcp = _FakeRelay("ws-1"), _FakeRelay("mcp-1")
    launch.relay, launch.mcp_relays = first_ws, [first_mcp]
    alive = [True]
    ended: list[str] = []
    before = set(launch_mod._WATCHERS)
    launch.close_relay_when_ended(lambda: alive[0], lambda: ended.append("first"))
    [watcher] = set(launch_mod._WATCHERS) - before
    first_close = launch.close_relays, launch.own_relays()

    # The respawn attaches new relays to the same launch before the old
    # runtime's end-watcher gets to run.
    second_ws, second_mcp = _FakeRelay("ws-2"), _FakeRelay("mcp-2")
    launch.relay, launch.mcp_relays = second_ws, [second_mcp]
    alive[0] = False
    await asyncio.wait_for(watcher, 30)

    assert ended == ["first"]
    assert first_ws.closed and first_mcp.closed
    assert not second_ws.closed and not second_mcp.closed
    assert launch.relay is second_ws and launch.mcp_relays == [second_mcp]

    # The first runtime's kill-time close is scoped the same way.
    close, owned = first_close
    await asyncio.wait_for(close(owned), 30)
    assert not second_ws.closed and not second_mcp.closed


async def test_b08_a_replaced_connection_ending_late_does_not_touch_its_replacement():
    """Connection generation: the old socket's late end and frames reach only itself."""
    broker = ConnectionBroker()
    old, old_sock = _connect(broker)
    old_channel = await _open(old, old_sock)
    new, new_sock = _connect(broker)
    assert new.generation > old.generation
    new_channel = await _open(new, new_sock)
    assert old_channel.state == STATE_INTERRUPTED

    old.handle(protocol.data_frames(new_channel.id, b"stale")[0])
    broker.detach(old, "the old socket finally closed")

    assert broker.live("laptop") is new
    assert new_channel.state == STATE_OPEN
    assert new_channel._recv.held == 0, "a frame from the old socket never reaches the new"
    with pytest.raises(ChannelInterrupted):
        await old.open(protocol.KIND_WORKSPACE, "s")
    assert new_sock.ops("stop") == []


class TestWire:
    def test_unknown_op_is_refused(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode('{"op":"exec","argv":["sh"]}', allowed=protocol.FROM_GATEWAY)

    def test_the_gateway_cannot_name_a_path_or_argv(self):
        frame = '{"op":"open","ch":1,"kind":"workspace","root":"/"}'
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(frame, allowed=protocol.FROM_GATEWAY)

    @pytest.mark.parametrize("kind", ["shell", "mcp:", "mcp:../x", "workspace2"])
    def test_unknown_kinds_are_refused(self, kind):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode(
                json.dumps({"op": "open", "ch": 1, "kind": kind}), allowed=protocol.FROM_GATEWAY
            )

    def test_a_boolean_is_not_a_channel_number(self):
        with pytest.raises(protocol.ProtocolError):
            protocol.decode('{"op":"eof","ch":true}', allowed=protocol.FROM_DEVICE)

    def test_data_is_split_into_bounded_frames(self):
        frames = protocol.data_frames(3, b"y" * (protocol.MAX_DATA_BYTES * 2 + 1))
        assert [len(protocol.data_of(f)) for f in frames] == [
            protocol.MAX_DATA_BYTES,
            protocol.MAX_DATA_BYTES,
            1,
        ]


@pytest.mark.parametrize(
    "entrypoint,stalled,refusal",
    [
        ("disconnect", "lock", "revoked"),
        ("disconnect", "lock", "removed"),
        ("shutdown", "lock", ""),
        ("stop", "send", ""),
        ("stop", "close", ""),
    ],
)
async def test_stop_bounds_lock_send_and_close(monkeypatch, entrypoint, stalled, refusal):
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    channel = await _open(conn, sock)
    entered = asyncio.Event()
    release = asyncio.Event()
    bounds: list[tuple[str, float]] = []
    close_attempts = []

    async def block():
        entered.set()
        await await_until(release, what=f"release stalled {stalled}")

    class HeldLock:
        async def __aenter__(self):
            await block()

        async def __aexit__(self, *args):
            return False

    async def send(text):
        if stalled == "send":
            await block()
        await sock.send(text)

    async def close():
        close_attempts.append(True)
        if stalled == "close":
            await block()
        await sock.close()

    async def expire_when_blocked(awaitable, timeout):
        phase = "send" if not bounds else "close"
        bounds.append((phase, timeout))
        task = asyncio.create_task(awaitable)
        try:
            if phase == ("close" if stalled == "close" else "send"):
                await await_until(entered, what=f"stop reaches stalled {stalled}")
                # Expire only after the bounded coroutine has reached the stall.
                task.cancel()
                await await_until(
                    asyncio.gather(task, return_exceptions=True), what="cancel stopped I/O"
                )
                raise asyncio.TimeoutError
            return await asyncio.wait_for(task, FALLBACK_TIMEOUT_SECS)
        finally:
            task.cancel()
            await await_until(
                asyncio.gather(task, return_exceptions=True), what="drain stopped I/O"
            )

    if stalled == "lock":
        conn._send_lock = HeldLock()
    conn._sender = send
    conn._closer = close
    # Only this module's timeout primitive changes; the test's hang guards stay real.
    monkeypatch.setattr(
        broker_mod,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "wait_for": expire_when_blocked}),
    )
    try:
        if entrypoint == "disconnect":
            result = await asyncio.wait_for(
                broker.disconnect("laptop", "access ended", refusal=refusal),
                FALLBACK_TIMEOUT_SECS,
            )
            assert result is True
        elif entrypoint == "shutdown":
            assert (
                await asyncio.wait_for(broker.shutdown_all("preview off"), FALLBACK_TIMEOUT_SECS)
                == 1
            )
        else:
            await asyncio.wait_for(conn.stop("access ended"), FALLBACK_TIMEOUT_SECS)
        assert bounds == [
            ("send", broker_mod._STOP_IO_TIMEOUT_SECS),
            ("close", broker_mod._STOP_IO_TIMEOUT_SECS),
        ]
        assert entered.is_set(), "the stalled phase must actually be attempted"
        assert close_attempts == [True], "a send timeout must not skip socket close"
        assert channel.state == STATE_INTERRUPTED
        assert broker.live("laptop") is None
        assert sock.closed is (stalled != "close")
    finally:
        release.set()


async def test_cancelling_stop_still_closes_the_socket():
    broker = ConnectionBroker()
    conn, sock = _connect(broker)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_send(text):
        entered.set()
        await await_until(release, what="release cancelled stop send")

    conn._sender = blocked_send
    task = asyncio.create_task(conn.stop("access ended"))
    try:
        await await_until(entered, what="stop enters its final send")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, FALLBACK_TIMEOUT_SECS)
        assert not conn.alive
        assert sock.closed
    finally:
        release.set()
        task.cancel()
        await await_until(asyncio.gather(task, return_exceptions=True), what="drain cancelled stop")
