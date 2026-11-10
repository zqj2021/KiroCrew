"""The session tree's boot replay reads each unit's log from where the last boot stopped.

The checkpoint keeps, per unit, a watermark: the newest segment it read and the byte
offset it read up to. A boot reads only the bytes after it and folds the decisions and
spawns found there, so the cost of a boot after a small append is the append, whatever
the size of the log behind it. A unit whose segments changed in any other way -- a new
segment, a shorter file, a replaced file -- is read in full, and only that unit.

Bytes are counted by wrapping ``open`` for files inside the store root, keyed by unit.
"""

from __future__ import annotations

import builtins
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from crew_log_drain import assert_drained

from kiro_crew import crew_log as lg
from kiro_crew.crew_log import CrewLog, emit
from kiro_crew.crew_log import session_tree_projection as stp
from kiro_crew.crew_log.session_tree_projection import SessionTreeProjection
from kiro_crew.crew_log.store import crew_log_root
from kiro_crew.session_ledger import _store_name

GATEWAY = "gateway"
#: Filler entries per unit for the four log sizes, 1x / 4x / 16x / 64x.
SIZES = (250, 1000, 4000, 16000)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, _floor_monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _floor_monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    _floor_monkeypatch.setenv(emit.CREW_LOG_ENV, "1")
    _floor_monkeypatch.setattr(
        "kiro_crew.executors.maintenance_executor",
        lambda: type("_NoPool", (), {"submit": staticmethod(lambda *a, **k: None)}),
    )
    stp.reset_for_tests()
    yield
    stp.reset_for_tests()
    emit.reset_caches()


class _CountingFile:
    """A binary file handle that adds every byte it hands out to *bucket*."""

    def __init__(self, handle: Any, bucket: Counter[str], key: str) -> None:
        self._handle = handle
        self._bucket = bucket
        self._key = key

    def _count(self, blob: bytes) -> bytes:
        self._bucket[self._key] += len(blob)
        return blob

    def read(self, *args: Any) -> bytes:
        return self._count(self._handle.read(*args))

    def read1(self, *args: Any) -> bytes:
        return self._count(self._handle.read1(*args))

    def readline(self, *args: Any) -> bytes:
        return self._count(self._handle.readline(*args))

    def readinto(self, buffer: Any) -> int:
        n = self._handle.readinto(buffer)
        self._bucket[self._key] += n or 0
        return n

    def __iter__(self) -> Iterator[bytes]:
        for line in self._handle:
            yield self._count(line)

    def __enter__(self) -> "_CountingFile":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._handle.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


@pytest.fixture
def bytes_read(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Bytes read from each unit's directory, keyed by the unit's directory name."""
    bucket: Counter[str] = Counter()
    real_open = builtins.open

    def counting_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(file, mode, *args, **kwargs)
        if "b" not in mode or "r" not in mode or not isinstance(file, (str, os.PathLike)):
            return handle
        path = Path(file)
        try:
            root = crew_log_root(lg.KIND_SESSION)
        except Exception:
            return handle
        if path.parent.parent != root:
            return handle
        return _CountingFile(handle, bucket, path.parent.name)

    monkeypatch.setattr(builtins, "open", counting_open)
    return bucket


def _unit(sid: str, slot: str, filler: int) -> CrewLog:
    handle = CrewLog.create(lg.KIND_SESSION, sid, owner="raymond", agent="kirocrew", slot=slot)
    handle.append(
        "session/opened",
        {
            "agent": "kirocrew",
            "slot": slot,
            "model": "opus",
            "cwd": "/w",
            "owner": "raymond",
            "resumed": False,
        },
        src=GATEWAY,
    )
    for turn in range(filler):
        handle.append("turn/started", {"turn": turn + 1, "actor": "user", "depth": 0}, src=GATEWAY)
    return handle


def _small_append(handle: CrewLog) -> None:
    handle.append("turn/started", {"turn": 0, "actor": "user", "depth": 0}, src=GATEWAY)


def _boot() -> SessionTreeProjection:
    """One gateway start: seed from the checkpoint on disk, then write it back."""
    projection = SessionTreeProjection()
    projection.ensure_seeded()
    projection.flush_checkpoint()
    projection.cancel_pending_checkpoint()
    return projection


def _dir(sid: str) -> Path:
    return crew_log_root(lg.KIND_SESSION) / _store_name(sid)


def _log_size(sid: str) -> int:
    return sum(p.stat().st_size for p in _dir(sid).glob("log*.jsonl"))


def _parent(projection: SessionTreeProjection, slot: str) -> str | None:
    node = projection.nodes().get(slot)
    return None if node is None else node.parent_slot


def test_a_boot_after_a_small_append_reads_only_the_append_at_every_log_size(
    bytes_read: Counter[str],
) -> None:
    handles = {f"s-{n}": _unit(f"s-{n}", f"slot-{n}", n) for n in SIZES}
    _boot()  # first start on this build: cold scan
    _boot()  # second start: replay that sets each unit's watermark
    for handle in handles.values():
        _small_append(handle)
    assert_drained()
    bytes_read.clear()

    _boot()

    read = {sid: bytes_read[_store_name(sid)] for sid in handles}
    sizes = {sid: _log_size(sid) for sid in handles}
    # One appended line is about 115 bytes. A full read of the smallest log is ~29 KB.
    # Flat up to the width of the seq number in the appended line.
    spread = max(read.values()) - min(read.values())
    assert spread <= 8, f"bytes read per unit should be flat: {read} of {sizes}"
    assert max(read.values()) < 1024, f"bytes read per unit: {read} of {sizes}"


def test_a_boot_with_nothing_appended_opens_no_unit_log(bytes_read: Counter[str]) -> None:
    _unit("s-a", "slot-a", SIZES[0])
    _boot()
    _boot()
    assert_drained()
    bytes_read.clear()

    _boot()

    assert bytes_read[_store_name("s-a")] == 0, dict(bytes_read)


@pytest.mark.parametrize(
    ("entry_type", "data", "expected"),
    [
        ("session/adopted", {"parent": {"slot": "slot-b"}}, "slot-b"),
        ("session/released", {}, None),
    ],
)
def test_a_decision_written_while_the_gateway_was_down_is_applied_from_the_delta(
    bytes_read: Counter[str], entry_type: str, data: dict[str, Any], expected: str | None
) -> None:
    adopted = _unit("s-a", "slot-a", SIZES[1])
    _unit("s-b", "slot-b", 1)
    adopted.append("session/adopted", {"parent": {"slot": "slot-c"}}, src=GATEWAY)
    _unit("s-c", "slot-c", 1)
    _boot()
    assert _parent(_boot(), "slot-a") == "slot-c"

    # Written by a process the next gateway did not see: no projection applied it.
    adopted.append(entry_type, data, src=GATEWAY)
    assert_drained()
    bytes_read.clear()

    booted = _boot()

    assert _parent(booted, "slot-a") == expected
    assert bytes_read[_store_name("s-a")] < 1024, dict(bytes_read)


def _rotate(sid: str) -> None:
    """Start a newer segment, the shape a rotating writer leaves."""
    directory = _dir(sid)
    head = (directory / "log.jsonl").read_bytes().splitlines(keepends=True)[0]
    (directory / "log.900000.jsonl").write_bytes(
        head + b'{"type":"turn/started","seq":900000,"time":1,"src":"gateway",'
        b'"data":{"turn":1,"actor":"user","depth":0}}\n'
    )


def _shrink(sid: str) -> None:
    """Cut the log back to an earlier line boundary."""
    path = _dir(sid) / "log.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[: len(lines) // 2]))


def _replace_inode(sid: str) -> None:
    """Same bytes, new file: what a copy-and-rename restore leaves."""
    path = _dir(sid) / "log.jsonl"
    copy = path.with_name("log.copy")
    copy.write_bytes(path.read_bytes())
    os.replace(copy, path)


@pytest.mark.parametrize("change", [_rotate, _shrink, _replace_inode])
def test_a_unit_whose_segments_changed_is_read_in_full_and_only_that_unit(
    bytes_read: Counter[str], change: Any
) -> None:
    _unit("s-changed", "slot-x", SIZES[1])
    steady = _unit("s-steady", "slot-y", SIZES[1])
    _boot()
    _boot()
    change("s-changed")
    _small_append(steady)
    assert_drained()
    bytes_read.clear()

    _boot()

    assert bytes_read[_store_name("s-changed")] >= _log_size("s-changed") // 2, dict(bytes_read)
    assert bytes_read[_store_name("s-steady")] < 1024, dict(bytes_read)


def test_a_fallback_read_finds_a_decision_in_a_replaced_file(bytes_read: Counter[str]) -> None:
    _unit("s-a", "slot-a", SIZES[0])
    _unit("s-b", "slot-b", 1)
    _boot()
    _boot()
    path = _dir("s-a") / "log.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    seq = len(lines)
    adopt = (
        b'{"type":"session/adopted","seq":%d,"time":5,"src":"gateway",'
        b'"data":{"parent":{"slot":"slot-b"}}}\n' % seq
    )
    # Inserted BEFORE the old end, so a delta read from the old offset cannot see it.
    rebuilt = path.with_name("log.copy")
    rebuilt.write_bytes(b"".join(lines[:2]) + adopt + b"".join(lines[2:]))
    os.replace(rebuilt, path)

    assert _parent(_boot(), "slot-a") == "slot-b"


def _children(projection: SessionTreeProjection) -> set[str]:
    return {spawn.slot for spawn in projection._spawned.values()}


def test_a_rolled_back_spawn_replaced_at_the_same_length_is_re_read() -> None:
    """A failed append is rolled back as a CUT, and the next append can bring the file
    back to the very length and inode the mark holds. The cut counter moved, so the
    unit is read in full: the rolled-back spawn goes and its replacement is found."""
    handle = _unit("s-a", "slot-a", SIZES[0])
    _boot()
    handle.append("session/spawned", {"child": {"slot": "W1"}}, src=GATEWAY)
    assert_drained()
    assert _children(_boot()) == {"W1"}

    path = _dir("s-a") / "log.jsonl"
    before = path.stat()
    lines = path.read_bytes().splitlines(keepends=True)
    replacement = lines[-1].replace(b'"W1"', b'"W2"')
    cuts = _dir("s-a") / ".cuts"
    settled = int(cuts.read_text())
    # What a rollback leaves: the line removed in place, the counter raised past it
    # (once before the cut and once after), then the next append of the same size.
    with open(path, "r+b") as raw:
        raw.truncate(before.st_size - len(lines[-1]))
        raw.seek(0, os.SEEK_END)
        raw.write(replacement)
    cuts.write_text(str(settled + 2))
    after = path.stat()
    assert (after.st_ino, after.st_size) == (before.st_ino, before.st_size)

    assert _children(_boot()) == {"W2"}


@pytest.mark.parametrize("child", ['"bad"', '{"slot":[]}'], ids=["not-a-map", "slot-not-a-str"])
def test_a_malformed_spawn_after_the_mark_does_not_abort_the_seed(child: str) -> None:
    _unit("s-a", "slot-a", SIZES[0])
    _boot()
    _boot()
    path = _dir("s-a") / "log.jsonl"
    seq = len(path.read_bytes().splitlines())
    with open(path, "ab") as raw:
        raw.write(
            b'{"type":"session/spawned","seq":%d,"time":5,"src":"gateway","data":{"child":%s}}\n'
            % (seq, child.encode())
        )

    booted = _boot()

    assert "slot-a" in booted.nodes()
    assert booted.reading().incomplete is False


def _adopted_unit() -> tuple[CrewLog, Path]:
    """slot-a adopted under slot-c, booted twice so the adoption is checkpointed and the
    unit holds a mark."""
    handle = _unit("s-a", "slot-a", SIZES[0])
    _unit("s-c", "slot-c", 1)
    handle.append("session/adopted", {"parent": {"slot": "slot-c"}}, src=GATEWAY)
    assert_drained()
    _boot()
    assert _parent(_boot(), "slot-a") == "slot-c"
    return handle, _dir("s-a") / "log.jsonl"


def test_a_complete_release_missing_its_newline_is_applied_and_read_again_once() -> None:
    """An interrupted append can leave a whole final record with no terminator. It is
    folded, as the tail reader folds it, and the mark stays before it, so the boot after
    the newline lands reads it once more and the tree does not move again."""
    _handle, path = _adopted_unit()
    seq = len(path.read_bytes().splitlines())
    release = b'{"type":"session/released","seq":%d,"time":9,"src":"gateway","data":{}}' % seq
    with open(path, "ab") as raw:
        raw.write(release)

    first = _boot()
    assert _parent(first, "slot-a") is None
    held = dict(first._edges)
    assert first._marks["s-a"][3] == path.stat().st_size - len(release)

    with open(path, "ab") as raw:
        raw.write(b"\n")
    second = _boot()
    assert _parent(second, "slot-a") is None
    assert second._edges == held
    assert second._marks["s-a"][3] == path.stat().st_size


def test_an_unparseable_partial_tail_is_left_for_the_next_read() -> None:
    _handle, path = _adopted_unit()
    size = path.stat().st_size
    with open(path, "ab") as raw:
        raw.write(b'{"type":"session/released","se')

    booted = _boot()

    assert _parent(booted, "slot-a") == "slot-c"
    assert booted.reading().incomplete is False
    assert booted._marks["s-a"][3] == size
