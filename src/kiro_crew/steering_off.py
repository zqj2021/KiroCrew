"""The steering files one chat has switched off: resolved paths, never bodies.

One file under the data home maps a chat's session key to the sorted resolved
paths it excludes. Context assembly reads :func:`steering_off_for` once per turn
and hands the set to the two writers Crew owns (the Claude Code ``[Steering
resources]`` load and the folder steering section), which skip those documents. A
chat with nothing switched off reads an empty set, so its turn text is unchanged.
The dashboard's per-chat steering route writes it.

The file is a log of whole records, one JSON object per line, and the newest line
that parses is the store. A save appends a line and never overwrites the record
before it, so a write cut short by a kill, a full disk or a power loss leaves an
unparseable last line and the previous record intact. Why a log on one inode and
not a rename: see :func:`_append_record`.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_FILE_NAME = "chat_steering_off.json"
# Past this size a save also folds the log down to its newest record.
_COMPACT_AT = 64 * 1024
# One gateway process writes this file; the lock keeps two toggles of one
# process from losing each other's read-modify-write, and keeps a reader from
# seeing an append half done.
_WRITE_LOCK = threading.Lock()


def _store_path() -> Path:
    from kiro_crew.config.paths import data_home

    return data_home() / _FILE_NAME


class SteeringOffStoreUnreadable(OSError):
    """The store exists but cannot be read, so a write would erase it."""


# The last record this process read or wrote whole, per store path, held under
# _WRITE_LOCK. A store that later cannot be read at all reads as this, never as
# "nothing switched off": an off switch must fail closed.
_LAST_GOOD: dict[str, dict[str, list[str]]] = {}
# Store paths whose unreadable record was already logged, so a turn loop logs once.
_WARNED_UNREADABLE: set[str] = set()


def _read_all() -> dict[str, list[str]]:
    """The store: the newest whole record, ``{}`` when the file is missing or empty.

    A torn last line (a save cut short) is skipped, so the record before it wins. A
    file with text but no whole record, or one that cannot be read, raises
    :class:`SteeringOffStoreUnreadable`, so a switch never overwrites every other
    chat's record with an empty one. The sandbox pre-creates the leaf as ``{}``.
    """
    try:
        text = _store_path().read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise SteeringOffStoreUnreadable(str(exc)) from exc
    if not text.strip():
        return {}
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            continue
        if isinstance(raw, dict):
            return {
                key: [p for p in paths if isinstance(p, str) and p]
                for key, paths in raw.items()
                if isinstance(key, str) and isinstance(paths, list)
            }
    raise SteeringOffStoreUnreadable("the store holds no whole record")


def steering_off_for(session_key: str | None) -> frozenset[str]:
    """The resolved paths *session_key* switched off; empty for none.

    A store that cannot be read reads as the last record this process held whole,
    logged once.
    """
    if not session_key:
        return frozenset()
    where = str(_store_path())
    # Under the write lock: a reader never sees an append half done.
    with _WRITE_LOCK:
        try:
            store = _read_all()
        except SteeringOffStoreUnreadable as exc:
            if where not in _WARNED_UNREADABLE:
                _WARNED_UNREADABLE.add(where)
                logger.warning(
                    "steering-off store unreadable (%s); keeping switched-off files off", exc
                )
            # shortcut: a process that never held a whole record (a restart onto a
            # file someone else damaged, or one it may not read) has nothing to keep
            # off, so it reads as nothing switched off; the next switch answers 409
            # store_unreadable, which makes it visible. A save Crew cut short never
            # gets here: its log still holds the record before it.
            return frozenset(_LAST_GOOD.get(where, {}).get(session_key, ()))
        _LAST_GOOD[where] = store
        _WARNED_UNREADABLE.discard(where)
        return frozenset(store.get(session_key, ()))


def set_steering_off(session_key: str, path: str, off: bool) -> frozenset[str]:
    """Switch one resolved *path* off (or back on) for *session_key*; the new set."""
    with _WRITE_LOCK:
        target = _store_path()
        store = _read_all()
        _LAST_GOOD[str(target)] = {key: list(paths) for key, paths in store.items()}
        current = set(store.get(session_key, ()))
        if off:
            current.add(path)
        else:
            current.discard(path)
        if current:
            store[session_key] = sorted(current)
        else:
            store.pop(session_key, None)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(store, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        _append_record(target, ("\n" + line + "\n").encode("ascii"))
        _LAST_GOOD[str(target)] = store
        return frozenset(current)


def _append_record(path: Path, record: bytes) -> None:
    """Append one whole *record* line to *path* on its own inode.

    Why one inode and not a rename: the leaf is a keystone sealed read-only in the
    sandbox (``sandbox._CREW_READONLY_LEAVES``). A running Linux sandbox bind-mounts
    this inode at the name, and a rename over a mount point in another namespace
    detaches that mount and would leave the new file writable from inside. So an
    existing record is extended on its inode, and an absent one is created with
    ``O_EXCL``. A symlink, reparse point or hard-linked record is refused, never
    followed. Callers hold ``_WRITE_LOCK``.

    The record opens with a newline, so it never joins a torn line before it. Past
    :data:`_COMPACT_AT` the log folds to its newest record: the record is written at
    offset 0 while its appended copy still ends the file, then the file is cut to
    it, so a cut-short fold leaves that copy as the newest whole line.
    """
    base = os.O_WRONLY | getattr(os, "O_BINARY", 0)
    base |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, base | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if stat.S_ISLNK(os.lstat(path).st_mode):
            raise OSError(f"refusing to rewrite {path.name}: it is a link") from None
        fd = os.open(path, base)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise OSError(f"refusing to rewrite {path.name}: not a single-link regular file")
        named = os.lstat(path)
        if (named.st_dev, named.st_ino) != (st.st_dev, st.st_ino):
            raise OSError(f"refusing to rewrite {path.name}: it changed while opening")
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        _write_all(fd, record, st.st_size)
        os.fsync(fd)
        size = st.st_size + len(record)
        if size > _COMPACT_AT and 2 * len(record) <= size:
            _write_all(fd, record, 0)
            os.fsync(fd)
            os.ftruncate(fd, len(record))
            os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes, offset: int) -> None:
    """Write all of *data* at *offset*, the way a short ``os.write`` needs."""
    os.lseek(fd, offset, os.SEEK_SET)
    while data:
        data = data[os.write(fd, data) :]
