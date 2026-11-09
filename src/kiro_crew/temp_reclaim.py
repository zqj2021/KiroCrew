"""Remove a temp tree from ``/tmp`` or ``/var/tmp`` once it is provably nobody's work.

``kirocrew tmp reclaim`` is the person's way to clean up what agent sessions left
in the system temp roots -- mostly fake ``HOME`` directories for test runs, each
carrying its own uv, npm and bytecode caches. An agent cannot do it from its own
shell: the deny rules refuse ``rm -rf`` on an absolute path, and the sandbox
shows it only the few processes inside its own user namespace, so nothing it
runs can tell whether a sibling session still uses a tree. The design record is
``docs/request-for-change/rfc-agent-temp-reclaim.md``; this module is its
Phase 1, the routine and the command. No agent surface reaches it yet.

REFUSE-SAFE IS THE WHOLE DESIGN. A path is removed only when every check below
proves it may go, and every doubt keeps it:

1. It is a plain directory or regular file, never a link.
2. It is a top-level entry of a temp root, ``/tmp/<name>``, owned by this user.
   Nothing deeper is accepted: a process can look things up below a directory
   it names, so with no directory between the temp root and the target, every
   such directory is inside the target or is the temp root itself. It is not a
   mount point and holds none: its device is the temp root's, and no entry of
   ``/proc/self/mountinfo`` is at or below it.
3. Nothing anywhere in it changed (``st_mtime`` or ``st_ctime``) in the last
   :data:`IDLE_SECONDS`.
4. It holds no git repository.
5. No process uses it. No working directory, root, executable, open file,
   mapped file, command-line argument or ``TMPDIR``/``TMP``/``TEMP`` value is
   inside it, no argument or other environment value (``PATH``,
   ``VIRTUAL_ENV``) mentions its path anywhere in its text, no working
   directory is the temp root itself (it looks the tree up by name), and no
   ``HOME``, ``PYTHONPATH`` entry, ``XDG_*`` or ``*_CACHE_DIR`` directory, nor
   the directory of a file an argument names, is inside it or holds it,
   because a process looks things up below those. A
   link the kernel prints with `` (deleted)`` counts under both readings. A
   relative or empty environment entry makes every tree unprovable: the process
   resolved it against the directory it started in, which ``/proc`` does not
   record. A relative argument is joined to the working directory instead,
   because nearly every process has one (a flag, a word), so treating it the
   same way would keep every tree on every host. A
   process of this user whose ``/proc`` entries cannot be read makes every tree
   unprovable, because it could be the one using it. The exception is a process
   the kernel marks not dumpable (ssh-agent, gpg-agent, a setuid sudo): it is
   skipped like another user's process and named in the output, and only its
   command line, which anyone may read, counts; one that runs with file
   capabilities stays unprovable. A tree that holds a socket
   file is always kept, because nothing in ``/proc`` ties a live socket to its
   file, so a socket such a process bound is covered too. A process in another
   mount namespace prints its own view of every path: its working directory,
   root, open and mapped files are also matched by device and inode. A mount
   any process sees, whatever its uid, whose root is the tree or inside it,
   is a use. A mount in another namespace's table whose root is the temp root
   or above it (``-v /tmp:/tmp``, a snap's view of the host) keeps every tree
   in that temp root while it runs: that namespace can reach each entry under
   spellings of its own, which are never translated. When that namespace's
   ``/`` is the same mount as here (point, device and root), as a systemd
   service's own namespace has, a mount it shares with this one does not count:
   it shows the entries under the same names. A container with a root of its
   own counts even for a ``/tmp`` mount identical to this one. A second mount
   of the temp root, or of a directory above it, in this namespace (a bind of
   ``/tmp`` on ``/var/tmp``) gives each tree a second name, checked like its own.
6. The removal is pinned to the inode the checks saw
   (:func:`pinned_fs.remove_tree_pinned`, or a verified unlink for a file).
   Right before it the change times are read again through the pinned descriptor and
   the process table is read again, and afterwards the path must still name
   that inode, so a swap, a rename, a write or a run started after the checks
   removes nothing. A failure keeps the path and prints why; it never stops the run.

Linux only: the in-use check reads ``/proc``, and without it nothing is proven.
"""

from __future__ import annotations

import os
import re
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TextIO

from kiro_crew import pinned_fs
from kiro_crew.agent_scratch import (
    _is_git_directory,
    _is_plain_dir,
    _regenerable,
    _search,
    _searched_venv,
)

#: The roots a person's leftovers live in. Anything outside them is kept.
TEMP_ROOTS: tuple[str, ...] = ("/tmp", "/var/tmp")
#: A tree changed more recently than this is kept as ``recent``.
IDLE_SECONDS = 30 * 60
#: The whole run stops starting new paths after this many seconds.
DEADLINE_SECONDS = 120.0
#: Environment variables whose value names one directory the process works in.
#: Also every ``XDG_*`` and ``*_CACHE_DIR`` not in :data:`_ENV_LIST_KEYS`.
_ENV_PATH_KEYS = frozenset({"HOME", "TMPDIR", "TMP", "TEMP"})
#: Environment variables whose value is a ``:``-separated list of directories.
_ENV_LIST_KEYS = frozenset({"PYTHONPATH", "XDG_DATA_DIRS", "XDG_CONFIG_DIRS"})
#: The two kinds of reference a process makes. A ``ROOT_REF`` is a directory the
#: process looks things up under, so a path inside it is in use too; a
#: ``PATH_REF`` protects itself and what is inside it, nothing above.
PATH_REF = "path"
ROOT_REF = "root"
#: The text of one command-line argument, or of an environment value no rule
#: below names. A tree whose path appears anywhere in it is in use, so
#: ``--directory=/tmp/site`` and ``PATH=/tmp/venv/bin`` count without parsing.
MENTION_REF = "mention"
#: A relative or empty environment entry. The process resolved it against the
#: directory it started in, which ``/proc`` does not record, so it could name
#: any tree and makes every tree unprovable.
UNPLACED_REF = "unplaced"
#: A working directory, recorded as a :data:`PATH_REF` too. One that is the
#: tree's temp root itself (``python3 -m http.server`` run from ``/tmp``) looks
#: things up in every top-level entry, so it is a use of each of them.
CWD_REF = "cwd"
#: The kernel appends this to a link whose target was deleted. A live directory
#: can also be named with it, so both readings are recorded.
_DELETED = " (deleted)"

REMOVED = "removed"
KEPT = "kept"
WOULD_REMOVE = "would-remove"


@dataclass(frozen=True)
class Outcome:
    """What happened to one requested path."""

    path: str
    outcome: str
    reason: str = ""


@dataclass(frozen=True)
class ProcessView:
    """Every path the user's live processes name, read once per run.

    ``uses`` holds ``(pid, comm, path)`` for each working directory, root,
    executable, open or mapped file, and ``roots`` the same for each directory
    an environment entry names (see :data:`ROOT_REF`). ``unreadable`` counts
    this user's processes whose entries could not be read, and ``unplaced``
    names each environment entry that cannot be placed on disk; either one
    makes every tree unprovable.
    """

    uses: tuple[tuple[int, str, str], ...]
    unreadable: int
    available: bool = True
    #: ``/proc`` is mounted to hide other users' processes (``hidepid``,
    #: ``subset=pid``), so their mounts of a tree cannot be seen.
    hidden: bool = False
    #: ``"<comm> <pid>"`` for each unreadable process, so a refusal can name them.
    unreadable_names: tuple[str, ...] = ()
    #: ``"<comm> <pid>"`` for each of this user's processes skipped as not dumpable.
    not_inspected: tuple[str, ...] = ()
    #: ``(pid, comm, directory)`` for each directory an environment entry names.
    roots: tuple[tuple[int, str, str], ...] = ()
    #: ``"pid <n> (<comm>) has <NAME> entry '<value>'"`` for each relative or
    #: empty environment entry (see :data:`UNPLACED_REF`).
    unplaced: tuple[str, ...] = ()
    #: ``(pid, comm, text)`` for each argument and unnamed environment value
    #: (see :data:`MENTION_REF`).
    mentions: tuple[tuple[int, str, str], ...] = ()
    #: ``(pid, comm, (st_dev, st_ino))`` for each working directory, root, open and
    #: mapped file of a process in another mount namespace (see :func:`_identities`).
    holds: tuple[tuple[int, str, tuple[int, int]], ...] = ()
    #: ``(pid, comm, maj:min, root)`` for each mount any process sees, whatever its
    #: uid or namespace: one whose root is the tree, or inside it, is a use.
    binds: tuple[tuple[int, str, str, str], ...] = ()
    #: The subset of :attr:`binds` read from a mount table other than this
    #: process's own, so from another mount namespace, less each mount it shares
    #: with this one when its ``/`` is the same mount as here (see :func:`_use_of`).
    foreign_binds: tuple[tuple[int, str, str, str], ...] = ()
    #: ``(pid, comm, directory)`` for each working directory (see :data:`CWD_REF`).
    cwds: tuple[tuple[int, str, str], ...] = ()


#: A run of ``/`` and ``/./`` in a mention's text, which names the same place as one ``/``.
_SQUASH = re.compile(r"/(?:\.?/)+")


def _lexical(path: str) -> str:
    """*path* with repeated ``/`` and ``.`` dropped, every ``..`` kept.

    ``/tmp/./site/../h`` reads ``/tmp/site/../h``: the kernel walks through
    ``/tmp/site`` to resolve the ``..``, which ``normpath`` would hide.
    """
    return "/" + "/".join(part for part in path.split("/") if part not in ("", "."))


def _env_paths(raw: bytes) -> list[tuple[str, str]]:
    """The directories one ``/proc/<pid>/environ`` blob names, as ``(kind, path)``.

    Only :data:`_ENV_LIST_KEYS` are split at ``:``; every other value is one
    directory, so ``TMPDIR=/tmp/run:cache`` stays whole. ``TMPDIR``, ``TMP`` and
    ``TEMP`` are :data:`PATH_REF`: they say where new temp files go, not that
    anything already under them is in use, and they are often ``/tmp`` itself.
    The rest are :data:`ROOT_REF`: ``HOME``, a ``PYTHONPATH`` entry, an ``XDG_*``
    or ``*_CACHE_DIR`` directory is where the process looks things up, so a
    package below it is in use before it is ever opened.

    Each absolute value counts as written, normalized and resolved, because it
    is whatever the parent wrote: ``/tmp//home/.``, a link into the tree, a link
    in the tree to elsewhere and ``/tmp/site/../h`` (lookups still pass through
    the tree) must all match. A relative or
    empty entry is :data:`UNPLACED_REF`, with ``"<NAME> entry '<value>'"`` as
    its text. Python resolves ``PYTHONPATH=deps``, and reads an empty entry as
    the working directory, once at startup, and ``/proc`` shows only the
    working directory the process has now. An ``XDG_*`` entry that is not
    absolute is skipped, as the XDG specification tells programs to do.
    Every other value is a :data:`MENTION_REF`, so ``PATH=/tmp/proj/venv/bin``
    keeps ``/tmp/proj`` without a list of names.
    """
    out: list[tuple[str, str]] = []
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if not sep or not value:
            continue
        # fsdecode, not utf-8 with "replace": a byte that is not UTF-8 must still
        # name the same file, the way os.readlink and os.path.realpath read it.
        name = os.fsdecode(key)
        text = os.fsdecode(value)
        if name in _ENV_LIST_KEYS:
            parts, kind = text.split(os.pathsep), ROOT_REF
        elif name in _ENV_PATH_KEYS:
            parts, kind = [text], PATH_REF if name != "HOME" else ROOT_REF
        elif name.startswith("XDG_") or (
            name.endswith("_CACHE_DIR") and not name.endswith("_NO_CACHE_DIR")
        ):
            # PIP_NO_CACHE_DIR=1 is pip's off switch, not a directory.
            parts, kind = [text], ROOT_REF
        else:
            # PATH, VIRTUAL_ENV, LD_LIBRARY_PATH...: any value naming the tree keeps it.
            out.append((MENTION_REF, text))
            continue
        for part in parts:
            if part.startswith("/"):
                # As written, normalized and resolved: HOME=/tmp/site/../h and
                # HOME=/tmp/site/home linking to /home/user both still look things
                # up through /tmp/site.
                spellings = (_lexical(part), os.path.normpath(part), os.path.realpath(part))
                for path in dict.fromkeys(spellings):
                    out.append((kind, path))
            elif not name.startswith("XDG_"):
                out.append((UNPLACED_REF, f"{name} entry {part!r}"))
    return out


def _status_uids(proc: Path) -> tuple[int, int]:
    """The real and effective uid from ``/proc/<pid>/status``.

    Not the owner of a ``/proc/<pid>`` entry: the kernel hands those to root for
    any process that is not dumpable (ssh-agent, gpg-agent, a setuid program),
    which would make this user's own process look like someone else's and skip
    it. ``status`` is world-readable and always names the real owner.
    """
    # Bytes, split only at LF: the Name line carries the process's own comm, which
    # may hold CR or U+2028, and text-mode reading turns CR into a line break.
    for line in (proc / "status").read_bytes().split(b"\n"):
        if line.startswith(b"Uid:"):
            fields = line.split()
            return int(fields[1]), int(fields[2])
    raise ProcessLookupError(str(proc))


def _has_capabilities(proc: Path) -> bool:
    """Whether ``/proc/<pid>/status`` gives the process any permitted capability.

    A program run with file capabilities (``setcap cap_net_bind_service=+ep``) is
    not dumpable either, but unlike ssh-agent it can be an ordinary server.
    """
    for line in (proc / "status").read_bytes().split(b"\n"):
        if line.startswith(b"CapPrm:"):
            return int(line.split()[1], 16) != 0
    return False


def _leader_exited(proc: Path) -> bool:
    """Whether ``/proc/<pid>/status`` shows a main thread that exited while others run.

    After ``pthread_exit()`` in the main thread the leader is a zombie: the
    kernel hands its ``fd`` directory to root and empties its command line,
    exactly as for a not-dumpable process, while the surviving threads still
    work in their own directories.
    """
    fields = {}
    for line in (proc / "status").read_bytes().split(b"\n"):
        key, _sep, value = line.partition(b":")
        fields[key] = value.split()
    state, threads = fields.get(b"State", [b""]), fields.get(b"Threads", [b"1"])
    return state[:1] == [b"Z"] and int(threads[0]) > 1


def _hides_processes(proc_root: Path) -> bool:
    """Whether the ``proc`` mount at *proc_root* hides other users' processes.

    ``hidepid=1``/``2`` (or ``noaccess``/``invisible``) and ``subset=pid`` keep a
    rootful container's processes out of the listing, and with them the mount
    of a tree they use. Read from :data:`_MOUNTINFO`; an unreadable table
    answers True.
    """
    try:
        raw = _MOUNTINFO.read_bytes()
    except OSError:
        return True
    for line in os.fsdecode(raw).split("\n"):
        fields, _sep, tail = line.partition(" - ")
        parts, described = fields.split(" "), tail.split(" ")
        if len(parts) < 5 or len(described) < 3 or described[0] != "proc":
            continue
        if parts[4] != str(proc_root):
            continue
        for option in described[2].split(","):
            # Every hidepid mode but the default hides something (1, 2, 4,
            # noaccess, invisible, ptraceable), and any subset leaves processes out.
            key, _eq, value = option.partition("=")
            if (key == "hidepid" and value not in ("0", "off")) or key == "subset":
                return True
    return False


def _dir_owner(proc: Path) -> int:
    """The owner of ``/proc/<pid>/fd``: root for a process the kernel marks not dumpable.

    Not ``/proc/<pid>`` itself: the kernel keeps that one directory owned by the
    process's effective uid whatever its dumpable state, so ``ssh-agent`` would
    look readable and then refuse every link.
    """
    return (proc / "fd").stat().st_uid


def _readings(name: str) -> list[str]:
    """*name* as the kernel printed it, plus without :data:`_DELETED` when it ends so.

    The kernel marks a deleted target by appending the suffix, and a directory
    can carry the same text in its real name, so the two cannot be told apart.
    """
    return [name, name.removesuffix(_DELETED)] if name.endswith(_DELETED) else [name]


def _map_paths(raw: str) -> list[str]:
    """The file paths in one ``/proc/<pid>/maps`` text (column six, absolute only).

    The kernel writes a newline in a mapped path as ``\\012``, and a file can
    carry that text in its real name, so both readings are kept, each also
    with and without :data:`_DELETED`.
    """
    out: list[str] = []
    for line in raw.split("\n"):
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            name = fields[5]
            out.extend(_readings(name))
            if "\\012" in name:
                out.extend(_readings(name.replace("\\012", "\n")))
    return out


def _argument_paths(raw: bytes, cwd: str | None) -> list[tuple[str, str]]:
    """What the arguments in one ``/proc/<pid>/cmdline`` name, as ``(kind, path)``.

    A program started on a file often names it nowhere else once it has read
    it: ``python /tmp/job/main.py`` closes the script, then imports ``helper``
    from ``/tmp/job``. So every argument is a :data:`MENTION_REF`, and also a
    :data:`PATH_REF`, a relative one joined to the working directory, and one
    that names a file makes that file's directory a :data:`ROOT_REF`. Each path
    counts as written and resolved: ``/tmp/venv/bin/python`` resolves out of
    the venv it runs from. An argument that is not a path names nothing, or
    keeps a tree it happens to spell.
    """
    out: list[tuple[str, str]] = []
    for arg in raw.split(b"\0"):
        text = os.fsdecode(arg)
        if text:
            out.append((MENTION_REF, text))
        # --directory=/home/me/site, or --directory=site from /tmp, names a path
        # after its "=", which may be a link into the tree or relative, so that
        # value counts like an argument of its own.
        _option, glued, value = text.partition("=")
        for word in (text, value) if glued and value else (text,):
            if not word.startswith("/"):
                if not word or cwd is None:
                    continue
                word = os.path.join(cwd, word)
            for path in {_lexical(word), os.path.normpath(word), os.path.realpath(word)}:
                out.append((PATH_REF, path))
                if os.path.isfile(path):
                    out.append((ROOT_REF, os.path.dirname(path)))
    return out


def _process_paths(proc: Path) -> list[tuple[str, str]]:
    """Every path one process names, as ``(kind, path)``.

    The working directory, root, executable, open and mapped files are
    :data:`PATH_REF`; command-line arguments and environment entries carry
    their own kind (see :func:`_argument_paths` and :func:`_env_paths`).
    Raises ``PermissionError`` when the process cannot be read.
    """
    try:
        cwd: str | None = os.readlink(proc / "cwd")
    except FileNotFoundError:
        cwd = None
    links = [] if cwd is None else [cwd]
    for link in ("root", "exe"):
        try:
            links.append(os.readlink(proc / link))
        except FileNotFoundError:
            continue
    for fd in os.listdir(proc / "fd"):
        try:
            links.append(os.readlink(proc / "fd" / fd))
        except FileNotFoundError:
            continue
    paths = [(PATH_REF, p) for link in links for p in _readings(link)]
    if cwd is not None:
        paths.extend((CWD_REF, p) for p in _readings(cwd))
    maps = os.fsdecode((proc / "maps").read_bytes())
    paths.extend((PATH_REF, p) for p in _map_paths(maps))
    paths.extend(_argument_paths((proc / "cmdline").read_bytes(), cwd))
    paths.extend(_env_paths((proc / "environ").read_bytes()))
    return paths


def _identities(proc: Path) -> list[tuple[int, int]]:
    """``(st_dev, st_ino)`` of the working directory, root, open and mapped files of *proc*.

    For a process in another mount namespace, such as a rootless container with
    ``-v /tmp/models:/models``, every path ``/proc`` prints is its own view
    (``/models``), which names nothing here. ``stat`` follows the same links to
    the files themselves, and ``maps`` carries each mapping's device and inode.
    """
    out: list[tuple[int, int]] = []
    links = [proc / "cwd", proc / "root", *(proc / "fd" / fd for fd in os.listdir(proc / "fd"))]
    for link in links:
        try:
            held = os.stat(link)
        except FileNotFoundError:
            continue
        out.append((held.st_dev, held.st_ino))
    for line in (proc / "maps").read_bytes().split(b"\n"):
        fields = line.split(None, 5)
        if len(fields) >= 5 and fields[4] != b"0":
            major, minor = fields[3].split(b":")
            out.append((os.makedev(int(major, 16), int(minor, 16)), int(fields[4])))
    return out


def read_processes(proc_root: Path = Path("/proc"), uid: int | None = None) -> ProcessView:
    """Read what every live process of *uid* (default: this user) names.

    A process counts as this user's when its real or effective uid is *uid*.
    This process itself is skipped: running the command from inside a tree, or
    with ``HOME`` pointing into it, says nothing about whether anything else
    uses it. A process of another uid is skipped, because the ownership check
    already limits removal to this user's own trees, and so is one of this
    user's whose ``/proc/<pid>/fd`` the kernel gave to another owner because it
    is not dumpable; those are listed in ``not_inspected``, and only their
    command line counts. One not dumpable because it runs with file
    capabilities is counted as unreadable instead. A process that vanishes
    mid-read is skipped. One of this user's that still cannot be read, which is
    what another user namespace does, is counted and named, and makes every
    tree unprovable. A process in another mount namespace is also matched by
    file identity (see :func:`_identities`), and every process's mount table,
    whatever its uid, is read for a mount of the tree.
    """
    if not (proc_root / "self").exists():
        return ProcessView(uses=(), unreadable=0, available=False)
    if _hides_processes(proc_root):
        return ProcessView(uses=(), unreadable=0, hidden=True)
    uid = os.getuid() if uid is None else uid
    me = os.getpid()
    try:
        namespace: str | None = os.readlink("/proc/self/ns/mnt")
    except OSError:
        namespace = None  # no mount namespaces here (WSL1): every process shares this view
    holds: list[tuple[int, str, tuple[int, int]]] = []
    binds: list[tuple[int, str, str, str]] = []
    foreign_binds: list[tuple[int, str, str, str]] = []
    uses: list[tuple[int, str, str]] = []
    roots: list[tuple[int, str, str]] = []
    mentions: list[tuple[int, str, str]] = []
    cwds: list[tuple[int, str, str]] = []
    unplaced: list[str] = []
    unreadable: list[str] = []
    skipped: list[str] = []
    tables: dict[bytes, int] = {}
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        proc = proc_root / name
        try:
            # Every process's, whatever its uid: anyone may read it, and a container
            # running as another uid (rootful, or a subuid) still mounts the tree.
            tables.setdefault((proc / "mountinfo").read_bytes(), int(name))
        except OSError:
            pass
        if int(name) == me:
            continue
        try:
            uids = _status_uids(proc)
            if uid not in uids:
                continue
            comm = os.fsdecode((proc / "comm").read_bytes()).removesuffix("\n")
            dir_owner = _dir_owner(proc)
            capable = dir_owner != uid and uids[1] == uid and _has_capabilities(proc)
            orphaned = dir_owner != uid and _leader_exited(proc)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            unreadable.append(f"? {name}")
            continue
        if capable or orphaned:
            # Not dumpable because it runs with file capabilities: an ordinary
            # program that may be serving a tree, whose links cannot be read. Or
            # its main thread exited and the threads still running cannot be told.
            unreadable.append(f"{comm} {name}")
            continue
        try:
            if dir_owner != uid:
                # The kernel gives root a not-dumpable process's /proc entries
                # (ssh-agent, gpg-agent, a setuid sudo). Its links cannot be read, and
                # it is treated like another user's process: skipped, and named in the
                # output. Its command line, which anyone may read, still counts. A
                # process hidden by another user namespace keeps this user as owner and
                # still counts as unreadable below.
                skipped.append(f"{comm} {name}")
                paths = _argument_paths((proc / "cmdline").read_bytes(), None)
            else:
                paths = _process_paths(proc)
                if namespace is not None and os.readlink(proc / "ns" / "mnt") != namespace:
                    holds.extend((int(name), comm, held) for held in _identities(proc))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            if dir_owner == uid:
                unreadable.append(f"{comm} {name}")
            continue
        for kind, path in paths:
            if kind == UNPLACED_REF:
                unplaced.append(f"pid {name} ({comm}) has {path}")
            elif kind == ROOT_REF:
                roots.append((int(name), comm, path))
            elif kind == MENTION_REF:
                mentions.append((int(name), comm, path))
            elif kind == CWD_REF:
                cwds.append((int(name), comm, path))
            else:
                uses.append((int(name), comm, path))
    try:
        own: bytes | None = _MOUNTINFO.read_bytes()
    except OSError:
        own = None  # no own table to tell a foreign one by: every table counts as foreign
    here = _mount_table(own) if own is not None else []
    for table, pid in tables.items():
        try:
            owner = os.fsdecode((proc_root / str(pid) / "comm").read_bytes()).removesuffix("\n")
        except OSError:
            owner = "?"
        entries = _mount_table(table)
        binds.extend((pid, owner, dev, root) for _at, dev, root in entries)
        if table != own:
            # A namespace that resolves "/" through the same mount as here (a systemd
            # service's) sees each mount it shares with this one under the same names.
            # A container with a root of its own does not, even for an identical -v /tmp:/tmp.
            same = set(here) if _root_mount(entries) == _root_mount(here) else set()
            foreign_binds.extend(
                (pid, owner, dev, root) for at, dev, root in entries if (at, dev, root) not in same
            )
    return ProcessView(
        uses=tuple(uses),
        unreadable=len(unreadable),
        unreadable_names=tuple(sorted(unreadable)),
        not_inspected=tuple(sorted(skipped)),
        roots=tuple(roots),
        unplaced=tuple(unplaced),
        mentions=tuple(mentions),
        holds=tuple(holds),
        binds=tuple(binds),
        foreign_binds=tuple(foreign_binds),
        cwds=tuple(cwds),
    )


def _inside(path: str, tree: str) -> bool:
    return path == tree or path.startswith(tree.rstrip("/") + "/")


def _temp_root(resolved: Path, roots: Sequence[str]) -> Path | None:
    """The temp root *resolved* sits strictly inside, or ``None``."""
    for root in roots:
        real = Path(os.path.realpath(root))
        if resolved != real and _inside(str(resolved), str(real)):
            return real
    return None


def _holds_repository(tree: Path) -> bool:
    """:func:`agent_scratch.holds_git_repository`, with a top-level ``.cache/uv`` skipped.

    uv keeps a ``.git`` inside ``.cache/uv/sdists-v10``, so every fake ``HOME``
    that ran uv would read as a repository and could never be reclaimed. Only
    ``.cache/uv`` is skipped: a clone anywhere else under ``.cache`` is still
    found. Below the first two levels the sweep's own search runs unchanged
    (:func:`agent_scratch._search`), and every doubt answers True, as it does there.
    """
    try:
        if _level_holds_repository(tree, skip=None):
            return True
        cache = tree / ".cache"
        return _is_plain_dir(cache) and _level_holds_repository(cache, skip="uv")
    except OSError:
        return True


def _level_holds_repository(directory: Path, *, skip: str | None) -> bool:
    """One level of the search: *directory* itself, then each child the sweep searches.

    ``.cache`` is never searched from the top level here, because the caller
    searches it on its own with ``uv`` skipped. Raises ``OSError`` on an
    unreadable directory; the caller turns that into True.
    """
    if _is_git_directory(directory):
        return True
    for entry in os.scandir(directory):
        child = directory / entry.name
        if entry.name == ".git":
            return True
        if entry.name == skip or (skip is None and entry.name == ".cache"):
            continue
        if not _is_plain_dir(child):
            continue
        if _regenerable(child, top_level=False) and not _searched_venv(child, top_level=False):
            continue
        if _search(child, basetemp_at_top=False):
            return True
    return False


#: This process's mount table. Removal runs in this mount namespace, so a mount
#: another namespace sees is not one the unlinks could reach.
_MOUNTINFO = Path("/proc/self/mountinfo")


def _mount_table(raw: bytes) -> list[tuple[str, str, str]]:
    """``(mount point, maj:min, root)`` for each line of a ``mountinfo`` text (fields 5, 3, 4).

    The kernel writes a space, tab, newline or backslash in a path as a
    three-digit octal escape (``\\040``), which is decoded here.
    """
    table: list[tuple[str, str, str]] = []
    # Only at LF: the kernel escapes that one, and str.splitlines would also cut a
    # path at CR, U+2028 or any other line boundary the kernel leaves as is.
    for line in os.fsdecode(raw).split("\n"):
        fields = line.split(" ")
        if len(fields) > 4:
            point, root = (
                re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)
                for field in (fields[4], fields[3])
            )
            table.append((point, fields[2], root))
    return table


def _root_mount(table: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    """The mount in effect at ``/`` in *table* (the last one listed there), or ``None``."""
    return next((entry for entry in reversed(table) if entry[0] == "/"), None)


def _mount_points() -> list[str] | None:
    """Every mount point in :data:`_MOUNTINFO`, or ``None`` when unreadable."""
    try:
        return [point for point, _dev, _root in _mount_table(_MOUNTINFO.read_bytes())]
    except OSError:
        return None


def _where(path: str) -> tuple[str, str] | None:
    """``(maj:min, path inside that filesystem)`` for *path*, or ``None`` when unreadable.

    The last mount in :data:`_MOUNTINFO` whose point holds *path* is the one in
    effect, so a bind mount of ``/tmp/site`` on ``/srv`` in another namespace
    shows the same pair as ``/tmp/site`` does here.
    """
    try:
        table = _mount_table(_MOUNTINFO.read_bytes())
    except OSError:
        return None
    found = None
    for point, device, root in table:
        if _inside(path, point) and (found is None or len(point) >= len(found[0])):
            found = (point, device, root)
    if found is None:
        return None
    point, device, root = found
    return device, root.rstrip("/") + path[len(point.rstrip("/")) :] or "/"


def _mount_doubt(resolved: Path, device: int, root: Path) -> str:
    """A reason to keep *resolved* when it is, or holds, a mount point, or ``""``.

    A mount owned by this user, such as ``rclone mount`` or ``sshfs -o
    idmap=user``, passes every other check, and the pinned walk takes its
    device from the target itself, so the whole mounted filesystem would go. A
    different device from the temp root is a mount; a bind mount of the same
    filesystem keeps the device, so the mount table is read too. A mount below
    the target counts as well: the pinned walk refuses another device but not
    a bind mount of this one.
    """
    try:
        if os.stat(root).st_dev != device:
            return "outside-scope: a mount point"
    except OSError as exc:
        return f"refused: {exc}"
    points = _mount_points()
    if points is None:
        return "unprovable: the mount table could not be read"
    tree = str(resolved)
    for point in points:
        if _inside(point, tree):
            return (
                "outside-scope: a mount point"
                if point == tree
                else f"outside-scope: holds mount point {point}"
            )
    return ""


def _user(path: Path, uid: int) -> bool:
    try:
        return os.lstat(path).st_uid == uid
    except OSError:
        return False


def _named_use(tree: str, view: ProcessView) -> str:
    """The first process whose paths, lookup roots, mentions or working directory name *tree*."""
    for pid, comm, path in view.uses:
        if _inside(path, tree):
            return f"in-use: pid {pid} ({comm})"
    for pid, comm, root in view.roots:
        if _inside(tree, root) or _inside(root, tree):
            return f"in-use: pid {pid} ({comm})"
    for pid, comm, text in view.mentions:
        # Also with repeated "/" and "/./" squashed: --directory=/tmp/./site.
        if tree in text or tree in _SQUASH.sub("/", text):
            return f"in-use: pid {pid} ({comm})"
    # shortcut: only the temp root itself counts as a working directory above the
    # tree, never ``/`` or ``/var``; widen this if a real use from there turns up.
    parent = os.path.dirname(tree)
    for pid, comm, cwd in view.cwds:
        if cwd == parent:
            return f"in-use: pid {pid} ({comm}) works in {parent}"
    return ""


def _second_names(tree: str) -> list[str] | None:
    """The other paths this namespace reaches *tree* by, or ``None`` when unreadable.

    A bind of ``/tmp`` on ``/var/tmp`` (a CIS hardening step) shows ``/tmp/site``
    as ``/var/tmp/site`` too, and a process working there prints that name. Each
    mount of the tree's filesystem whose root holds the tree gives one name; the
    one that is the tree's own path is left out.
    """
    located = _where(tree)
    try:
        table = _mount_table(_MOUNTINFO.read_bytes())
    except OSError:
        return None
    if located is None:
        return None
    device, inner = located
    names = (
        point.rstrip("/") + inner[len(root.rstrip("/")) :] or "/"
        for point, dev, root in table
        if dev == device and _inside(inner, root)
    )
    return [name for name in names if name != tree]


def _use_of(tree: str, view: ProcessView, inodes: frozenset[tuple[int, int]]) -> str:
    """The first process using *tree*, as a reason, or ``""``.

    A path a process names is a use when it is inside *tree*. A directory it
    looks things up under (``view.roots``) is a use when either holds the
    other: ``PYTHONPATH=/tmp/deps`` imports from ``/tmp/deps/pkg`` before it
    opens a file there, and ``HOME=/tmp/h/home`` lives inside ``/tmp/h``. A
    mention (see :data:`MENTION_REF`) is a use when the tree's path appears
    anywhere in it, even as the start of a longer name: in doubt, the tree is
    kept. A file a process in another mount namespace holds is a use when it is
    one of *inodes*, the ``(st_dev, st_ino)`` of every entry in the tree, and so
    is a mount any process sees whose root is the tree or inside it
    (``-v /tmp/site:/site``). A working directory that is the tree's temp root,
    and for a process in another mount namespace a held directory that is the
    temp root (``-v /tmp:/work`` with its working directory ``/work``), is a
    use too: the process looks things up below it. So is any mount in another
    namespace's table (``view.foreign_binds``) whose root is the temp root or
    above it, unless that namespace resolves ``/`` through the same mount as
    here and the mount is one it shares with this namespace. Every name this
    namespace reaches *tree* by (:func:`_second_names`) is checked like *tree*.
    """
    names = _second_names(tree)
    if names is None:
        return "unprovable: the mount table could not be read"
    for name in (tree, *names):
        reason = _named_use(name, view)
        if reason:
            return reason
    parent = os.path.dirname(tree)
    top: tuple[int, int] | None = None
    if view.holds:
        try:
            info = os.stat(parent)
        except OSError:
            return f"unprovable: {parent} could not be read"
        top = (info.st_dev, info.st_ino)
    for pid, comm, held in view.holds:
        if held in inodes or held == top:
            return f"in-use: pid {pid} ({comm})"
    located = _where(tree) if view.binds else None
    for pid, comm, device, root in view.binds:
        if located is None:
            return "unprovable: the mount table could not be read"
        if device == located[0] and _inside(root, located[1]):
            return f"in-use: pid {pid} ({comm})"
    # Fail closed: a namespace that mounts the temp root, or a directory above it
    # (-v /tmp:/tmp, a snap's host filesystem), can reach every entry under any
    # path, link or argument spelling, so none of them is provably unused while
    # it runs. Its own spellings are never translated. A mount it shares with this
    # namespace, seen through the same "/", is not in foreign_binds (read_processes).
    above = _where(parent) if view.foreign_binds else None
    for pid, comm, device, root in view.foreign_binds:
        if above is None:
            return "unprovable: the mount table could not be read"
        if device == above[0] and _inside(above[1], root):
            return f"in-use: pid {pid} ({comm}) can see {parent} from another mount namespace"
    return ""


def _unproven(view: ProcessView) -> str:
    """Why *view* cannot prove a tree unused, or ``""`` when it can."""
    if not view.available:
        return "unprovable: no /proc to check process use"
    if view.hidden:
        return "unprovable: /proc hides other users' processes (hidepid), so their mounts cannot be read"
    if view.unreadable:
        names = ", ".join(view.unreadable_names[:3])
        more = ", ..." if view.unreadable > 3 else ""
        detail = f" ({names}{more})" if names else ""
        noun = "process" if view.unreadable == 1 else "processes"
        return f"unprovable: {view.unreadable} {noun} unreadable{detail}"
    if view.unplaced:
        return (
            f"unprovable: {view.unplaced[0]}, resolved against the directory it "
            "started in, which /proc does not record"
        )
    return ""


#: Nothing readable without privilege ties a live socket to its file:
#: ``/proc/net/unix`` keeps the name a socket was bound by, which may be relative
#: or may have moved, and a descriptor reads only ``socket:[inode]``. So any tree
#: holding a socket file is kept, and a tree with none cannot hold a bound socket.
_SOCKET_REASON = "unprovable: holds a socket file, and /proc cannot say whether it is live"


def _blocker(resolved: Path, view: ProcessView, inodes: frozenset[tuple[int, int]]) -> str:
    """The first reason *view* gives to keep *resolved*, or ``""``."""
    return _unproven(view) or _use_of(str(resolved), view, inodes)


def _changed(info: os.stat_result) -> float:
    """When *info*'s entry last changed: ``tar``, ``cp -a`` and ``mv`` put an old
    mtime back, but each also sets ctime to now, and nothing can set that back."""
    return max(info.st_mtime, info.st_ctime)


def _pinned_scan(root_fd: int) -> tuple[float, bool, frozenset[tuple[int, int]]]:
    """``(newest change, holds a socket file, every entry's (st_dev, st_ino))`` under *root_fd*.

    Read through descriptors only: each directory is opened relative to its
    parent's descriptor with ``O_NOFOLLOW``, so a tree renamed after it was
    pinned is still the tree walked, and no link is followed. An entry that
    cannot be read answers ``(+inf, True, ...)``, which keeps it.
    """
    # Depth-first, one open descriptor per level: each frame is a directory's
    # descriptor and the subdirectory names still to visit, and a subdirectory is
    # opened only when it is visited, so a wide directory costs no descriptors.
    frames: list[tuple[int, list[str]]] = []
    socket = False
    try:
        top = os.fstat(root_fd)
        newest, inodes = _changed(top), {(top.st_dev, top.st_ino)}
        fd = os.dup(root_fd)
        while True:
            subdirs: list[str] = []
            frames.append((fd, subdirs))
            for name in os.listdir(fd):
                entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
                newest = max(newest, _changed(entry))
                inodes.add((entry.st_dev, entry.st_ino))
                socket = socket or stat.S_ISSOCK(entry.st_mode)
                if stat.S_ISDIR(entry.st_mode):
                    subdirs.append(name)
            while frames and not frames[-1][1]:
                os.close(frames.pop()[0])
            if not frames:
                break
            parent, pending = frames[-1]
            fd = os.open(pending.pop(), pinned_fs.dir_flags(), dir_fd=parent)
    except OSError:
        return float("inf"), True, frozenset()
    finally:
        for fd, _names in frames:
            os.close(fd)
    return newest, socket, frozenset(inodes)


def _scan(tree: Path) -> tuple[float, bool, frozenset[tuple[int, int]]]:
    """:func:`_pinned_scan` of the directory *tree*, opened without following a link."""
    try:
        fd = os.open(tree, pinned_fs.dir_flags())
    except OSError:
        return float("inf"), True, frozenset()
    try:
        return _pinned_scan(fd)
    finally:
        os.close(fd)


def _remove(
    resolved: Path,
    info: os.stat_result,
    *,
    now: float,
    reread: Callable[[], ProcessView],
) -> Outcome:
    """Remove *resolved*, pinned to the ``(st_dev, st_ino, st_uid)`` the checks saw.

    The process table the checks used was read once for the whole run, so the
    removal re-reads it, and re-reads the change times, right before acting: a run
    started on the path since then keeps it. For a directory that happens inside
    the pinned approval; for a file, under a fresh ``lstat`` just before the
    pinned unlink. Every failure, an ``OSError`` from the pinned walk included,
    is reported as a kept path, never raised.
    """
    if stat.S_ISREG(info.st_mode):
        return _remove_file(resolved, info, now=now, reread=reread)

    def _approve(root_fd: int, _tree: object) -> str | None:
        opened = os.fstat(root_fd)
        same = (opened.st_dev, opened.st_ino, opened.st_uid) == (
            info.st_dev,
            info.st_ino,
            info.st_uid,
        )
        if not same:
            return "not the directory checked"
        # The process table first, the change times after it: a writer that
        # exits before the table is read has already left its change time.
        view = reread()
        newest, socket, inodes = _pinned_scan(root_fd)
        if now - newest < IDLE_SECONDS:
            return "written since the check"
        if socket:
            return _SOCKET_REASON
        # A bind mount made inside the tree since the checks shares its device, so
        # the pinned walk would follow it; the mount table is read again.
        blocked = (
            _unproven(view)
            or _use_of(str(resolved), view, inodes)
            or _mount_doubt(resolved, opened.st_dev, resolved.parent)
        )
        if blocked:
            return blocked
        # The process table names paths, so it speaks for this tree only if the path
        # still names it after the read. A tree renamed since it was pinned is kept;
        # a process working in it under the new name would not have matched.
        try:
            here = os.lstat(resolved)
        except OSError:
            return "moved during the check"
        if (here.st_dev, here.st_ino) != (opened.st_dev, opened.st_ino):
            return "moved during the check"
        return None

    try:
        result = pinned_fs.remove_tree_pinned(
            str(resolved), what="temp tree", approve=_approve, refusal=pinned_fs.PinnedPathRefusal
        )
    except (pinned_fs.PinnedPathRefusal, OSError) as exc:
        return Outcome(str(resolved), KEPT, f"refused: {exc}")
    if result.removed:
        return Outcome(str(resolved), REMOVED)
    codes = {
        pinned_fs.REMOVAL_FAILED,
        pinned_fs.REMOVAL_IDENTITY_CHANGED,
        pinned_fs.REMOVAL_STAGE_FAILED,
        pinned_fs.REMOVAL_UNVERIFIABLE,
    }
    if result.reason not in codes:
        # The approval's own reason: nothing was touched.
        return Outcome(str(resolved), KEPT, result.reason or "refused")
    where = f", left at {resolved.parent / result.staged_name}" if result.staged_name else ""
    if result.reason == pinned_fs.REMOVAL_FAILED and result.survivors:
        return Outcome(str(resolved), KEPT, f"partial: {result.survivors} left{where}")
    return Outcome(str(resolved), KEPT, f"refused: {result.reason}{where}")


def _remove_file(
    resolved: Path,
    info: os.stat_result,
    *,
    now: float,
    reread: Callable[[], ProcessView],
) -> Outcome:
    """Unlink the regular file *resolved* after the same fresh checks a directory gets."""
    # The process table before the change time, as in :func:`_remove`.
    view = reread()
    try:
        current = os.lstat(resolved)
    except OSError as exc:
        return Outcome(str(resolved), KEPT, f"refused: {exc}")
    if (current.st_dev, current.st_ino, current.st_uid) != (
        info.st_dev,
        info.st_ino,
        info.st_uid,
    ):
        return Outcome(str(resolved), KEPT, "changed: not the file checked")
    if now - _changed(current) < IDLE_SECONDS:
        return Outcome(str(resolved), KEPT, "written since the check")
    blocked = _blocker(resolved, view, frozenset({(current.st_dev, current.st_ino)}))
    if blocked:
        return Outcome(str(resolved), KEPT, blocked)
    try:
        ok = pinned_fs.unlink_verified_by_name(
            resolved.parent, resolved.name, (info.st_dev, info.st_ino)
        )
    except (pinned_fs.PinnedPathRefusal, OSError) as exc:
        return Outcome(str(resolved), KEPT, f"refused: {exc}")
    if not ok:
        return Outcome(str(resolved), KEPT, "changed: not the file checked")
    return Outcome(str(resolved), REMOVED)


def reclaim_one(
    path: str,
    view: ProcessView,
    *,
    dry_run: bool = False,
    now: float | None = None,
    roots: Sequence[str] = TEMP_ROOTS,
    uid: int | None = None,
    reread: Callable[[], ProcessView] = read_processes,
) -> Outcome:
    """Check one requested path and remove it when every check passes (see the module doc).

    *view* is the process table read once for the run; *reread* reads it again
    right before the removal, for a file and a directory alike.
    """
    uid = os.getuid() if uid is None else uid
    now = time.time() if now is None else now
    # A trailing slash makes ``lstat`` follow a link the name ends in, so it goes.
    path = path.rstrip("/") or "/"
    try:
        info = os.lstat(path)
    except OSError as exc:
        return Outcome(path, KEPT, f"missing: {exc.strerror or exc}")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        return Outcome(path, KEPT, "not a plain directory or file")
    resolved = Path(os.path.realpath(path))
    root = _temp_root(resolved, roots)
    if root is None or resolved.parent != root:
        return Outcome(
            str(resolved), KEPT, "outside-scope: not a top-level entry of " + " or ".join(roots)
        )
    if not _user(resolved, uid):
        return Outcome(str(resolved), KEPT, "outside-scope: not owned by you")
    # Before any walk: a FUSE or network mount can be slow, and its files are not temp.
    mounted = _mount_doubt(resolved, info.st_dev, root)
    if mounted:
        return Outcome(str(resolved), KEPT, mounted)
    is_dir = stat.S_ISDIR(info.st_mode)
    # An unreadable entry answers +inf, so a tree that cannot be walked reads as recent.
    if is_dir:
        newest, socket, inodes = _scan(resolved)
    else:
        newest, socket, inodes = _changed(info), False, frozenset({(info.st_dev, info.st_ino)})
    if newest == float("inf"):
        return Outcome(str(resolved), KEPT, "recent: part of it could not be read")
    if now - newest < IDLE_SECONDS:
        return Outcome(
            str(resolved), KEPT, f"recent: written {int(max(now - newest, 0) // 60)} min ago"
        )
    if is_dir and _holds_repository(resolved):
        return Outcome(
            str(resolved), KEPT, "holds-repository: use git worktree remove, or move the work"
        )
    blocked = _blocker(resolved, view, inodes) or (_SOCKET_REASON if socket else "")
    if blocked:
        return Outcome(str(resolved), KEPT, blocked)
    if dry_run:
        return Outcome(str(resolved), WOULD_REMOVE)
    return _remove(resolved, info, now=now, reread=reread)


def reclaim(
    paths: Iterable[str], *, dry_run: bool = False, view: ProcessView | None = None
) -> list[Outcome]:
    """Run :func:`reclaim_one` over *paths*, reading the process table once.

    One path's ``OSError`` keeps that path and moves on, so a failure late in a
    run never hides what earlier paths came to.
    """
    view = read_processes() if view is None else view
    started = time.monotonic()
    out = []
    for path in paths:
        if time.monotonic() - started > DEADLINE_SECONDS:
            out.append(Outcome(path, KEPT, "timeout: not checked"))
            continue
        try:
            out.append(reclaim_one(path, view, dry_run=dry_run))
        except OSError as exc:
            out.append(Outcome(path, KEPT, f"refused: {exc}"))
    return out


def _say(text: str, stream: TextIO) -> None:
    """Print *text*, escaping what *stream* cannot encode.

    A name that is not UTF-8 arrives as a lone surrogate, and printing it raw
    would raise after the removals ran, losing the rest of the report.
    """
    encoding = stream.encoding or "utf-8"
    print(text.encode(encoding, "backslashreplace").decode(encoding), file=stream)


def run_command(paths: Sequence[str], *, dry_run: bool) -> None:
    """``kirocrew tmp reclaim``: print one line per path; exit 0 only when none was kept."""
    if not sys.platform.startswith("linux"):
        print("kirocrew tmp reclaim needs /proc and runs on Linux only.", file=sys.stderr)
        sys.exit(2)
    view = read_processes()
    outcomes = reclaim(paths, dry_run=dry_run, view=view)
    for o in outcomes:
        _say(f"{o.outcome:<13} {o.path}" + (f"  ({o.reason})" if o.reason else ""), sys.stdout)
    if view.not_inspected:
        names = ", ".join(view.not_inspected[:5])
        more = ", ..." if len(view.not_inspected) > 5 else ""
        fate = "may be removed" if dry_run else "may have been removed"
        _say(
            f"\nNot inspected: {names}{more}. The kernel marks these processes of yours as "
            "not dumpable, so their open files and working directories cannot be read; "
            f"a path one of them still uses {fate}.",
            sys.stderr,
        )
    if view.unreadable:
        print(
            f"\n{view.unreadable} of your processes refused access to their /proc entries, "
            "so no path is proven unused. An agent sandbox does this for every process "
            "outside it; run the command from a terminal outside the sandbox.",
            file=sys.stderr,
        )
    sys.exit(1 if any(o.outcome == KEPT for o in outcomes) else 0)
