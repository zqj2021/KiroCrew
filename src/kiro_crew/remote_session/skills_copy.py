"""Gateway side of a remote session's skill copy: the agent's skills, runnable in its workspace.

A remote session's agent reads its skills here, on the gateway (the skills index,
``skill_search``, KAS's own skill tool), but its commands run in the workspace,
where the skill's ``scripts/`` never were. At every runtime start (a new session,
a resume, a restart) this copies the skills that agent may see, and only those,
into the workspace, and every skill path the agent is shown names that copy.

What is copied, per skill: its directory's regular files (``SKILL.md``,
``scripts/``, ``references/``, assets), read without following links. Left out,
each reported: every symlink and hard link; every hidden file or directory (``.env``, ``.git``,
provenance markers); a directory holding its own ``SKILL.md`` (another skill,
copied only when it is in scope itself); credential-named files (private keys,
``credentials``); a file whose text the shared credential scanner matches; a file
over :data:`MAX_FILE_BYTES`; and every skill past :data:`MAX_TOTAL_BYTES` or
:data:`MAX_FILES`. Project skills are not copied: they live in the repository the
workspace checks out.

The bytes travel on the stdin of one guest command over the session's existing
shell (``coder ssh``), as the worker files do; nothing listens. In the guest
``skills_install.py`` writes a content-addressed, read-only copy at
``~/.kirocrew-worker/skills/<digest>/<skill key>/``; a digest already there is
reused without a transfer.

Refreshed at each runtime start, not mid-turn: the copy a running command uses
never changes under it. shortcut: a skill edited mid-session reaches the
workspace at the next start (resume, reset or restart); refresh on change when a
per-turn check can swap copies between turns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import re
import shlex
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kiro_crew.remote_session import skills_install
from kiro_crew.session_environment import _REMOTE_WORKSPACES_MAX, remote_workspace_for

logger = logging.getLogger(__name__)

#: Where copies live in the workspace, as documented and as the agent is told.
GUEST_SKILLS_DIR = "~/.kirocrew-worker/skills"
MAX_FILE_BYTES = 1 << 20
#: The guest's own bounds, which it checks again: one definition for both sides.
MAX_TOTAL_BYTES = skills_install.MAX_TOTAL_BYTES
MAX_FILES = skills_install.MAX_FILES
#: Names that hold credentials whatever their content.
_SECRET_NAMES = frozenset(
    {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials", "secring.gpg"}
)
_SECRET_SUFFIXES = (".pem", ".key", ".p8", ".p12", ".pfx", ".jks", ".keystore")
_KEY_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@+-]*")
#: Left-out paths kept as examples; :attr:`CopyPlan.skipped_total` counts them all.
_MAX_SKIPPED = 200
#: How long the scope read waits for a first-run skill catalog to finish building.
CATALOG_WAIT_SECONDS = 60.0


@dataclass(frozen=True)
class Skipped:
    """A path left out of the copy, and why (shown to the person and the agent)."""

    path: str
    reason: str


@dataclass
class CopyPlan:
    """What one copy carries: skills by key, their files, and what was left out."""

    #: key -> every spelling of the skill's local directory the agent may be shown.
    skills: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: (path under the copy, mode, bytes), in order.
    files: list[tuple[str, int, bytes]] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    #: Every path left out, including those past the :data:`_MAX_SKIPPED` examples.
    skipped_total: int = 0
    #: Set by :func:`plan_copy` once the files are final (off the event loop).
    digest: str = ""

    def seal(self) -> None:
        h = hashlib.sha256()
        for rel, mode, data in self.files:
            h.update(f"{rel}\0{mode & 0o111}\0{len(data)}\0".encode("utf-8", "surrogateescape"))
            h.update(data)
        self.digest = h.hexdigest()[:32]

    def payload(self) -> bytes:
        header = json.dumps([[rel, len(data), mode] for rel, mode, data in self.files])
        return header.encode("utf-8") + b"\n" + b"".join(data for _r, _m, data in self.files)

    def skip(self, path: str, reason: str) -> None:
        self.skipped_total += 1
        if len(self.skipped) < _MAX_SKIPPED:
            self.skipped.append(Skipped(path, reason))


@dataclass(frozen=True)
class SkillCopy:
    """A copy in the workspace, as the gateway records it for one session."""

    root: str
    #: key -> local directory spellings (see :attr:`CopyPlan.skills`).
    skills: dict[str, tuple[str, ...]]
    skipped: tuple[Skipped, ...] = ()
    #: Every left-out path; ``skipped`` holds the first examples.
    skipped_total: int = 0
    python: str = ""
    #: (path under the copy, reason) for copied ``.py`` files the guest cannot compile.
    incompatible: tuple[tuple[str, str], ...] = ()

    @property
    def python_older(self) -> bool:
        """Whether the workspace's python3 is older than the gateway's own Python."""
        try:
            guest = tuple(int(part) for part in self.python.split(".")[:2])
        except ValueError:
            return False
        return len(guest) == 2 and guest < sys.version_info[:2]

    def guest_dir(self, key: str) -> str | None:
        return f"{self.root}/{key}" if key in self.skills else None

    def rewrite(self, text: str) -> str:
        """``text`` with each copied skill's local directory replaced by its copy."""
        pairs = sorted(
            (
                (local, f"{self.root}/{key}")
                for key, spellings in self.skills.items()
                for local in spellings
            ),
            key=lambda pair: -len(pair[0]),
        )
        for local, guest in pairs:
            # Both separators: a Windows gateway renders its paths with ``\``.
            for sep in dict.fromkeys(("/", os.sep)):
                text = text.replace(local + sep, guest + "/")
        return text


def _gateway_python() -> str:
    return "%d.%d" % sys.version_info[:2]


def _secret_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in _SECRET_NAMES or lowered.endswith(_SECRET_SUFFIXES)


def _read_regular(
    name: str, limit: int, dir_fd: int | None = None, ident: list[Any] | None = None
) -> tuple[bytes | None, int, str]:
    """``(bytes, mode, "")`` of a regular file opened without following a link, or
    ``(None, 0, reason)``. *name* is opened relative to *dir_fd* when one is given.
    *ident*, when given, receives the descriptor's ``(st_dev, st_ino)``."""
    try:
        # Non-blocking, so a FIFO swapped in after the walk's type check cannot hang
        # the open; the descriptor check below refuses it.
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError:
        return None, 0, "unreadable or a link"
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            return None, 0, "not a regular file"
        # A hard link can name a file outside the skill, as the fenced skill reader
        # (``hooks.safe_read_prefix``) also refuses.
        if info.st_nlink > 1:
            return None, 0, "hard link"
        if info.st_size > limit:
            return None, 0, f"over the {limit // 1024} KiB file cap"
        data = handle.read(limit + 1)
    if ident is not None:
        ident.append((info.st_dev, info.st_ino))
    if len(data) > limit:
        return None, 0, f"over the {limit // 1024} KiB file cap"
    return data, info.st_mode, ""


def _holds_credential(data: bytes) -> bool:
    from kiro_crew.security import _contains_fixed_credential, iter_credential_matches

    # Binary and non-UTF-8 files are scanned too: a token in a pickle or a SQLite
    # cache is still a token. Base64-encoded ones (a kubeconfig's client-key-data)
    # count as well.
    text = data.decode("utf-8", errors="replace")
    return next(iter(iter_credential_matches(text)), None) is not None or bool(
        _contains_fixed_credential(text)
    )


class _OverBudget(Exception):
    """A skill's files would pass what is left of the copy's byte or file budget."""


class _NotAdmitted(Exception):
    """The ``SKILL.md`` this walk read is not the one the scope check admitted."""


def _skill_files(
    local_dir: str,
    plan: CopyPlan,
    bytes_left: int,
    files_left: int,
    admit: Callable[[str, tuple[int, int], bytes], bool] | None = None,
) -> list[tuple[str, int, bytes]] | None:
    """The copyable files under ``local_dir`` (relative paths), or None if unreadable.

    *admit*, when given, is asked about the skill's own ``SKILL.md`` with the
    descriptor identity and the bytes this walk read; a refusal leaves the whole
    skill out, so the decision rests on the file actually copied.

    Walked by descriptor (``pinned_fs``): the root is opened once with its ancestors
    pinned, and every directory and file below it is opened relative to its
    parent's descriptor with ``O_NOFOLLOW``, so a component swapped for a link
    mid-walk is refused rather than followed out of the skill.

    Raises :class:`_OverBudget` as soon as the skill passes ``bytes_left`` or
    ``files_left``, before reading further, so a large tree costs no full walk.
    """
    from kiro_crew import pinned_fs
    from kiro_crew.security.paths import is_sensitive_resolved_path

    out: list[tuple[str, int, bytes]] = []
    budget = [bytes_left, files_left]

    def walk(dir_fd: int, where: str, rel: str) -> None:
        try:
            entries = sorted(os.scandir(dir_fd), key=lambda e: e.name)
        except OSError:
            plan.skip(where, "unreadable directory")
            return
        for entry in entries:
            path = os.path.join(where, entry.name)
            child_rel = f"{rel}/{entry.name}" if rel else entry.name
            if entry.name.startswith("."):
                plan.skip(path, "hidden")
            elif is_sensitive_resolved_path(path):
                # The fence every file surface applies (a skill mapped inside the
                # data home must not carry its ``diag/`` logs). *path* is canonical:
                # the root is resolved and no link below it is ever followed.
                plan.skip(path, "sensitive path")
            elif "\\" in entry.name or any(ord(c) < 32 for c in entry.name):
                # The guest refuses a backslash (``skills_install.safe_rel``), and a
                # control character could carry a line into the agent's context.
                plan.skip(path, "a name the workspace copy cannot hold")
            elif entry.is_symlink():
                plan.skip(path, "symlink")
            elif entry.is_dir(follow_symlinks=False):
                try:
                    child = os.open(entry.name, pinned_fs.dir_flags(), dir_fd=dir_fd)
                except OSError:
                    plan.skip(path, "unreadable directory or a link")
                    continue
                try:
                    # Its children are named from where the held descriptor really
                    # is, not the name it was opened by, so the fence below judges
                    # the file actually read even if a folder was renamed in.
                    real_child = pinned_fs.fd_real_path(child)
                    try:
                        os.stat("SKILL.md", dir_fd=child, follow_symlinks=False)
                    except FileNotFoundError:
                        if real_child is None:
                            plan.skip(path, "unreadable directory")
                        else:
                            walk(child, real_child, child_rel)
                    except OSError:
                        plan.skip(path, "unreadable directory")
                    else:
                        plan.skip(path, "another skill")
                finally:
                    os.close(child)
            elif not entry.is_file(follow_symlinks=False):
                plan.skip(path, "not a regular file")
            elif _secret_name(entry.name):
                plan.skip(path, "credential file name")
            else:
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    size = 0
                # A file over its own cap is refused below and costs no bytes.
                budget[0] -= size if size <= MAX_FILE_BYTES else 0
                budget[1] -= 1
                if budget[0] < 0 or budget[1] < 0:
                    raise _OverBudget
                ident: list[Any] = []
                data, mode, reason = _read_regular(
                    entry.name, MAX_FILE_BYTES, dir_fd=dir_fd, ident=ident
                )
                if data is None:
                    plan.skip(path, reason)
                elif (
                    admit is not None
                    and child_rel == "SKILL.md"
                    and not (ident and admit(path, ident[0], data))
                ):
                    raise _NotAdmitted
                elif _holds_credential(data):
                    plan.skip(path, "looks like it holds a credential")
                else:
                    out.append((child_rel, mode, data))

    if not pinned_fs.supports_pinned_tree_walk():
        # Fail closed: a by-name walk is the hole the descriptors close.
        plan.skip(local_dir, "this platform cannot read a skill without following links")
        return None
    try:
        root = pinned_fs.open_dir_pinned(local_dir, what="skill directory")
    except (OSError, pinned_fs.PinnedPathRefusal):
        return None
    try:
        real_root = pinned_fs.fd_real_path(root)
        if real_root is None or is_sensitive_resolved_path(real_root):
            plan.skip(local_dir, "sensitive path or unreadable directory")
            return None
        walk(root, real_root, "")
    finally:
        os.close(root)
    return out if any(rel == "SKILL.md" for rel, _m, _d in out) else None


def plan_copy(rows: list[dict[str, Any]]) -> CopyPlan:
    """The copy of ``rows`` (``scoped_skills`` rows: ``key`` and ``dir``)."""
    plan = CopyPlan()
    total = 0
    for row in rows:
        key = str(row.get("key") or "")
        local = str(row.get("dir") or "")
        if key in plan.skills:
            continue
        # A key is a POSIX path in the guest, whatever this gateway's OS.
        if not local or not all(_KEY_SEGMENT.fullmatch(s) for s in key.split(posixpath.sep)):
            plan.skip(local or key, "skill key cannot name a workspace path")
            continue
        # A selected row names the directory its check was fixed against; the
        # copy reads there and never resolves the row's path again.
        real = str(row.get("real_dir") or os.path.realpath(local))
        try:
            files = _skill_files(
                real,
                plan,
                MAX_TOTAL_BYTES - total,
                MAX_FILES - len(plan.files),
                admit=row.get("admit"),
            )
        except _OverBudget:
            plan.skip(local, f"over the {MAX_TOTAL_BYTES >> 20} MiB / {MAX_FILES} file copy cap")
            continue
        except _NotAdmitted:
            plan.skip(local, "not a skill this session loads (or it changed since checked)")
            continue
        if files is None:
            plan.skip(local, "SKILL.md missing, unreadable or a link")
            continue
        size = sum(len(d) for _r, _m, d in files)
        if total + size > MAX_TOTAL_BYTES:  # a file that grew between stat and read
            plan.skip(local, f"over the {MAX_TOTAL_BYTES >> 20} MiB / {MAX_FILES} file copy cap")
            continue
        total += size
        plan.skills[key] = tuple(dict.fromkeys((local.rstrip("/"), real)))
        plan.files.extend((f"{key}/{rel}", mode, data) for rel, mode, data in files)
    plan.seal()
    return plan


def allowed_skill_rows(
    session_key: str, agent: str, project_dir: str | Path | None
) -> list[dict[str, Any]]:
    """The skills the KAS session for *agent* may use, read from the spec KAS runs.

    The scope is the one KAS applies to that session's spec
    (``agent_sdk.kas_skill_view.kas_session_skill_scope``), so a checkout cannot
    widen what is copied. A skill from a tree KAS reads is copied only if KAS
    loads it (``kas_skill_scope.loaded_skill_files``, the classifier a KAS start
    runs), and a Crew catalog skill only if the session's rules do not deny it.
    A whole-catalog agent (``kirocrew``,
    ``kirocrew-worker``, the crewmate's) gets Crew's whole catalog plus the
    ``~/.kiro/skills`` tree KAS offers it; an agent that inherits the default
    trees gets ``~/.kiro/skills``. Project trees and project skills are left out:
    they live in the repository the workspace checks out. A failure raises, and
    the caller copies nothing.
    """
    from kiro_crew.agent_discovery import WHOLE_CATALOG_TEMPLATES
    from kiro_crew.agent_sdk.kas_skill_view import kas_session_skill_scope
    from kiro_crew.config.paths import kiro_home
    from kiro_crew.skills import SkillsLoader

    native = [str(kiro_home() / "skills" / "*" / "SKILL.md")]
    kas = kas_session_skill_scope(session_key, agent, project_dir)
    globs = kas.globs
    whole_catalog = globs is None and agent in WHOLE_CATALOG_TEMPLATES
    if globs is None:
        globs = native
    if project_dir:
        checkout = os.path.realpath(project_dir) + os.sep
        globs = [g for g in globs if not os.path.realpath(g).startswith(checkout)]
    scope: list[str] = globs
    loader = SkillsLoader(install_builtins=False)
    try:
        # A first-run catalog still building lists only part of the scope, and a
        # copy taken from it would miss skills for the whole runtime: wait for it,
        # or fail so the agent is told the copy is missing. A read that timed out
        # can be followed by the builder publishing before the status is checked,
        # so the rows kept are only those read after the catalog reported complete.
        def read() -> list[dict[str, Any]]:
            # Each row carries its check, which ``plan_copy`` asks again with the
            # descriptor and bytes it copies: a file replaced after this first
            # pass matches no verdict and is left out.
            rows = (
                [
                    {**row, **bound}
                    for row in loader.scoped_skills(project_dir=project_dir, only=None)
                    if (bound := kas.bind(str(row.get("path") or ""))) is not None
                ]
                if whole_catalog
                else []
            )
            # The catalog matches globs with fnmatch (``*`` crosses folders), and
            # lists skills KAS refuses; keep only the files KAS loads.
            return rows + [
                {**row, **bound}
                for row in loader.scoped_skills(project_dir=project_dir, only=scope)
                if (bound := kas.bind(str(row.get("path") or ""), crew_only=False)) is not None
            ]

        deadline = time.monotonic() + CATALOG_WAIT_SECONDS
        read()  # starts discovery on a cold catalog
        while True:
            while loader.catalog_status(project_dir) == "building":
                if time.monotonic() > deadline:
                    raise RuntimeError("the skill catalog is still building")
                time.sleep(0.5)
            rows = read()
            if loader.catalog_status(project_dir) != "building":
                break
    finally:
        loader.close()
    return [row for row in rows if not row.get("confine_root")]


def _command(mode: str, digest: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16,64}", digest):
        raise ValueError("invalid skills digest")
    return (
        'set -eu; umask 077; cd "$HOME/.kirocrew-worker"; '
        f"exec python3 skills_install.py {shlex.quote(mode)} {digest}"
    )


#: The guest's answer reaches the agent's context, and the workspace is the agent's
#: to change, so only shapes the gateway can check are kept: a plain absolute
#: path ending in this copy's own folder, a dotted version, and paths it sent.
#: ``mcp_tools/skills._read_capacity`` reserves room for a root this long.
ROOT_MAX_CHARS = 256
_ROOT_RE = re.compile(r"/[A-Za-z0-9._@+/-]{1,%d}" % (ROOT_MAX_CHARS - 1))
_VERSION_RE = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}")


def _parse(output: bytes, plan: CopyPlan) -> SkillCopy:
    try:
        result = json.loads(output.decode("utf-8").strip().splitlines()[-1])
    except (UnicodeDecodeError, IndexError, json.JSONDecodeError) as exc:
        raise ValueError("the workspace sent no readable skills report") from exc
    if not isinstance(result, dict):
        raise ValueError("the workspace sent no readable skills report")
    root = result.get("root")
    if (
        result.get("present") is not True
        or not isinstance(root, str)
        or not _ROOT_RE.fullmatch(root)
        or not root.endswith(f"/.kirocrew-worker/skills/{plan.digest}")
        or "/../" in root
    ):
        raise ValueError("the workspace did not report a complete skills copy")
    python = result.get("python")
    sent = {rel for rel, _m, _d in plan.files}
    bad = result.get("incompatible")
    return SkillCopy(
        root=root,
        skills=dict(plan.skills),
        skipped=tuple(plan.skipped),
        skipped_total=plan.skipped_total,
        python=python if isinstance(python, str) and _VERSION_RE.fullmatch(python) else "",
        # Each sent path once, so the guest cannot set the count by repeating one.
        incompatible=tuple(
            (path, "")
            for path in list(
                dict.fromkeys(
                    item[0]
                    for item in (bad if isinstance(bad, list) else [])
                    if isinstance(item, list) and item and item[0] in sent
                )
            )[: skills_install.MAX_REPORTED]
        ),
    )


async def sync_skills(shell: Any, plan: CopyPlan) -> SkillCopy:
    """Make ``plan``'s copy present in the workspace behind ``shell``; return it.

    Asks first, so a copy another session (or this one's last start) installed is
    reused with no transfer. Needs ``skills_install.py`` beside the worker files.
    """
    digest = plan.digest
    found = await shell.run(_command("check", digest))
    copy = None
    try:
        copy = _parse(found, plan)
    except ValueError:
        pass
    if copy is None:
        copy = _parse(await shell.run(_command("install", digest), plan.payload()), plan)
    return copy


#: A start's copy may not hold the session's start past this.
COPY_TIMEOUT_SECONDS = 90.0
#: The record when the switch is off, and when the copy was tried and failed.
DISABLED = "disabled"
FAILED = "failed"
Record = SkillCopy | str | None
#: One record per session, read by the context and the environment panel: the copy
#: this session's current runtime made, ``DISABLED``, ``FAILED``, or ``None`` (no
#: skill to copy). Bounded by the one cap ``session_environment`` keeps for the
#: same sessions' per-session maps.
_RECORDS_MAX = _REMOTE_WORKSPACES_MAX
_records: dict[str, Record] = {}


def _remember(session_key: str, record: Record) -> None:
    _records.pop(session_key, None)
    if len(_records) >= _RECORDS_MAX:
        # A session that does not run remotely goes first, so ended sessions'
        # records never push out a live one's.
        ended = next((k for k in _records if remote_workspace_for(k) is None), None)
        _records.pop(ended if ended is not None else next(iter(_records)))
    _records[session_key] = record


def record_for(session_key: str | None) -> Record:
    """This session's record, only while it runs remotely (else ``None``)."""
    if not session_key or not (remote_workspace_for(session_key) or ("", ""))[1]:
        return None
    return _records.get(session_key)


def copy_for(session_key: str | None) -> SkillCopy | None:
    record = record_for(session_key)
    return record if isinstance(record, SkillCopy) else None


async def copy_for_session(
    session_key: str, agent: str, project_dir: str | Path | None, shell: Any, *, enabled: bool
) -> None:
    """Copy ``session_key``'s skills into its workspace, at this runtime's start.

    Best effort: a failure costs the session its skill scripts, never its start,
    and is recorded so the agent is not shown a copy that is not there.
    """
    import asyncio

    if not enabled:
        _remember(session_key, DISABLED)
        return

    async def copy() -> SkillCopy | None:
        rows = await asyncio.to_thread(allowed_skill_rows, session_key, agent, project_dir)
        plan = await asyncio.to_thread(plan_copy, rows)
        for item in plan.skipped:
            logger.info("skills copy for %s left out %s: %s", session_key, item.path, item.reason)
        if plan.skipped_total > len(plan.skipped):
            logger.info(
                "skills copy for %s left out %d more paths not listed",
                session_key,
                plan.skipped_total - len(plan.skipped),
            )
        return await sync_skills(shell, plan) if plan.skills else None

    record: Record = FAILED
    try:
        # One deadline for the scope read, the walk and the transfer together.
        record = await asyncio.wait_for(copy(), COPY_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 -- the session starts without the copy
        logger.warning("Could not copy skills into the remote workspace: %r", exc)
    _remember(session_key, record)


def context_note(record: Record) -> str:
    """The per-turn line telling a remote agent where its skills' files are."""
    if record == DISABLED:
        return (
            "[SKILLS] Skill files are not copied into this workspace (Settings > "
            "Session Environments > Copy skills to the workspace is off), so a "
            "skill's scripts cannot run here.\n\n"
        )
    if record == FAILED:
        return (
            "[SKILLS] Copying this session's skills into the workspace failed when it "
            "started, so a skill's scripts are not there; the next start tries again.\n\n"
        )
    if not isinstance(record, SkillCopy):
        return ""
    text = (
        f"[SKILLS] Kiro Crew copied the {len(record.skills)} skills you can use into this "
        f"workspace, read-only, at {record.root}/<skill key>/ (SKILL.md, scripts/, "
        "references/, assets). That folder is the skill's directory here: run "
        "`python3 <it>/scripts/<name>.py` with run_command, and read its files with "
        "run_command too (they sit outside the folder the file tools reach). A skill "
        "path that names the gateway "
        "machine (its home, or ~/.kiro/crew/skills) means the same file in this copy. "
        "The copy was made when this session started; a skill edited since then "
        "reaches it at the next start."
    )
    if record.python_older:
        text += (
            f" The workspace's python3 is {record.python}, older than the Python "
            f"{_gateway_python()} the gateway runs skill scripts with. A script that fails "
            "here with a SyntaxError, a TypeError such as `unsupported operand type(s) "
            "for |`, or a missing standard-library name is most likely failing on that "
            "older interpreter, not on its own code; say so rather than patching the script."
        )
    if record.incompatible:
        from kiro_crew.context import _neutralize_structural_markers

        # A file name is the skill author's text, so it gets the same marker scrub as
        # every other piece of skill content that reaches the context.
        shown = _neutralize_structural_markers(", ".join(p for p, _r in record.incompatible[:5]))
        if len(record.incompatible) > 5:
            shown += f", and {len(record.incompatible) - 5} more"
        text += (
            f" {len(record.incompatible)} copied scripts do not compile on the "
            f"workspace's python3 ({shown})"
            + (": they need a newer Python there." if record.python_older else ".")
        )
    if record.skipped_total:
        text += (
            f" {record.skipped_total} files were left out (links, hidden or credential "
            "files, oversized files)."
        )
    return text + "\n\n"


def summary(record: Record) -> dict[str, Any] | None:
    """What the environment panel shows about the copy, or ``None`` for nothing."""
    if record == DISABLED:
        return {"state": "off"}
    if record == FAILED:
        return {"state": "failed"}
    if not isinstance(record, SkillCopy):
        return None
    return {
        "state": "copied",
        "root": record.root,
        "count": len(record.skills),
        "python": record.python,
        # The card warns when the workspace's python3 is older than this one.
        "gateway_python": _gateway_python() if record.python_older else None,
        "incompatible": len(record.incompatible),
        "skipped": [{"path": s.path, "reason": s.reason} for s in record.skipped[:50]],
        "skipped_count": record.skipped_total,
    }
