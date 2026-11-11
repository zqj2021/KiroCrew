"""A moved Bot's restrictions its worker cannot carry exactly, and the ask guard for them.

When the upgrade moves a Bot onto its own file (``mate_config_migration``), or a
start repairs one an earlier release moved without its old agent's limits
(``mate_policy_repair``), each ``hooks`` entry, ``permissions`` rule and
``toolsSettings`` entry of the old agent is carried into the Bot's file when the
worker keeps it exactly (:func:`split_policy`). One that cannot be carried is
never written in a form that would stop the worker from starting. It is
recorded instead, under ``held_rules`` in ``migrated.json``, with the capability
it speaks about, and:

- on KAS, an ``ask`` rule for that capability is carried in its place, so KAS
  sends every such call to Crew instead of answering it from ``allowedTools``;
- Crew's own approval gates never approve such a call without a person
  (:func:`held_for`): an ``auto`` cron, Trust, YOLO, a trusted pattern or a hook
  grant sends it to the approval card instead, and a surface with no card refuses it.

Every other call behaves as before. Blocking file I/O; never raises from the gate.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from kiro_crew.user_json import copy_json_document

logger = logging.getLogger(__name__)

#: The ``migrated.json`` key: Bot name -> its held restrictions.
HELD_KEY = "held_rules"

#: A held restriction whose capability cannot be told: every call of that Bot asks.
ANY = "*"

#: ``toolsSettings`` family -> the KAS capability its entries limit.
_FAMILY_CAPABILITY = {
    "shell": "shell",
    "read": "fs_read",
    "write": "fs_write",
    "web_fetch": "web_fetch",
    "subagent": "subagent",
}

#: Built-in tool ids (kiro-cli and KAS spellings) -> capability.
_TOOL_CAPABILITY = {
    **dict.fromkeys(
        ("execute_bash", "execute_pwsh", "shell", "run_command", "control_bash_process"), "shell"
    ),
    **dict.fromkeys(("fs_read", "read", "read_file", "grep", "glob", "list_directory"), "fs_read"),
    **dict.fromkeys(("fs_write", "write", "fs_append", "str_replace", "write_file"), "fs_write"),
    **dict.fromkeys(("web_fetch", "fetch"), "web_fetch"),
    **dict.fromkeys(("subagent", "use_subagent"), "subagent"),
}

#: ACP semantic kind -> capability.
_KIND_CAPABILITY = {
    "execute": "shell",
    "read": "fs_read",
    "search": "fs_read",
    "edit": "fs_write",
    "delete": "fs_write",
    "move": "fs_write",
    "fetch": "web_fetch",
}

#: Capabilities a carried ``ask`` rule is written for.
_ASKABLE = frozenset(_FAMILY_CAPABILITY.values())

_KNOWN = frozenset({*_ASKABLE, "mcp", "web_search"})


def _capability_of_tool(name: str) -> str:
    from kiro_crew.agent_sdk.kas_tool_view import tools_settings_family

    return _TOOL_CAPABILITY.get(name) or _FAMILY_CAPABILITY.get(tools_settings_family(name), "")


def _hook_capability(entry: Any) -> str:
    matcher = entry.get("matcher") if isinstance(entry, dict) else None
    if not isinstance(matcher, str) or matcher in ("", "*"):
        return ANY
    if matcher.startswith("@builtin"):
        # ``@builtin`` covers every built-in tool, ``@builtin/<tool>`` that one.
        tool = matcher.partition("/")[2]
        return (_capability_of_tool(tool) or ANY) if tool else ANY
    if matcher.startswith("@") or "/" in matcher:
        return "mcp"
    return _capability_of_tool(matcher) or ANY


def _pieces(policy: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Each restriction in *policy*, as ``(capability, the policy holding it alone)``."""
    out: list[tuple[str, dict[str, Any]]] = []
    hooks = policy.get("hooks")
    if isinstance(hooks, dict):
        for event, entries in hooks.items():
            for entry in entries if isinstance(entries, list) else [entries]:
                out.append((_hook_capability(entry), {"hooks": {event: [entry]}}))
    elif "hooks" in policy:
        out.append((ANY, {"hooks": hooks}))
    perms = policy.get("permissions")
    if isinstance(perms, dict) and set(perms) <= {"rules"} and isinstance(perms.get("rules"), list):
        for rule in perms["rules"]:
            cap = rule.get("capability") if isinstance(rule, dict) else None
            cap = cap if isinstance(cap, str) and cap in _KNOWN else ANY
            out.append((cap, {"permissions": {"rules": [rule]}}))
    elif "permissions" in policy:
        out.append((ANY, {"permissions": perms}))
    settings = policy.get("toolsSettings")
    if isinstance(settings, dict):
        for tool, entry in settings.items():
            out.append((_capability_of_tool(str(tool)) or ANY, {"toolsSettings": {tool: entry}}))
    elif "toolsSettings" in policy:
        out.append((ANY, {"toolsSettings": settings}))
    return out


def _lands(base: dict[str, Any], mate_file: dict[str, Any], piece: dict[str, Any]) -> bool:
    """Whether the worker merged from *mate_file* runs and keeps every restriction of *piece*."""
    from kiro_crew.mate_config_migration import dropped_restrictions
    from kiro_crew.mate_spec_policy import SpecPolicyUncarried
    from kiro_crew.mate_worker import merge_worker_spec, worker_additions

    try:
        merged = merge_worker_spec(base, worker_additions(base, mate_file))
    except SpecPolicyUncarried:
        return False
    return not dropped_restrictions(base, piece, merged)


def _phrases(base: dict[str, Any], piece: dict[str, Any]) -> list[str]:
    from kiro_crew.mate_config_migration import dropped_restrictions

    return dropped_restrictions(base, piece, base) or ["one of its limits"]


def split_policy(
    applies: bool, base: dict[str, Any], policy: dict[str, Any], mate_file: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """*mate_file* with every restriction of *policy* it can carry exactly, and the rest.

    *applies* says the Bot's worker backend runs a file's restrictions (KAS);
    without it nothing is written, since such a file would stop the worker, and
    every restriction is held. Each held one is ``{"text", "capability"}``. On KAS
    an ``ask`` rule is carried for each held capability that has one. The result
    always merges: the worker starts. *mate_file* is not changed.
    """
    from kiro_crew.mate_spec_policy import SpecPolicyUncarried, layer_policy

    out = copy_json_document(mate_file)
    held: list[dict[str, str]] = []
    for capability, piece in _pieces(policy):
        trial = copy_json_document(out)
        try:
            layer_policy(trial, piece)
        except SpecPolicyUncarried:
            trial = None
        if applies and trial is not None and _lands(base, trial, piece):
            out = trial
            continue
        held.extend({"text": t, "capability": capability} for t in _phrases(base, piece))
    if applies:
        for capability in sorted({h["capability"] for h in held} & _ASKABLE):
            trial = copy_json_document(out)
            try:
                layer_policy(trial, {"permissions": {"rules": [_ask(capability)]}})
            except SpecPolicyUncarried:
                continue
            if _lands(base, trial, {}):
                out = trial
    return out, held


def _ask(capability: str) -> dict[str, str]:
    return {"capability": capability, "effect": "ask"}


def record_held(name: str, member_id: str, held: list[dict[str, str]]) -> None:
    """Record *held* as Bot *name*'s restrictions that need a person; none clears it."""
    from kiro_crew.mate_config_migration import _read_ledger, _write_ledger

    data = _read_ledger()
    entries = data.get(HELD_KEY)
    entries = entries if isinstance(entries, dict) else {}
    if held:
        entries[name] = {"member_id": member_id, "rules": held}
    else:
        entries.pop(name, None)
    data[HELD_KEY] = dict(sorted(entries.items()))
    _write_ledger(data)


#: The phrase a Bot's call is held under while ``migrated.json`` cannot be read.
UNREADABLE = "the record of its limits could not be read"


def _held_entries() -> dict[str, Any] | None:
    """``held_rules`` from ``migrated.json``: ``{}`` when absent, ``None`` when unreadable."""
    from kiro_crew.config import paths as paths_mod

    path = paths_mod.mate_agents_dir().parent / "migrated.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("held Bot rules: %s unreadable; every Bot call asks a person", path)
        return None
    if not isinstance(data, dict):
        return None
    entries = data.get(HELD_KEY)
    return entries if isinstance(entries, dict) else {}


def _is_bot(agent: str) -> bool:
    """Whether *agent* names a Bot (a binding with a ``member_id``); False when unknown."""
    if not agent:
        return False
    from kiro_crew.config.loader import KiroCrewConfig

    try:
        binding = KiroCrewConfig.load().agents.get(agent)
    except Exception:  # noqa: BLE001 - an unreadable config names no Bot
        return False
    return bool(getattr(binding, "member_id", "") or "")


def event_capabilities(event: Any) -> set[str]:
    """The capabilities a permission request *event* speaks about; empty when unknown."""
    from kiro_crew.hook_runtime.tool_identity import ToolCall
    from kiro_crew.platform.tool_names import KIRO_TOOL_ALIASES

    call = ToolCall.from_event(event)
    caps: set[str] = set()
    if call.is_shell or call.command:
        caps.add("shell")
    if call.mcp_server:
        caps.add("mcp")
    elif call.mcp_tool:
        for name in (call.mcp_tool, KIRO_TOOL_ALIASES.get(call.mcp_tool, call.mcp_tool)):
            if _capability_of_tool(name):
                caps.add(_capability_of_tool(name))
    if call.kind in _KIND_CAPABILITY:
        caps.add(_KIND_CAPABILITY[call.kind])
    return caps


def held_for(event: Any, *, session_key: str = "", agent: str = "") -> list[str]:
    """The held restrictions of the Bot behind *session_key* (or *agent*) *event* falls under.

    Non-empty means no gate may approve the call without a person. A call whose
    capability cannot be told falls under every held restriction. While
    ``migrated.json`` exists but cannot be read, every call of a Bot is held
    (:data:`UNREADABLE`): what was recorded for it is unknown. Never raises; any
    other error reads as nothing held, so a call nobody recorded anything for
    behaves as before.
    """
    try:
        entries = _held_entries()
        if entries == {}:
            return []
        member_id = ""
        if session_key:
            from kiro_crew.execution_context import read_session_execution

            try:
                execution = read_session_execution(session_key)
            except Exception:  # noqa: BLE001 - an unreadable record names no Bot
                execution = None
            member_id = str(getattr(execution, "member_id", "") or "")
        if entries is None:
            return [UNREADABLE] if member_id or _is_bot(agent) else []
        rules: list[Any] = []
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                continue
            if (member_id and entry.get("member_id") == member_id) or (agent and name == agent):
                rules.extend(entry.get("rules") or [])
        if not rules:
            return []
        caps = event_capabilities(event)
        return [
            str(r.get("text") or "")
            for r in rules
            if isinstance(r, dict)
            and (r.get("capability") == ANY or not caps or r.get("capability") in caps)
        ]
    except Exception:  # noqa: BLE001 - the gate never breaks on this record
        logger.warning("held Bot rules: could not be read for a call", exc_info=True)
        return []


def held_sentence(held: list[dict[str, str]]) -> str:
    """The notice's sentence on what did not apply, in plain words."""
    texts = list(dict.fromkeys(h["text"] for h in held if h.get("text")))
    return (
        "some limits from the agent it ran on could not be applied as written ("
        + "; ".join(texts)
        + "), so it now asks you before any action they cover, even when it would "
        "otherwise go ahead on its own"
    )


def owe_held_notice(name: str, held: list[dict[str, str]]) -> None:
    """Owe Bot *name*'s user one notice, in its chat and the bell, naming *held*.

    For a Bot this start moved: its notice says the limits apply from now on, not
    that earlier work ran without them (``upgrade_notice_held``).
    """
    from kiro_crew.mate_upgrade_notice import mark_policy_owed

    mark_policy_owed(name, held_sentence(held), held=True, fresh=True)


#: Set while an unattended turn's approver is asked about a held call
#: (``llm_helpers_parts.tool_approval``): an approver that would answer on its
#: own (an ``auto`` cron, the heartbeat allowlist) reads it and asks a person or
#: refuses instead.
_HELD_CALL: ContextVar[tuple[str, ...]] = ContextVar("mate_held_call", default=())


@contextmanager
def held_call(phrases: list[str]) -> Iterator[None]:
    """Mark the approval asked inside this block as one a person must make."""
    token = _HELD_CALL.set(tuple(phrases))
    try:
        yield
    finally:
        _HELD_CALL.reset(token)


def call_is_held() -> bool:
    """Whether the approval being asked now falls under a held Bot limit."""
    return bool(_HELD_CALL.get())
