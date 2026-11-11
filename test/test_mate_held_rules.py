"""A moved Bot's limits its worker cannot carry exactly (``mate_held_rules``).

Pins that such a limit is never written in a form that stops the worker (it is
held instead, with a KAS ``ask`` rule in its place); that a held limit is found
for the calls it covers and only those; and that no gate approves such a call
without a person: an ``auto`` cron's approver asks, a surface with no approver
refuses, and a trusting dashboard thread shows the card. A call no held limit
covers is approved as before (each case has that control).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kiro_crew.config.paths as paths_mod
import kiro_crew.slack.gateway  # noqa: F401  (binds sel before any patch; see deny_notice)
from kiro_crew import mate_held_rules, mate_worker
from kiro_crew.cron_service.fields import auto_job_approver
from kiro_crew.llm_helpers import ToolApprovalPolicy, _resolve_permission
from kiro_crew.providers.base import EVENT_PERMISSION_REQUEST, LLMEvent

pytestmark = pytest.mark.usefixtures("ample_host_free_memory")

BASE = {
    "name": "kirocrew-mate-worker",
    "prompt": "worker base prompt",
    "tools": ["fs_read", "@kirocrew-core"],
    "allowedTools": ["fs_read"],
    "mcpServers": {"kirocrew-core": {"command": "kc", "args": ["core"]}},
}
_FILE = {"name": "zxc", "mcpServers": {}, "tools": [], "allowedTools": []}
#: A ``toolsSettings`` key outside what the worker carries.
_ODD = {"toolsSettings": {"execute_bash": {"denyByDefault": True}}}
_DENY = {"permissions": {"rules": [{"capability": "web_fetch", "effect": "deny"}]}}


@pytest.fixture
def ledger(monkeypatch, tmp_path) -> Path:
    mates = tmp_path / "home" / "mates" / "agents"
    mates.mkdir(parents=True)
    monkeypatch.setattr(paths_mod, "kiro_home", lambda: tmp_path / "home")
    return mates.parent / "migrated.json"


def _hold(ledger: Path, capability: str) -> None:
    rules = [{"text": "its permissions rule to deny shell", "capability": capability}]
    ledger.write_text(json.dumps({"held_rules": {"zxc": {"member_id": "", "rules": rules}}}))


def _event(kind: str, title: str = "a call") -> LLMEvent:
    return LLMEvent(kind=EVENT_PERMISSION_REQUEST, title=title, request_id="r1", tool_kind=kind)


def test_an_uncarriable_limit_is_held_and_the_worker_still_starts():
    out, held = mate_held_rules.split_policy(True, BASE, {**_ODD, **_DENY}, _FILE)

    assert [h["capability"] for h in held] == ["shell"]
    merged = mate_worker.merge_worker_spec(BASE, mate_worker.worker_additions(BASE, out))
    rules = merged["permissions"]["rules"]
    assert {"capability": "web_fetch", "effect": "deny"} in rules
    assert {"capability": "shell", "effect": "ask"} in rules
    assert "toolsSettings" not in out


def test_a_worker_off_kas_gets_no_restriction_written_and_every_one_held():
    out, held = mate_held_rules.split_policy(False, BASE, _DENY, _FILE)

    assert out == _FILE
    assert [h["capability"] for h in held] == ["web_fetch"]


def test_a_held_limit_covers_its_capability_and_nothing_else(ledger):
    _hold(ledger, "shell")

    assert mate_held_rules.held_for(_event("execute"), agent="zxc")
    assert mate_held_rules.held_for(_event("read"), agent="zxc") == []
    assert mate_held_rules.held_for(_event("execute"), agent="other") == []


def test_a_limit_whose_capability_is_unknown_covers_every_call(ledger):
    _hold(ledger, mate_held_rules.ANY)

    assert mate_held_rules.held_for(_event("read"), agent="zxc")


class _Provider:
    supports_steer = True
    supports_refusal_steer = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def steer(self, message: str) -> bool:
        return True

    async def approve_tool(self, request_id) -> None:
        self.calls.append("approve")

    async def reject_tool(self, request_id) -> None:
        self.calls.append("reject")


async def _resolve(kind: str, approver=None) -> list[str]:
    provider = _Provider()
    with patch("kiro_crew.permission_floor.refusal_for", return_value=None):
        await _resolve_permission(
            provider,
            _event(kind),
            ToolApprovalPolicy.AUTO_APPROVE,
            None,
            on_tool_approval=approver,
            agent="zxc",
        )
    return provider.calls


@pytest.mark.asyncio
async def test_an_auto_cron_asks_a_person_for_a_held_call_and_runs_the_rest(ledger):
    _hold(ledger, "shell")
    card = AsyncMock(return_value=False)
    approver = auto_job_approver(card)

    assert await _resolve("execute", approver) == ["reject"]
    card.assert_awaited_once()
    card.reset_mock()
    assert await _resolve("read", approver) == ["approve"]
    card.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_surface_with_no_approver_refuses_a_held_call(ledger):
    _hold(ledger, "shell")

    assert await _resolve("execute") == ["reject"]
    assert await _resolve("read") == ["approve"]


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [True, False])
async def test_a_trusting_dashboard_thread_shows_the_card_for_a_held_call(tmp_path, held):
    from chat_runner_coverage_helpers import (
        _complete,
        _drive,
        _permission,
        _runner_state,
        _set_stream,
        _slot,
    )

    from kiro_crew.dashboard import chat_runner

    state, client = _runner_state(tmp_path)
    slot = _slot()
    slot._trust = True
    _set_stream(client, [_permission(tool_input=json.dumps({"command": "ls"})), _complete()])
    phrases = ["its permissions rule to deny shell"] if held else []
    with (
        patch.object(mate_held_rules, "held_for", return_value=phrases),
        patch.object(chat_runner, "sel", return_value=MagicMock()),
        patch.object(chat_runner, "tool_approval_timeout_secs", return_value=0.0),
        patch.object(chat_runner, "_name_grant_refusal_off_loop", new=AsyncMock(return_value=None)),
        patch.object(chat_runner, "_name_grant_refusal_for", new=AsyncMock(return_value=None)),
    ):
        await _drive(state, slot)

    if held:
        client.approve_tool.assert_not_awaited()
        assert [m for m in slot.messages if m.get("role") == "permission"]
    else:
        client.approve_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_hook_grant_does_not_approve_a_held_call(ledger):
    """The operator's auto-approve globs approve by name; a held call still asks."""
    from kiro_crew.hooks import TOOL_AUTO_APPROVE, ToolHookResult

    _hold(ledger, "shell")
    hooks = MagicMock()
    hooks.effective_denied_regexes = MagicMock(return_value=[])
    hooks.on_tool_call = MagicMock(return_value=ToolHookResult(action=TOOL_AUTO_APPROVE))

    async def resolve(kind: str, card: AsyncMock) -> list[str]:
        provider = _Provider()
        with patch("kiro_crew.name_grant.refusal_for_event", new=AsyncMock(return_value=None)):
            await _resolve_permission(
                provider,
                _event(kind),
                ToolApprovalPolicy.HOOK_BASED,
                hooks,
                on_tool_approval=card,
                agent="zxc",
            )
        return provider.calls

    card = AsyncMock(return_value=False)
    assert await resolve("execute", card) == ["reject"]
    card.assert_awaited_once()
    card = AsyncMock(return_value=False)
    assert await resolve("read", card) == ["approve"]
    card.assert_not_awaited()


@pytest.mark.parametrize(
    "matcher, capability",
    [
        ("@builtin", mate_held_rules.ANY),
        ("@builtin/execute_bash", "shell"),
        ("@builtin/fs_write", "fs_write"),
        ("@jira", "mcp"),
        ("execute_bash", "shell"),
    ],
)
def test_a_held_hook_is_recorded_under_the_tools_its_matcher_covers(matcher, capability):
    hooks = {"preToolUse": [{"matcher": matcher, "command": "x"}]}

    _out, held = mate_held_rules.split_policy(False, BASE, {"hooks": hooks}, _FILE)

    assert [h["capability"] for h in held] == [capability]


def test_a_bot_moved_now_is_not_told_earlier_work_ran_without_its_limits():
    from kiro_crew import mate_upgrade_notice as notice

    fresh = notice.policy_notice_prompt("x", fresh=True)
    repaired = notice.policy_notice_prompt("x")

    assert "since the upgrade" not in fresh and "Permissions" in fresh
    assert "since the upgrade" in repaired


def test_an_unreadable_ledger_holds_every_call_of_a_bot_and_no_other(ledger, monkeypatch):
    from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig

    ledger.write_text("{not json", encoding="utf-8")
    cfg = KiroCrewConfig()
    cfg.agents = {
        "zxc": KiroCrewAgentConfig(member_id="zxc", kiro_agent="zxc"),
        "kirocrew": KiroCrewAgentConfig(member_id="", kiro_agent="kirocrew"),
    }
    monkeypatch.setattr(KiroCrewConfig, "load", staticmethod(lambda: cfg))

    assert mate_held_rules.held_for(_event("read"), agent="zxc") == [mate_held_rules.UNREADABLE]
    assert mate_held_rules.held_for(_event("read"), agent="kirocrew") == []
