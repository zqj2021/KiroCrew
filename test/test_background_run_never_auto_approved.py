"""``@kirocrew-cron/background_run`` is never auto-approved, by any grant, on any route.

It runs an arbitrary command, so like ``execute_bash`` every call has to reach the
approval gate. A covering grant (``@kirocrew-cron``, ``@kirocrew-cron/*``, ``*``) is
replaced by the server's other verbs; an exact or verb-glob grant of it is dropped.
Pinned on the three routes a grant reaches a session by: the kiro-cli spec writers
(``auto_approve``), ``kirocrew doctor``'s repair, and the KAS projection.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from cli_helpers import (  # noqa: F401 -- the autouse fixture registers by import
    _doctor_reads_the_warm_probe_cache,
    _healthy_agent_file,
)
from kas_agents_helpers import _spec

import kiro_crew.mcp_discovery_managed as managed
from kiro_crew import agent
from kiro_crew.acp import kas_agents
from kiro_crew.acp.kas_agents import to_client_custom_agent
from kiro_crew.agent_materialization import auto_approve
from kiro_crew.agent_sdk.drivers.acp import derived_agent_permissions
from kiro_crew.agent_sdk.mcp_refs import _glob_hits, entry_reaches

BACKGROUND_RUN = "@kirocrew-cron/background_run"

#: The cron server in another letter case, which ``entry_reaches`` folds onto it.
MIXED_CASE = "@KiroCrew-Cron"  # brand-ok - a server-name spelling, not the product name

#: Every spelling that covers ``background_run`` without naming it alone.
COVERING = ["@kirocrew-cron", "@kirocrew-cron/", "@kirocrew-cron/*", "*", MIXED_CASE]


def _other_cron_verbs() -> list[str]:
    verbs = managed.managed_tool_names_for_policy("kirocrew-cron")
    assert verbs and "background_run" in verbs, verbs
    return [f"@kirocrew-cron/{v}" for v in verbs if v != "background_run"]


def _covers(entries: list[str]) -> list[str]:
    return [e for e in entries if entry_reaches(e, BACKGROUND_RUN)]


@pytest.fixture
def ungoverned(monkeypatch):
    """No ceiling and no profile: the host the finding names."""
    monkeypatch.setattr(agent, "may_skip_gate_now", lambda ref: True)
    monkeypatch.setattr(kas_agents, "may_skip_gate_now", lambda ref: True)


# ── kiro-cli write route ──────────────────────────────────────────────────


@pytest.mark.parametrize("grant", COVERING)
def test_the_ceiling_narrows_a_covering_grant_to_the_other_cron_verbs(ungoverned, grant):
    config = {"allowedTools": ["fs_read", grant, "@kirocrew-core/wait"]}
    auto_approve._apply_allowed_tools_ceiling(config, source="test")
    allowed = config["allowedTools"]
    assert not _covers(allowed), allowed
    for ref in _other_cron_verbs():
        assert ref in allowed, ref
    assert "@kirocrew-core/wait" in allowed
    assert len(allowed) == len(set(allowed))


@pytest.mark.parametrize("grant", [BACKGROUND_RUN, "@kirocrew-cron/back*", "background_run"])
def test_the_ceiling_drops_a_grant_that_names_only_background_run(ungoverned, grant):
    config = {"allowedTools": ["@kirocrew-cron/cron_list", grant]}
    auto_approve._apply_allowed_tools_ceiling(config, source="test")
    assert config["allowedTools"] == ["@kirocrew-cron/cron_list"]


@pytest.mark.parametrize("grant", [*COVERING, BACKGROUND_RUN])
def test_the_predicate_never_lets_a_covering_grant_on_a_list(ungoverned, grant):
    assert auto_approve._may_auto_approve(grant) is False
    assert auto_approve._may_auto_approve("@kirocrew-cron/cron_list") is True


def test_build_agent_config_carries_no_grant_covering_background_run(ungoverned, tmp_path):
    """The shipped template plus a user override that widened cron to the whole server."""
    override = tmp_path / "agent.json"
    override.write_text(json.dumps({"allowedTools": ["fs_read", "@kirocrew-cron"]}))
    with patch("kiro_crew.agent._user_overrides_path", return_value=override):
        allowed = agent.build_agent_config()["allowedTools"]
    assert not _covers(allowed), allowed
    for ref in _other_cron_verbs():
        assert ref in allowed, ref


def test_the_rebuild_final_pass_narrows_an_on_disk_whole_server_grant(ungoverned):
    """``rebuild_agent_config`` keeps entries from the spec on disk, which may predate the fix."""
    config = {"allowedTools": ["@kirocrew-cron", "@kirocrew-cron/cron_list"]}
    auto_approve.final_ceiling_pass(config)
    assert not _covers(config["allowedTools"])
    assert config["allowedTools"].count("@kirocrew-cron/cron_list") == 1


def test_the_conductor_grant_filter_narrows_too(ungoverned):
    granted = auto_approve._filter_auto_approve(("@kirocrew-cron",), source="test")
    assert not _covers(granted)
    assert "@kirocrew-cron/cron_list" in granted


def test_the_ceiling_still_judges_each_substituted_verb(monkeypatch):
    monkeypatch.setattr(agent, "may_skip_gate_now", lambda ref: ref != "@kirocrew-cron/cron_add")
    config = {"allowedTools": ["@kirocrew-cron"]}
    auto_approve._apply_allowed_tools_ceiling(config, source="test")
    assert "@kirocrew-cron/cron_add" not in config["allowedTools"]
    assert "@kirocrew-cron/cron_list" in config["allowedTools"]


def test_an_unreadable_cron_declaration_fails_closed_without_dropping_named_verbs(
    ungoverned, monkeypatch
):
    """No verb list to substitute: the covering grant goes, the per-verb ones stay."""
    real = managed.managed_tool_names_for_policy
    monkeypatch.setattr(
        managed,
        "managed_tool_names_for_policy",
        lambda name: None if name == "kirocrew-cron" else real(name),
    )
    config = {"allowedTools": ["@kirocrew-cron", "@kirocrew-cron/cron_list"]}
    auto_approve._apply_allowed_tools_ceiling(config, source="test")
    assert config["allowedTools"] == ["@kirocrew-cron/cron_list"]


def test_the_withhold_is_audited(ungoverned):
    with patch.object(agent, "sel") as sel:
        auto_approve._apply_allowed_tools_ceiling(
            {"allowedTools": ["@kirocrew-cron"]}, source="test"
        )
    (call,) = sel.return_value.log_api_access.call_args_list
    assert call.kwargs["operation"] == "mcp_auto_approve_withheld"
    assert "@kirocrew-cron (narrowed to" in call.kwargs["resources"]


def test_the_derived_kas_block_on_disk_carries_no_covering_allow(ungoverned):
    config = {"allowedTools": ["@kirocrew-cron"]}
    auto_approve._apply_allowed_tools_ceiling(config, source="test")
    block = derived_agent_permissions(config["allowedTools"], "kirocrew.json")
    assert not _kas_allows_background_run(block)
    assert _kas_allows("kirocrew-cron/cron_list", block)


# ── KAS projection route ──────────────────────────────────────────────────


def _kas_allows(tool: str, permissions: dict | None) -> bool:
    for rule in (permissions or {}).get("rules", []):
        if rule["capability"] != "mcp" or rule["effect"] != "allow":
            continue
        if any(_glob_hits(tool, p) for p in rule.get("match") or ["*"]):
            return True
    return False


def _kas_allows_background_run(permissions: dict | None) -> bool:
    return _kas_allows("kirocrew-cron/background_run", permissions)


@pytest.mark.parametrize("grant", COVERING)
def test_the_projection_narrows_a_spec_on_disk_that_predates_the_fix(ungoverned, grant):
    spec = _spec(
        tools=["@kirocrew-cron"],
        allowedTools=[grant],
        mcpServers={"kirocrew-cron": {"command": "x"}},
    )
    permissions = to_client_custom_agent("a", spec, "p").get("permissions")
    assert not _kas_allows_background_run(permissions), permissions
    if grant != MIXED_CASE:  # KAS matches servers case-sensitively
        assert _kas_allows("kirocrew-cron/cron_list", permissions), permissions


def test_ceiling_permitted_withholds_every_covering_spelling(ungoverned):
    permitted = kas_agents._ceiling_permitted([*COVERING, BACKGROUND_RUN, "*_run"], "a")
    assert not _covers(permitted), permitted
    assert "@kirocrew-cron/cron_list" in permitted


def test_an_authored_kas_allow_cannot_cover_background_run(ungoverned):
    """A pure-KAS spec (no ``allowedTools``) whose own block allows the whole cron server."""
    spec = _spec(
        tools=["@kirocrew-cron"],
        mcpServers={"kirocrew-cron": {"command": "x"}},
        permissions={
            "rules": [{"capability": "mcp", "match": ["kirocrew-cron/*"], "effect": "allow"}]
        },
    )
    permissions = to_client_custom_agent("a", spec, "p").get("permissions")
    assert not _kas_allows_background_run(permissions), permissions


def test_an_authored_kas_deny_still_travels(ungoverned):
    rule = {"capability": "mcp", "match": ["kirocrew-cron/cron_add"], "effect": "deny"}
    spec = _spec(
        tools=["@kirocrew-cron"],
        mcpServers={"kirocrew-cron": {"command": "x"}},
        permissions={"rules": [rule]},
    )
    assert rule in to_client_custom_agent("a", spec, "p")["permissions"]["rules"]


# ── doctor route ──────────────────────────────────────────────────────────


def _probe_nothing():
    from kiro_crew.mcp_discovery import McpServerInfo

    async def fake(target: McpServerInfo) -> McpServerInfo:
        target.status, target.tools, target.error = "ok", [], ""
        return target

    return patch("kiro_crew.cli_doctor.probe_server", side_effect=fake)


def test_doctor_mints_no_grant_covering_background_run(tmp_path):
    from kiro_crew.cli_doctor import _doctor_mcp_tools

    agent_path = tmp_path / "kirocrew.json"
    _healthy_agent_file(agent_path)
    data = json.loads(agent_path.read_text(encoding="utf-8"))
    data["allowedTools"] = []
    agent_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    with (
        patch("kiro_crew.cli_doctor.may_skip_gate_now", return_value=True),
        _probe_nothing(),
    ):
        _doctor_mcp_tools(agent_path, [])
    allowed = json.loads(agent_path.read_text(encoding="utf-8"))["allowedTools"]
    assert not _covers(allowed), allowed
    for ref in _other_cron_verbs():
        assert ref in allowed, ref
    assert "@kirocrew-core" in allowed


def test_doctor_is_idempotent_on_its_narrowed_grant(tmp_path):
    from kiro_crew.cli_doctor import _doctor_mcp_tools

    agent_path = tmp_path / "kirocrew.json"
    _healthy_agent_file(agent_path)
    with (
        patch("kiro_crew.cli_doctor.may_skip_gate_now", return_value=True),
        _probe_nothing(),
    ):
        _doctor_mcp_tools(agent_path, [])
        first = agent_path.read_text(encoding="utf-8")
        _doctor_mcp_tools(agent_path, [])
    assert agent_path.read_text(encoding="utf-8") == first
