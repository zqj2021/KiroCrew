"""The governance row judges a KAS shell call on its recovered command.

KAS titles its shell call ``Run Command``, so a ``commands`` rule read off the
title alone never met the command: a policy deny such as ``*backdoor*`` let
``echo backdoor`` through to the read-only row's auto-approve. The row also asks
the recovered command, so the deny refuses it before any grant or classify row.
"""

from __future__ import annotations

from test_kas_allowed_commands import (  # noqa: F401 - _no_profiles is an autouse fixture
    _install_governance,
    _judge,
    _no_profiles,
    _shell,
)

from kiro_crew.hooks import TOOL_ALLOW, TOOL_AUTO_APPROVE, TOOL_DENY


def _deny_backdoor() -> None:
    _install_governance({"commands": {"mode": "deny", "deny": ["*backdoor*"]}})


def test_a_policy_deny_refuses_the_command_a_read_only_call_runs() -> None:
    _deny_backdoor()
    verdict = _judge(_shell("echo backdoor"), [])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


def test_a_policy_deny_refuses_a_granted_chain_holding_the_command() -> None:
    _deny_backdoor()
    verdict = _judge(_shell("git status && echo backdoor"), ["git status", "echo .*"])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


def test_a_harmless_read_only_command_is_still_auto_approved() -> None:
    _deny_backdoor()
    verdict = _judge(_shell("ls"), [])
    assert (verdict.action, verdict.tier) == (TOOL_AUTO_APPROVE, "read-only")


def test_a_command_the_deny_does_not_name_still_asks() -> None:
    _deny_backdoor()
    verdict = _judge(_shell("git status --porcelain"), ["git status.*"])
    assert verdict.action == TOOL_ALLOW


# ── an allow-mode ``commands`` policy ─────────────────────────────────────────


def _allow_git_status() -> None:
    _install_governance(
        {"commands": {"mode": "allow", "allow": ["git status", "git status --porcelain"]}}
    )


def test_an_allow_mode_policy_admits_a_listed_command() -> None:
    """The title ``Run Command`` is not on the list, and it names no command: the
    recovered command is what the list is about. The call then prompts as usual."""
    _allow_git_status()
    verdict = _judge(_shell("git status --porcelain"), [])
    assert verdict.action == TOOL_ALLOW
    assert verdict.tier != "governance"


def test_an_allow_mode_policy_lets_a_listed_command_take_its_grant() -> None:
    _allow_git_status()
    verdict = _judge(_shell("git status"), ["git status"])
    assert verdict.action != TOOL_DENY


def test_an_allow_mode_policy_still_refuses_an_unlisted_command() -> None:
    _allow_git_status()
    verdict = _judge(_shell("ls"), [])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


def test_an_allow_mode_policy_refuses_a_chain_with_an_unlisted_part() -> None:
    _allow_git_status()
    verdict = _judge(_shell("git status && echo x"), ["git status", "echo .*"])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


def test_an_explicit_deny_on_the_title_still_refuses() -> None:
    _install_governance({"commands": {"mode": "deny", "deny": ["Run Command"]}})
    verdict = _judge(_shell("git status"), [])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


# ── an allow-mode glob is checked against each chained command ────────────────


def _allow_git_glob() -> None:
    _install_governance({"commands": {"mode": "allow", "allow": ["git *"]}})


def test_an_allow_mode_glob_refuses_a_chain_whose_glob_spans_the_separator() -> None:
    """``git *`` matches the whole ``git status && python3 -c x`` because ``*``
    crosses ``&&``; the chained ``python3 -c x`` is not listed, so it refuses."""
    _allow_git_glob()
    verdict = _judge(_shell("git status && python3 -c x"), [])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


def test_an_allow_mode_glob_admits_a_chain_whose_every_part_is_listed() -> None:
    _allow_git_glob()
    verdict = _judge(_shell("git status && git log"), [])
    assert verdict.tier != "governance"
    assert verdict.action != TOOL_DENY


def test_an_allow_mode_glob_refuses_a_command_that_cannot_be_split() -> None:
    _allow_git_glob()
    verdict = _judge(_shell("git status > out.txt"), [])
    assert (verdict.action, verdict.tier) == (TOOL_DENY, "governance")


def test_a_deny_mode_policy_permits_a_command_that_cannot_be_split() -> None:
    _deny_backdoor()
    verdict = _judge(_shell("git status > out.txt"), [])
    assert verdict.tier != "governance"
