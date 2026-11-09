"""A KAS project-only agent's own ``hooks``: whether they run, and the notice state.

Split out of :mod:`kiro_crew.agent_sdk.spec_hooks`, which re-exports every name
here, so callers keep importing them from there.
"""

from __future__ import annotations

from typing import Any, NamedTuple


def _live_bot_owns(agent_id: str) -> bool:
    """Whether a live Bot's own crewmate file answers *agent_id* before the checkout.

    The rule the KAS projection applies (``acp.harness.kas._bot_owns_mate_file``):
    a file a deleted Bot left behind yields to a checkout spec of the same name, so
    its presence alone must not withhold that folder agent's hooks.
    """
    from kiro_crew.acp.kas_agents import agent_spec_absent
    from kiro_crew.config.paths import mate_agents_dir
    from kiro_crew.mate_capabilities import binding_is_mate

    return not agent_spec_absent(mate_agents_dir(), agent_id) and binding_is_mate(agent_id)


def _consented_kas_project_hooks(agent_id: str, checkout: str) -> dict[str, Any] | None:
    """A KAS project-only agent's ``hooks``, when *checkout*'s choice lets them run.

    Mirrors the order KAS resolves the agent in (``acp.harness.kas.resolve_projected_spec``):
    called only when the user level does not declare *agent_id*, and a crewmate's own
    config of that name wins over the checkout's. Only ``hooks`` is handed back, so no
    other key of the checkout's spec (``mcpServers`` among them) reaches a reader here.
    ``None`` when the checkout does not declare the agent or its hooks are off; a
    declared spec that cannot be read raises, so PreToolUse fails closed.
    """
    from kiro_crew.acp.session_mcp import project_agent_spec
    from kiro_crew.project_hooks_consent import hooks_enabled

    if _live_bot_owns(agent_id):
        return None
    declared, spec = project_agent_spec(agent_id, checkout)
    if not declared or not hooks_enabled(checkout)[0]:
        return None
    if spec is None:
        raise ValueError(f"the project spec for agent {agent_id!r} could not be read")
    return {"hooks": spec["hooks"]} if "hooks" in spec else {}


class KasProjectHooks(NamedTuple):
    """What the session-start notice says about a KAS folder agent's hooks."""

    #: ``""`` (no notice), ``"off"``, ``"on"`` or ``"kiro_cli"``.
    state: str
    #: The checkout's agent file the hooks come from.
    path: str = ""
    #: ``(event, command)`` for each hook, as Crew would fire it.
    hooks: tuple[tuple[str, str], ...] = ()


def kas_project_hooks_state(agent_id: str, checkout: str | None) -> KasProjectHooks:
    """Whether a KAS session's folder agent carries hooks, whether they run, and which.

    ``state`` is ``""`` when *agent_id* is not a project-only agent of *checkout*
    with non-empty ``hooks`` (the user level or a crewmate config declares it, or
    the folder's spec sets none or cannot be read); else ``"on"`` (the owner turned
    them on in Crew), ``"kiro_cli"`` (they run because kiro-cli trusts the folder),
    or ``"off"``. For the session-start notice, which names the file and lists the
    commands so the owner can read them before turning them on.
    """
    from kiro_crew.acp.kas_agents import agent_spec_absent

    # circular import: spec_hooks re-exports this module.
    from kiro_crew.acp.session_mcp import _project_spec_path_for, project_agent_spec
    from kiro_crew.agent_sdk import spec_hooks
    from kiro_crew.config.paths import spec_dir_for
    from kiro_crew.project_hooks_consent import SOURCE_KIRO_CLI, hooks_enabled

    if not checkout or not agent_spec_absent(spec_dir_for(agent_id), agent_id):
        return KasProjectHooks("")
    if _live_bot_owns(agent_id):
        return KasProjectHooks("")
    declared, spec = project_agent_spec(agent_id, checkout)
    if not declared or not isinstance(spec, dict) or not spec.get("hooks"):
        return KasProjectHooks("")
    path = _project_spec_path_for(agent_id, checkout)
    listed = tuple((h.event, h.command) for h in spec_hooks._convert(agent_id, spec)[0])
    enabled, source = hooks_enabled(checkout)
    if not enabled:
        state = "off"
    else:
        state = "kiro_cli" if source == SOURCE_KIRO_CLI else "on"
    return KasProjectHooks(state, str(path) if path else "", listed)
