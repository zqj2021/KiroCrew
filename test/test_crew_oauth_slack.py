"""Crew-owned OAuth for Slack's hosted MCP server (``mcp.slack.com/mcp``).

Slack is a pre-registered PUBLIC client: PKCE S256, no client secret unless the
operator adds one, ``client_secret_post`` when there is one. Errors arrive as a
200 with ``{"ok": false}``, refresh tokens rotate and are single-use, and
``auth.revoke`` takes the access token and answers in the body.
"""

from __future__ import annotations

import json
import logging
import time
import types
import urllib.parse

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from kas_agents_helpers import _spec

from kiro_crew.acp import kas_agents
from kiro_crew.connections import crew_oauth, crew_oauth_proxy, registry
from kiro_crew.connections.crew_oauth import CrewOAuthError, Tokens, TokenStore
from kiro_crew.connections.oauth_clients import resolve_oauth_client
from kiro_crew.secrets import SecretVault

SLUG = "slack"
SESSION = "dashboard_chat-crew-oauth-slack"
REDIRECT = "http://localhost:48106/callback"


@pytest.fixture(autouse=True)
def _pinned_modules(_floor_monkeypatch):
    """The code under test imports these at call time; pin them to this file's."""
    import sys

    import kiro_crew.connections as connections_pkg

    for name, module in (("crew_oauth", crew_oauth), ("crew_oauth_proxy", crew_oauth_proxy)):
        _floor_monkeypatch.setitem(sys.modules, f"kiro_crew.connections.{name}", module)
        _floor_monkeypatch.setattr(connections_pkg, name, module, raising=False)
    crew_oauth._refresh_locks.clear()
    crew_oauth_proxy._bridges.clear()
    _floor_monkeypatch.setenv("KIROCREW_BOUND_PORT", "8765")
    _floor_monkeypatch.delenv("KIROCREW_CONNECTIONS_SLACK_CLIENT_ID", raising=False)
    _floor_monkeypatch.delenv("KIROCREW_CONNECTIONS_SLACK_CLIENT_SECRET", raising=False)
    yield
    crew_oauth_proxy._bridges.clear()


def _public(client_id="123.456"):
    return types.SimpleNamespace(client_id=client_id, client_secret=None)


# --- registry and consent URL -----------------------------------------------------


def test_slack_is_a_public_client_on_its_localhost_port():
    provider = registry.get_provider(SLUG)
    assert SLUG in crew_oauth.SUPPORTED_SLUGS and SLUG not in crew_oauth.DCR_SLUGS
    assert provider["mcp_url"] == "https://mcp.slack.com/mcp"
    assert provider["auth"]["confidential"] is False
    assert registry.redirect_uri(provider) == REDIRECT


def test_the_consent_url_is_slacks_user_token_page_with_pkce():
    url = crew_oauth.authorize_url(SLUG, "123.456", REDIRECT, "st", "ch")
    parts = urllib.parse.urlsplit(url)
    query = dict(urllib.parse.parse_qsl(parts.query))
    assert (parts.scheme, parts.netloc, parts.path) == (
        "https",
        "slack.com",
        "/oauth/v2_user/authorize",
    )
    assert query["scope"].split() == list(registry.get_provider(SLUG)["recommended_scopes"])
    assert query["code_challenge_method"] == "S256" and query["code_challenge"] == "ch"
    assert query["redirect_uri"] == REDIRECT and query["response_type"] == "code"
    # Desktop redirects may not ask for bot scopes, and none are sent.
    assert "user_scope" not in query
    from kiro_crew.security import oauth_url_contains_credential

    vetted = crew_oauth.authorize_url(SLUG, "123.456", REDIRECT, "A" * 43, "A" * 43)
    assert oauth_url_contains_credential(vetted) is False


_SLACK_USER_QUERY = (
    "?response_type=code&client_id=3336676.569200954261"
    "&redirect_uri=http%3A%2F%2Flocalhost%3A48106%2Fcallback"
    "&scope=search%3Aread.public+channels%3Ahistory"
    "&code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    "&code_challenge_method=S256"
    "&state=" + ("Kp7mQ2xR" * 12)
)


def test_the_banner_gate_admits_a_high_entropy_state_at_slacks_user_consent_page():
    """Without the exact pair the PKCE values read as a credential and sign-in fails closed."""
    from kiro_crew.security import oauth_url_contains_credential

    url = "https://slack.com/oauth/v2_user/authorize" + _SLACK_USER_QUERY
    assert oauth_url_contains_credential(url) is False


def test_the_banner_gate_still_rejects_it_at_a_neighbouring_path():
    from kiro_crew.security import oauth_url_contains_credential

    url = "https://slack.com/oauth/v2_user/authorize/x" + _SLACK_USER_QUERY
    assert oauth_url_contains_credential(url) is True


# --- the client: public, secret optional -------------------------------------------


def test_a_public_client_resolves_without_a_secret(tmp_path):
    config = {"connections": {"oauth_clients": {SLUG: {"client_id": "123.456"}}}}
    resolved = resolve_oauth_client(
        registry.get_provider(SLUG), config=config, vault=SecretVault(tmp_path), environ={}
    )
    assert resolved is not None and resolved.client_secret is None
    assert resolved.redirect_uri == REDIRECT


def test_crew_signs_in_with_a_public_client_but_not_a_secretless_confidential_one(
    tmp_path, monkeypatch
):
    from kiro_crew.config import loader, paths

    config = {
        "connections": {
            "oauth_clients": {SLUG: {"client_id": "123.456"}, "asana": {"client_id": "a"}}
        }
    }
    monkeypatch.setattr(loader, "read_config_for_update", lambda: config)
    monkeypatch.setattr(paths, "config_dir", lambda: tmp_path)
    assert crew_oauth._client(SLUG).client_id == "123.456"
    with pytest.raises(CrewOAuthError) as err:
        crew_oauth._client("asana")
    assert err.value.code == "client_not_configured"


# --- token endpoint ---------------------------------------------------------------


def _slack(seen: list, token_answer: dict, revoke_answer: dict | None = None):
    async def _token(request):
        seen.append(("token", dict(await request.post())))
        return web.json_response(token_answer)

    async def _revoke(request):
        seen.append(("revoke", dict(await request.post())))
        return web.json_response(revoke_answer or {"ok": True, "revoked": True})

    app = web.Application()
    app.router.add_post("/api/oauth.v2.user.access", _token)
    app.router.add_post("/api/auth.revoke", _revoke)
    return app


async def _serve(monkeypatch, seen, token_answer=None, revoke_answer=None):
    server = TestServer(_slack(seen, token_answer or {"ok": True}, revoke_answer))
    await server.start_server()
    base = str(server.make_url(""))
    monkeypatch.setitem(
        crew_oauth._ENDPOINTS,
        SLUG,
        {
            **crew_oauth._ENDPOINTS[SLUG],
            "token": f"{base}/api/oauth.v2.user.access",
            "revoke": f"{base}/api/auth.revoke",
        },
    )
    return server


async def _swap(tmp_path, monkeypatch, answer, client):
    seen: list = []
    server = await _serve(monkeypatch, seen, answer)
    store = TokenStore(SecretVault(tmp_path))
    sign_in = crew_oauth.SignIn(SLUG)
    sign_in._client, sign_in._redirect = client, REDIRECT
    sign_in._done.set_result("the-code")
    try:
        await sign_in.finish(store)
    finally:
        await server.close()
    return seen, store, sign_in


@pytest.mark.asyncio
async def test_the_code_swap_sends_the_verifier_and_no_secret(tmp_path, monkeypatch):
    answer = {
        "ok": True,
        "access_token": "xoxe.xoxp-1",
        "refresh_token": "xoxe-1-r1",
        "expires_in": 43200,
        "token_type": "user",
        "authed_user": {"id": "U1", "scope": "search:read.public"},
    }
    seen, store, sign_in = await _swap(tmp_path, monkeypatch, answer, _public())
    form = seen[0][1]
    assert form == {
        "grant_type": "authorization_code",
        "client_id": "123.456",
        "code": "the-code",
        "redirect_uri": REDIRECT,
        "code_verifier": sign_in._verifier,
    }
    held = store.read(SLUG)
    assert held is not None
    assert (held[0].access, held[0].refresh, held[0].client) == (
        "xoxe.xoxp-1",
        "xoxe-1-r1",
        "123.456",
    )
    assert held[0].expires_at is not None and held[0].expires_at > time.time() + 43000


@pytest.mark.asyncio
async def test_an_operator_secret_is_posted_in_the_form(tmp_path, monkeypatch):
    answer = {"ok": True, "access_token": "a", "refresh_token": "r", "expires_in": 43200}
    client = types.SimpleNamespace(client_id="123.456", client_secret="sec")
    seen, _, _ = await _swap(tmp_path, monkeypatch, answer, client)
    assert seen[0][1]["client_secret"] == "sec"


@pytest.mark.asyncio
async def test_slacks_200_ok_false_is_a_failed_swap_that_stores_nothing(tmp_path, monkeypatch):
    with pytest.raises(CrewOAuthError) as err:
        await _swap(tmp_path, monkeypatch, {"ok": False, "error": "invalid_code"}, _public())
    assert err.value.code == "invalid_code"
    assert TokenStore(SecretVault(tmp_path)).read(SLUG) is None


@pytest.mark.asyncio
async def test_a_refresh_rotates_the_single_use_refresh_token(tmp_path, monkeypatch):
    seen: list = []
    answer = {"ok": True, "access_token": "a2", "refresh_token": "r2", "expires_in": 43200}
    server = await _serve(monkeypatch, seen, answer)
    monkeypatch.setattr(crew_oauth, "_client", lambda s: _public())
    store = TokenStore(SecretVault(tmp_path))
    store.replace(SLUG, Tokens("a1", "r1", time.time() - 10, "", None, "123.456"))
    try:
        assert await crew_oauth.access_token(SLUG, store=store) == "a2"
    finally:
        await server.close()
    assert seen[0][1] == {
        "grant_type": "refresh_token",
        "refresh_token": "r1",
        "client_id": "123.456",
    }
    held = store.read(SLUG)
    assert held is not None and (held[0].access, held[0].refresh) == ("a2", "r2")
    assert held[0].client == "123.456"


@pytest.mark.asyncio
async def test_a_dead_refresh_token_asks_for_a_reconnect_and_keeps_the_pair(tmp_path, monkeypatch):
    seen: list = []
    server = await _serve(monkeypatch, seen, {"ok": False, "error": "invalid_refresh_token"})
    monkeypatch.setattr(crew_oauth, "_client", lambda s: _public())
    store = TokenStore(SecretVault(tmp_path))
    store.replace(SLUG, Tokens("a1", "r1", time.time() - 10, ""))
    try:
        with pytest.raises(CrewOAuthError) as err:
            await crew_oauth.access_token(SLUG, store=store)
    finally:
        await server.close()
    assert err.value.code == "needs_auth"
    assert store.read(SLUG) is not None


# --- Disconnect -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_revokes_the_access_token(tmp_path, monkeypatch):
    seen: list = []
    server = await _serve(monkeypatch, seen)

    def _gone(slug):
        raise CrewOAuthError("client_not_configured")

    # The revoke takes the token alone: a removed OAuth Apps record never blocks it.
    monkeypatch.setattr(crew_oauth, "_client", _gone)
    store = TokenStore(SecretVault(tmp_path))
    store.replace(SLUG, Tokens("a1", "r1", None, ""))
    try:
        result = await crew_oauth.disconnect(SLUG, store=store)
    finally:
        await server.close()
    assert result == {"removed": True, "revoked": True}
    assert seen == [("revoke", {"token": "a1"})]
    assert not store.present(SLUG)


@pytest.mark.asyncio
async def test_a_200_that_says_not_revoked_is_reported_and_still_deletes(tmp_path, monkeypatch):
    seen: list = []
    server = await _serve(monkeypatch, seen, revoke_answer={"ok": False, "error": "token_expired"})
    store = TokenStore(SecretVault(tmp_path))
    store.replace(SLUG, Tokens("a1", "r1", None, ""))
    try:
        result = await crew_oauth.disconnect(SLUG, store=store)
    finally:
        await server.close()
    assert result == {"removed": True, "revoked": False}
    assert not store.present(SLUG)


# --- custody: the vault, never a log, a spec or the session ------------------------


@pytest.mark.asyncio
async def test_the_pair_lives_only_in_the_vault(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    answer = {"ok": True, "access_token": "xoxp-SECRET-A", "refresh_token": "xoxe-SECRET-R"}
    _, store, _ = await _swap(tmp_path, monkeypatch, answer, _public())
    crew_oauth._audit("connections_crew_oauth_sign_in", "ok", SLUG)
    assert "SECRET" not in caplog.text
    # Encrypted at rest: no file under the data home carries either token.
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"SECRET" not in path.read_bytes(), path
    assert store.present(SLUG)


def test_the_kas_projection_carries_the_bridge_not_the_client_or_token(tmp_path, monkeypatch):
    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    monkeypatch.setattr(crew_oauth, "enabled", lambda: True)
    monkeypatch.setattr(crew_oauth, "_chat_backend_bridged", lambda: True)
    url = registry.get_provider(SLUG)["mcp_url"]
    spec = _spec(
        tools=[f"@{SLUG}"],
        allowedTools=[f"@{SLUG}"],
        mcpServers={SLUG: {"url": url, "oauth": {"clientId": "123.456"}}},
    )
    out = kas_agents.to_client_custom_agent("kirocrew", spec, "p", session_key=SESSION)
    entry = out["mcpServers"][SLUG]
    assert "url" not in entry and "oauth" not in entry
    assert url not in json.dumps(out) and "123.456" not in json.dumps(out)
    assert crew_oauth_proxy.is_connection_server(SESSION, SLUG)


# --- kiro-cli projection: a user's own Slack client survives -----------------------


def _project(monkeypatch, tmp_path, entry, *, slug=SLUG, raw=(), kas=False):
    from kiro_crew import agent as agent_mod
    from kiro_crew.agent_materialization import default_spec_commit

    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path))
    monkeypatch.setattr(crew_oauth, "strips_spec_secret", lambda s: kas)
    monkeypatch.setattr(agent_mod, "_load_json", lambda p: {})
    return default_spec_commit._apply_operator_oauth_client(
        slug, entry, managed=True, raw_scopes=tuple(raw)
    )


SLACK = "https://mcp.slack.com/mcp"
RENDERED = {"url": SLACK, "oauth": {"clientId": "op", "clientSecret": "s", "redirectUri": "r"}}


@pytest.mark.parametrize("kas", [False, True])
def test_an_unconfigured_slack_keeps_the_id_the_user_wrote_in_a_raw_scope(
    tmp_path, monkeypatch, kas
):
    mine = {
        "clientId": "mine",
        "redirectUri": "http://localhost:3118/callback",
        "clientSecret": "u",
    }
    raw = ({}, {SLUG: {"url": SLACK, "oauth": dict(mine)}})
    out = _project(monkeypatch, tmp_path, dict(RENDERED, oauthScopes=["x"]), raw=raw, kas=kas)
    # The user's whole client, as written: id, the redirect their app registered,
    # and its secret if the app needs one.
    assert out["oauth"] == mine and out["oauthScopes"] == ["x"]


@pytest.mark.parametrize("kas", [False, True])
def test_a_projected_operator_id_never_survives_without_a_raw_one(tmp_path, monkeypatch, kas):
    """The previous render's id may be the removed OAuth app's: it goes."""
    assert "oauth" not in _project(monkeypatch, tmp_path, dict(RENDERED), kas=kas)
    # A raw entry at another URL is somebody else's server, not this user's id.
    raw = ({SLUG: {"url": "https://elsewhere.example/mcp", "clientId": "x"}},)
    assert "oauth" not in _project(monkeypatch, tmp_path, dict(RENDERED), raw=raw, kas=kas)


def test_an_unconfigured_confidential_provider_still_loses_all_three_keys(tmp_path, monkeypatch):
    url = registry.get_provider("asana")["mcp_url"]
    entry = {"url": url, "oauth": {"clientId": "c", "clientSecret": "s", "redirectUri": "r"}}
    raw = ({"asana": {"url": url, "oauth": {"clientId": "c"}}},)
    assert "oauth" not in _project(monkeypatch, tmp_path, entry, slug="asana", raw=raw)


# --- the user's own Slack entry: its clientId and scopes are reused, never edited --


def _servers(monkeypatch, *entries):
    from kiro_crew import mcp_discovery

    rows = [types.SimpleNamespace(name=n, url=u, client_id=c, scopes=s) for n, u, c, s in entries]
    monkeypatch.setattr(mcp_discovery, "list_servers", lambda: rows)


def test_the_hint_reads_the_users_slack_entry_at_slacks_own_url(monkeypatch):
    url = registry.get_provider(SLUG)["mcp_url"]
    _servers(
        monkeypatch,
        ("slack", "https://elsewhere.example/mcp", "other", ["x"]),
        ("slack", url, "123.456", ["search:read.public", "users:read"]),
    )
    assert crew_oauth.server_oauth_hints(SLUG) == (
        "123.456",
        ("search:read.public", "users:read"),
    )


def test_the_hint_ignores_another_url_and_a_confidential_provider(monkeypatch):
    asana = registry.get_provider("asana")["mcp_url"]
    _servers(
        monkeypatch,
        ("slack", "https://elsewhere.example/mcp", "other", []),
        ("asana", asana, "a", []),
    )
    assert crew_oauth.server_oauth_hints(SLUG) == ("", ())
    assert crew_oauth.server_oauth_hints("asana") == ("", ())


def test_with_no_oauth_apps_record_crew_signs_in_with_the_entrys_client_id(tmp_path, monkeypatch):
    from kiro_crew.config import loader, paths

    monkeypatch.setattr(loader, "read_config_for_update", lambda: {})
    monkeypatch.setattr(paths, "config_dir", lambda: tmp_path)
    monkeypatch.setattr(crew_oauth, "server_oauth_hints", lambda s: ("123.456", ("users:read",)))
    client = crew_oauth._client(SLUG)
    assert (client.client_id, client.client_secret, client.client_id_source) == (
        "123.456",
        None,
        "server",
    )
    assert client.redirect_uri == REDIRECT
    # The OAuth Apps record still wins over the entry.
    config = {"connections": {"oauth_clients": {SLUG: {"client_id": "777.888"}}}}
    monkeypatch.setattr(loader, "read_config_for_update", lambda: config)
    assert crew_oauth._client(SLUG).client_id == "777.888"


def test_the_card_counts_the_entrys_client_id_only_for_a_public_provider():
    from kiro_crew.connections.oauth_clients import oauth_client_view

    view = oauth_client_view(
        registry.get_provider(SLUG),
        config={},
        vault_names=set(),
        environ={},
        server_client_id="1.2",
    )
    assert view["configured"] is True and view["client_id_source"] == "server"
    asana = oauth_client_view(
        registry.get_provider("asana"),
        config={},
        vault_names=set(),
        environ={},
        server_client_id="1",
    )
    assert asana["configured"] is False and asana["client_id"] is None


def test_the_card_reads_the_entry_with_the_preview_off_too(monkeypatch):
    """With the preview off kiro-cli signs in with the same id, so Connect stays offered."""
    from kiro_crew.dashboard.handlers import connections as handler

    monkeypatch.setattr(crew_oauth, "server_oauth_hints", lambda s: ("123.456", ()))
    monkeypatch.setattr(crew_oauth, "custody_on", lambda s: False)
    assert crew_oauth.server_client_id(SLUG) == "123.456"
    assert handler._oauth_client_configured(registry.get_provider(SLUG)) is True
    monkeypatch.setattr(crew_oauth, "server_oauth_hints", lambda s: ("", ()))
    assert handler._oauth_client_configured(registry.get_provider(SLUG)) is False


def test_the_entrys_scopes_replace_the_registrys_in_the_consent_url():
    url = crew_oauth.authorize_url(SLUG, "1.2", REDIRECT, "st", "ch", ("users:read", "im:read"))
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert query["scope"] == "users:read im:read"


@pytest.mark.asyncio
async def test_a_sign_in_asks_for_the_entrys_scopes(monkeypatch):
    provider = dict(registry.get_provider(SLUG))
    provider["auth"] = {**provider["auth"], "redirect_port": 0}
    monkeypatch.setattr(registry, "get_provider", lambda s: provider)
    monkeypatch.setattr(crew_oauth, "_client", lambda s: _public())
    monkeypatch.setattr(crew_oauth, "server_oauth_hints", lambda s: ("123.456", ("users:read",)))
    sign_in = crew_oauth.SignIn(SLUG)
    try:
        url = await sign_in.start()
    finally:
        await sign_in.shutdown()
    query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert query["scope"] == "users:read" and query["client_id"] == "123.456"


# --- every copy or scrub keeps a public client's user-owned id ---------------------

SLACK_URL = "https://mcp.slack.com/mcp"


def test_the_shared_strip_drops_all_three_and_restores_only_the_given_user_client():
    from kiro_crew.connections.oauth_clients import strip_preregistered_oauth_client
    from kiro_crew.connections.oauth_clients import user_oauth_client as user_client

    entry = {"url": SLACK_URL, "oauth": {"clientId": "op", "clientSecret": "s", "redirectUri": "r"}}
    assert "oauth" not in strip_preregistered_oauth_client(entry)
    mine = {"clientId": "mine", "redirectUri": "http://localhost:3118/callback"}
    assert strip_preregistered_oauth_client(entry, user_oauth=mine)["oauth"] == mine
    slack, asana = registry.get_provider(SLUG), registry.get_provider("asana")
    raw = [{SLUG: {"url": SLACK_URL, "oauth": {**mine, "issuer": "i"}}}]
    assert user_client(slack, SLUG, SLACK_URL, raw) == mine
    assert user_client(slack, SLUG, "https://elsewhere.example/mcp", raw) == {}
    # No usable id: nothing is the user's, not even a stray redirect.
    no_id = [{SLUG: {"url": SLACK_URL, "oauth": {"redirectUri": "r"}}}]
    assert user_client(slack, SLUG, SLACK_URL, no_id) == {}
    asana_raw = [{"asana": {"url": asana["mcp_url"], "clientId": "c"}}]
    assert user_client(asana, "asana", asana["mcp_url"], asana_raw) == {}


def _scopes(monkeypatch, tmp_path, store: dict, kiro_global: dict):
    from kiro_crew.dashboard.handlers import mcp as mcp_handlers

    paths = tmp_path / "mcp.json", tmp_path / "kiro-mcp.json"
    for path, servers in zip(paths, (store, kiro_global)):
        path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    monkeypatch.setattr(mcp_handlers, "_kirocrew_mcp_json", lambda: paths[0])
    monkeypatch.setattr(mcp_handlers, "_GLOBAL_MCP_JSON", paths[1])
    monkeypatch.setattr(mcp_handlers, "_extra_mcp_scopes", lambda: [])
    return mcp_handlers, paths


@pytest.mark.parametrize(
    ("raw_global", "want"),
    [
        # The user's own id in the Kiro global survives a move to Crew-only.
        (
            {SLUG: {"url": SLACK_URL, "oauth": {"clientId": "mine", "redirectUri": "u"}}},
            {"clientId": "mine", "redirectUri": "u"},
        ),
        # Only the operator's projected id exists: the copy carries none of it.
        ({}, None),
    ],
)
def test_a_scope_copy_takes_the_raw_id_never_the_rendered_one(
    tmp_path, monkeypatch, raw_global, want
):
    mcp_handlers, _ = _scopes(monkeypatch, tmp_path, {}, raw_global)
    agents = tmp_path / "agents"
    agents.mkdir()
    rendered = {
        "url": SLACK_URL,
        "oauth": {"clientId": "op", "clientSecret": "s", "redirectUri": "r", "oauthScopes": ["a"]},
    }
    (agents / "kirocrew.json").write_text(
        json.dumps({"mcpServers": {SLUG: rendered}}), encoding="utf-8"
    )
    monkeypatch.setattr(mcp_handlers, "kiro_agents_dir", lambda: agents)
    monkeypatch.setattr(mcp_handlers, "_read_agent_spec", lambda p, **_k: json.loads(p.read_text()))
    found = mcp_handlers._find_server_spec_anywhere(SLUG)
    assert found["oauth"] == {"oauthScopes": ["a"], **(want or {})}


def test_a_client_scrub_leaves_the_users_slack_client_id(tmp_path, monkeypatch):
    """The user's own file keeps their whole client: the scrub writes nothing."""
    mine = {"clientId": "mine", "clientSecret": "s", "redirectUri": "u"}
    user = {SLUG: {"url": SLACK_URL, "oauth": dict(mine)}}
    mcp_handlers, paths = _scopes(monkeypatch, tmp_path, {}, user)
    assert mcp_handlers._scrub_preregistered_oauth_copies(SLUG)["kiroGlobal"] == "noop"
    after = json.loads(paths[1].read_text(encoding="utf-8"))["mcpServers"][SLUG]
    assert after["oauth"] == mine
