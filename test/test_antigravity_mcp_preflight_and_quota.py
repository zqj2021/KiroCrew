"""Tests for Google Antigravity MCP initialize preflight and remaining-quota API."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from kiro_crew.acp import antigravity_mcp_preflight as preflight
from kiro_crew.dashboard.handlers import antigravity_usage_api as usage_api


@pytest.fixture(autouse=True)
def _clear_caches() -> Any:
    preflight.clear_preflight_cache()
    usage_api.clear_antigravity_usage_cache()
    yield
    preflight.clear_preflight_cache()
    usage_api.clear_antigravity_usage_cache()


def test_preflight_enabled_respects_env_and_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(preflight.ANTIGRAVITY_MCP_PREFLIGHT_ENV, raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_case")
    assert preflight.preflight_enabled_for_live_session() is False

    monkeypatch.setenv(preflight.ANTIGRAVITY_MCP_PREFLIGHT_ENV, "1")
    assert preflight.preflight_enabled_for_live_session() is True

    monkeypatch.setenv(preflight.ANTIGRAVITY_MCP_PREFLIGHT_ENV, "0")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert preflight.preflight_enabled_for_live_session() is False


@pytest.mark.asyncio
async def test_probe_antigravity_mcp_servers_keeps_ok_and_prunes_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Avoid OS namespace setup in unit tests; test the JSON-RPC handshake directly.
    monkeypatch.setattr(preflight, "_prepare_sandbox_spec", lambda mode: ("off", (), ()))
    monkeypatch.setattr(preflight, "wrap_argv", lambda argv, **kw: (list(argv), None))
    monkeypatch.setattr(preflight.agent_scratch, "allocate_scratch", lambda prefix: None)

    ok_script = (
        "import json, sys\n"
        "req = json.loads(sys.stdin.readline())\n"
        "sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'capabilities': {}}}) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    rpc_err_script = (
        "import json, sys\n"
        "req = json.loads(sys.stdin.readline())\n"
        "sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], "
        "'error': {'code': -32603, 'message': 'No AWS credentials available.'}}) + '\\n')\n"
        "sys.stdout.flush()\n"
    )
    crash_script = (
        "import sys\n"
        "sys.stderr.write('ValueError: No AWS credentials available.\\n')\n"
        "sys.stderr.flush()\n"
        "sys.exit(1)\n"
    )
    slow_alive_script = (
        "import time\n"
        "time.sleep(5)\n"
    )

    servers = [
        {"name": "kirocrew-core", "command": "/nonexistent/ignored", "args": []},
        {"name": "brokered", "command": sys.executable, "args": ["-m", "kiro_crew.mcp_gateway.stub"]},
        {"name": "remote-http", "url": "https://example.com/mcp"},
        {"name": "healthy-mcp", "command": sys.executable, "args": ["-c", ok_script]},
        {"name": "aws-mcp", "command": sys.executable, "args": ["-c", rpc_err_script]},
        {"name": "crash-mcp", "command": sys.executable, "args": ["-c", crash_script]},
        {"name": "slow-mcp", "command": sys.executable, "args": ["-c", slow_alive_script]},
    ]

    kept, pruned = await preflight.probe_antigravity_mcp_servers(servers, timeout=0.6)
    kept_names = [s["name"] for s in kept]
    pruned_map = {p.name: p.reason for p in pruned}

    assert kept_names == ["kirocrew-core", "brokered", "remote-http", "healthy-mcp", "slow-mcp"]
    assert "aws-mcp" in pruned_map
    assert "No AWS credentials available" in pruned_map["aws-mcp"]
    assert "crash-mcp" in pruned_map
    assert "No AWS credentials available" in pruned_map["crash-mcp"]


def test_usage_api_signin_required_and_api_key_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(usage_api.hooks, "emit_internal_read_audit", lambda *a, **kw: True)

    res_missing = usage_api.fetch_antigravity_usage_sync()
    assert res_missing["available"] is False
    assert res_missing["auth_state"] == usage_api.AUTH_SIGNIN_REQUIRED

    store = tmp_path / ".gemini" / "antigravity-acp"
    store.mkdir(parents=True)
    (store / "settings.json").write_text(json.dumps({"auth": {"type": "api-key"}}), encoding="utf-8")

    res_apikey = usage_api.fetch_antigravity_usage_sync()
    assert res_apikey["available"] is False
    assert res_apikey["auth_state"] == usage_api.AUTH_API_KEY


def test_usage_api_parses_pools_models_and_enforces_security_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(usage_api.hooks, "emit_internal_read_audit", lambda *a, **kw: True)

    store = tmp_path / ".gemini" / "antigravity-acp"
    store.mkdir(parents=True)
    (store / "settings.json").write_text(
        json.dumps({"auth": {"type": "oauth-personal"}}),
        encoding="utf-8",
    )
    (store / "acp_token.json").write_text(
        json.dumps(
            {
                "client_id": "cid.apps.googleusercontent.com",
                "client_secret": "csecret",
                "refresh_token": "1//refresh-token-value",
                "project_id": "demo-proj",
            }
        ),
        encoding="utf-8",
    )

    def _fake_http(
        url: str,
        *,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> usage_api._Resp:
        if url == usage_api._OAUTH_TOKEN_URL:
            return usage_api._Resp(200, json.dumps({"access_token": "ya29.mock", "expires_in": 3600}))
        if url == usage_api._USERINFO_URL:
            return usage_api._Resp(200, json.dumps({"email": "dev@example.com"}))
        if url.endswith(":loadCodeAssist"):
            return usage_api._Resp(
                200,
                json.dumps(
                    {
                        "cloudaicompanionProject": "demo-proj",
                        "paidTier": {
                            "id": "g1-ultra-tier",
                            "name": "Google AI Ultra",
                            "description": "Google AI Ultra",
                            "upgradeSubscriptionUri": "https://evil.example.com/phish",
                        },
                    }
                ),
            )
        if url.endswith(":retrieveUserQuotaSummary"):
            return usage_api._Resp(
                200,
                json.dumps(
                    {
                        "groups": [
                            {
                                "displayName": "Claude and GPT models",
                                "buckets": [
                                    {
                                        "displayName": "Five Hour Limit Remaining",
                                        "remainingFraction": 0.88,
                                        "resetTime": "2026-11-11T10:02:50Z",
                                    },
                                    {
                                        "displayName": "Weekly Limit Remaining",
                                        "remainingFraction": 0.94,
                                        "resetTime": "2026-11-18T05:02:50Z",
                                    },
                                ],
                            }
                        ]
                    }
                ),
            )
        if url.endswith(":fetchAvailableModels"):
            return usage_api._Resp(
                200,
                json.dumps(
                    {
                        "models": {
                            "gemini-3.1-pro-high": {
                                "displayName": "Gemini 3.1 Pro (High)",
                                "quotaInfo": {
                                    "remainingFraction": 0.70,
                                    "resetTime": "2026-11-11T09:44:23Z",
                                },
                            },
                            "claude-sonnet-4-6": {
                                "displayName": "Claude Sonnet 4.6",
                                "quotaInfo": {
                                    "remainingFraction": 0.88,
                                    "resetTime": "2026-11-11T10:02:50Z",
                                },
                            },
                        }
                    }
                ),
            )
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(usage_api, "_http_request", _fake_http)

    payload = usage_api.fetch_antigravity_usage_sync()
    assert payload["available"] is True
    assert payload["email"] == "dev@example.com"
    assert payload["tier_name"] == "Google AI Ultra"
    # Untrusted upgrade host must be stripped!
    assert payload["upgrade_uri"] is None
    assert [p["window"] for p in payload["pools"]] == ["5h", "5h", "weekly"]
    assert payload["pools"][0]["group"] == "Gemini Models"
    assert payload["pools"][0]["remaining_pct"] == 70
    assert len(payload["models"]) == 2

    # Allowed upgrade host passes through cleanly.
    assert usage_api._safe_upgrade_uri("https://antigravity.google/upgrade") == "https://antigravity.google/upgrade"
    assert usage_api._safe_upgrade_uri("http://antigravity.google/upgrade") is None

    # Redirect handler refuses every redirect.
    no_redir = usage_api._NoRedirect()
    assert no_redir.redirect_request(None, None, 302, "Found", {}, "https://example.com") is None


@pytest.mark.asyncio
async def test_usage_api_handlers_stale_fallback_and_409_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = {
        "available": True,
        "auth_state": usage_api.AUTH_OK,
        "auth_type": "oauth-personal",
        "email": "dev@example.com",
        "tier_id": "g1-ultra-tier",
        "tier_name": "Google AI Ultra",
        "tier_description": None,
        "upgrade_uri": None,
        "pools": [{"id": "g", "group": "Gemini Models", "label": "5h", "window": "5h", "remaining_fraction": 0.75, "remaining_pct": 75, "reset_time": None}],
        "models": [],
        "fetched_at": 1700000000.0,
        "stale": False,
    }
    calls = [good, {"available": False, "auth_state": usage_api.AUTH_OTHER, "auth_type": "oauth-personal"}]
    monkeypatch.setattr(usage_api, "fetch_antigravity_usage_sync", lambda **kw: calls.pop(0))

    resp1 = await usage_api.api_antigravity_usage(MagicMock())
    body1 = json.loads(resp1.body.decode("utf-8"))
    assert body1["usage"]["available"] is True
    assert body1["usage"]["stale"] is False

    # Transient failure on forced refresh preserves cached reading with stale=True.
    resp2 = await usage_api.api_antigravity_usage_refresh(MagicMock())
    body2 = json.loads(resp2.body.decode("utf-8"))
    assert body2["usage"]["available"] is True
    assert body2["usage"]["stale"] is True

    # Concurrent refresh while _REFRESH_IN_FLIGHT returns 409.
    monkeypatch.setattr(usage_api, "_REFRESH_IN_FLIGHT", True)
    resp409 = await usage_api.api_antigravity_usage_refresh(MagicMock())
    assert resp409.status == 409
