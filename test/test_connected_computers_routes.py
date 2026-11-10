"""Connected computers: registering the routes loads none of the feature's code.

Every gateway registers the connected-computers routes, with the preview off as
well, so registration must not put the connector socket, the broker or the
credential code on the boot path. The handlers import them on first request.
The import check runs in a fresh interpreter: in this warm test process those
modules are already loaded by the other tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from aiohttp import web

from kiro_crew.connected_computers import remote_routes, ws_route
from kiro_crew.dashboard.token_auth_paths import CONNECTOR_CONNECT_PATH
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS

#: What a request loads, and registration must not.
_LOADED_ON_FIRST_REQUEST = (
    "kiro_crew.connected_computers.ws_route",
    "kiro_crew.connected_computers.broker",
    "kiro_crew.connected_computers.connector_token",
    "kiro_crew.connected_computers.devices",
    "kiro_crew.connected_computers.identity",
    "kiro_crew.connected_computers.vault_doc",
    "kiro_crew.connected_computers.authenticated",
    "kiro_crew.connected_computers.protocol",
    "kiro_crew.remote_session.link",
)

_PROBE = textwrap.dedent("""
    import sys
    from aiohttp import web
    # The dashboard gate is loaded before any route on a real boot; whatever it
    # loads is not this feature's doing.
    import kiro_crew.dashboard.token_auth_paths  # noqa: F401
    before = set(sys.modules)
    from kiro_crew.connected_computers.remote_routes import register_all

    app = web.Application()
    register_all(app)
    paths = sorted(r.canonical for r in app.router.resources())
    assert paths == sorted(sys.argv[1].split(",")), paths
    assert "kiro_crew.connected_computers.ws_route" not in sys.modules
    loaded = [m for m in sys.argv[2].split(",") if m in sys.modules and m not in before]
    assert not loaded, f"registration imported {loaded}"
    """)


def _expected_paths() -> list[str]:
    devices = remote_routes.DEVICES_PATH
    return [
        CONNECTOR_CONNECT_PATH,
        devices,
        devices + "/{device_id}/revoke",
        devices + "/{device_id}/remove",
    ]


def test_registering_the_routes_imports_no_connector_code() -> None:
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            ",".join(_expected_paths()),
            ",".join(_LOADED_ON_FIRST_REQUEST),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=FALLBACK_TIMEOUT_SECS,
    )
    assert res.returncode == 0, res.stderr


def test_the_socket_path_is_the_one_the_dashboard_gate_lets_through() -> None:
    assert ws_route.CONNECT_PATH == CONNECTOR_CONNECT_PATH


@pytest.mark.asyncio
async def test_the_connect_route_hands_the_request_to_the_socket_handler(monkeypatch) -> None:
    seen: list[object] = []
    answer = web.Response(status=503)

    async def fake_connect(request: object) -> web.Response:
        seen.append(request)
        return answer

    monkeypatch.setattr(ws_route, "api_connector_connect", fake_connect)
    request = object()
    assert await remote_routes.api_connector_connect(request) is answer  # type: ignore[arg-type]
    assert seen == [request]
