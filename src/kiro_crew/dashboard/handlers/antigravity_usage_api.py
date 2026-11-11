"""Google Antigravity remaining-quota client and dashboard HTTP handlers.

Reads the signed-in Google Antigravity OAuth credential maintained by
``agy_acp_server`` under ``~/.gemini/antigravity-acp/acp_token.json`` through
Kiro Crew's SEL-audited ``hooks.safe_read_file_internal`` carve-out and queries
Google's Cloud Code quota endpoints (``cloudcode-pa.googleapis.com``) over TLS
using stdlib ``urllib`` only.

Security controls (matching :mod:`kiro_crew.dashboard.handlers.kiro_usage_api`):
  1. Endpoints are hardcoded Google production hosts -- never derived from
     config or from ``token_uri`` on disk.
  2. TLS certificate chain + hostname verification are always enabled.
  3. HTTP redirects are disabled (:class:`_NoRedirect`) so bearer and refresh
     tokens can never be replayed to a redirected origin.
  4. Tokens are never logged; only HTTP status codes are recorded.
  5. Every string in the returned payload passes through
     ``redact_credentials`` and ``redact_exfiltration_urls`` before caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import ssl
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from aiohttp import web

from kiro_crew import hooks, platform_compat
from kiro_crew.loop_lock import LoopBoundLock
from kiro_crew.security import redact_credentials, redact_exfiltration_urls

logger = logging.getLogger(__name__)

# Hardcoded Google production endpoints (control 1).
_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
_CCPA_BASE_URL = "https://cloudcode-pa.googleapis.com/v1internal"
_ALLOWED_UPGRADE_HOSTS = frozenset({"antigravity.google", "one.google.com"})

_TOKEN_READ_ID = "antigravity_usage_api.acp_token"
_CUSTOM_HOME_AUDIT_ID = "antigravity_usage_api.custom_home_token"

_TIMEOUT_SECS = 12
_MAX_RESP_BYTES = 1_000_000
_CACHE_TTL_SECS = 60.0
_TOKEN_EXPIRY_SKEW_SECS = 60.0
_EMAIL_CACHE_MAX = 16

AUTH_OK = "ok"
AUTH_API_KEY = "api_key"
AUTH_SIGNIN_REQUIRED = "signin_required"
AUTH_OTHER = "other"

_TOKEN_LOCK = threading.Lock()
_CACHED_ACCESS_TOKEN: tuple[str, str, float] | None = None  # (refresh_digest, token, exp_mono)
_EMAIL_CACHE: dict[str, str] = {}

_USAGE_CACHE: dict[str, Any] | None = None
_USAGE_CACHE_TS: float = 0.0
_REFRESH_LOCK = LoopBoundLock()
_REFRESH_IN_FLIGHT = False

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class _RequestError(Exception):
    """Transport-level HTTP failure (DNS, connect, TLS, timeout)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse all 3xx redirects so credentials never follow a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


class _Resp:
    """Minimal response wrapper exposing ``status_code`` and ``json()``."""

    __slots__ = ("status_code", "_text")

    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self._text = text

    def json(self) -> dict[str, Any]:
        data = json.loads(self._text) if self._text else {}
        return data if isinstance(data, dict) else {}


def _build_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(
        _NoRedirect(),
        urllib.request.HTTPSHandler(context=ctx),
    )


def _read_capped(fp: object) -> str:
    data = fp.read(_MAX_RESP_BYTES + 1)  # type: ignore[attr-defined]
    if len(data) > _MAX_RESP_BYTES:
        return ""
    return data.decode("utf-8", "replace")


def _http_request(
    url: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> _Resp:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    opener = _build_opener()
    try:
        with opener.open(req, timeout=_TIMEOUT_SECS) as resp:
            text = _read_capped(resp)
            return _Resp(getattr(resp, "status", 200) or 200, text)
    except urllib.error.HTTPError as exc:
        try:
            text = _read_capped(exc)
        except Exception:  # noqa: BLE001
            text = ""
        return _Resp(exc.code, text)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise _RequestError(str(exc)) from exc


def _redact_obj(value: Any) -> Any:
    """Recursively redact credentials and exfiltration URLs from strings."""
    if isinstance(value, str):
        redacted, _ = redact_exfiltration_urls(value)
        redacted, _ = redact_credentials(redacted)
        return redacted
    if isinstance(value, list):
        return [_redact_obj(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _redact_obj(v) for k, v in value.items()}
    return value


def _safe_upgrade_uri(raw: object) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = urllib.parse.urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_UPGRADE_HOSTS:
        return None
    return raw[:200]


def _read_custom_home_file(path: Path, *, audit_required: bool) -> bytes | None:
    """Read a regular non-symlink file under an explicit ``GEMINI_CLI_HOME``."""
    try:
        fd = platform_compat.open_file_no_reparse(str(path), nonblocking=True)
    except FileNotFoundError:
        if audit_required:
            hooks.emit_internal_read_audit(_CUSTOM_HOME_AUDIT_ID, "missing")
        return None
    except OSError:
        if audit_required:
            hooks.emit_internal_read_audit(_CUSTOM_HOME_AUDIT_ID, "unreadable")
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            if audit_required:
                hooks.emit_internal_read_audit(_CUSTOM_HOME_AUDIT_ID, "not_regular")
            return None
        with os.fdopen(fd, "rb", closefd=True) as fh:
            fd = -1
            data = fh.read(_MAX_RESP_BYTES + 1)
    except OSError:
        if audit_required:
            hooks.emit_internal_read_audit(_CUSTOM_HOME_AUDIT_ID, "unreadable")
        return None
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
    if len(data) > _MAX_RESP_BYTES:
        if audit_required:
            hooks.emit_internal_read_audit(_CUSTOM_HOME_AUDIT_ID, "too_large")
        return None
    if audit_required and not hooks.emit_internal_read_audit(_CUSTOM_HOME_AUDIT_ID, "success"):
        return None
    return data


def _read_antigravity_store_json(leaf: str) -> dict[str, Any] | None:
    """Read ``settings.json`` (plain) or ``acp_token.json`` (via SEL-audited gate)."""
    custom_home = os.environ.get("GEMINI_CLI_HOME", "").strip()
    home = Path.home()
    raw: bytes | None
    if leaf != "acp_token.json":
        base = Path(custom_home).expanduser() if custom_home else home
        target = base / ".gemini" / "antigravity-acp" / leaf
        raw = _read_custom_home_file(target, audit_required=False)
    elif custom_home and Path(custom_home).expanduser().resolve() != home.resolve():
        target = Path(custom_home).expanduser() / ".gemini" / "antigravity-acp" / leaf
        raw = _read_custom_home_file(target, audit_required=True)
    else:
        raw = hooks.safe_read_file_internal(_TOKEN_READ_ID)
    if not raw:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _exchange_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    *,
    force_refresh: bool = False,
) -> tuple[str | None, str]:
    """Return ``(access_token, auth_state)`` using an in-memory TTL cache."""
    global _CACHED_ACCESS_TOKEN
    digest = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:24]
    now_mono = time.monotonic()
    if not force_refresh:
        with _TOKEN_LOCK:
            if (
                _CACHED_ACCESS_TOKEN is not None
                and _CACHED_ACCESS_TOKEN[0] == digest
                and now_mono < _CACHED_ACCESS_TOKEN[2]
            ):
                return (_CACHED_ACCESS_TOKEN[1], AUTH_OK)

    form = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode("utf-8")
    try:
        resp = _http_request(
            _OAUTH_TOKEN_URL,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=form,
        )
    except _RequestError:
        return (None, AUTH_OTHER)

    if resp.status_code in (400, 401):
        with _TOKEN_LOCK:
            if _CACHED_ACCESS_TOKEN is not None and _CACHED_ACCESS_TOKEN[0] == digest:
                _CACHED_ACCESS_TOKEN = None
        return (None, AUTH_SIGNIN_REQUIRED)
    if resp.status_code != 200:
        return (None, AUTH_OTHER)

    try:
        body = resp.json()
    except ValueError:
        return (None, AUTH_OTHER)

    access_token = body.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        return (None, AUTH_OTHER)
    expires_raw = body.get("expires_in", 3600)
    try:
        expires_in = max(60.0, min(float(expires_raw), 7200.0))
    except (TypeError, ValueError):
        expires_in = 3600.0

    with _TOKEN_LOCK:
        _CACHED_ACCESS_TOKEN = (
            digest,
            access_token,
            time.monotonic() + expires_in - _TOKEN_EXPIRY_SKEW_SECS,
        )
    return (access_token, AUTH_OK)


def _fetch_account_email(access_token: str, refresh_digest: str) -> str | None:
    with _TOKEN_LOCK:
        cached = _EMAIL_CACHE.get(refresh_digest)
    if cached:
        return cached
    try:
        resp = _http_request(
            _USERINFO_URL,
            method="GET",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    except _RequestError:
        return None
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    email = body.get("email")
    if not isinstance(email, str) or "@" not in email:
        return None
    bounded = email.strip()[:120]
    with _TOKEN_LOCK:
        if refresh_digest not in _EMAIL_CACHE and len(_EMAIL_CACHE) >= _EMAIL_CACHE_MAX:
            oldest = next(iter(_EMAIL_CACHE), None)
            if oldest is not None:
                _EMAIL_CACHE.pop(oldest, None)
        _EMAIL_CACHE[refresh_digest] = bounded
    return bounded


def _ccpa_post(access_token: str, rpc_method: str, payload: dict[str, Any]) -> _Resp:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _http_request(
        f"{_CCPA_BASE_URL}:{rpc_method}",
        method="POST",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "User-Agent": "antigravity-acp-probe/1.0 KiroCrew",
        },
        body=body,
    )


def _bounded_fraction(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return max(0.0, min(1.0, f))


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:64] or "pool"


def _classify_window(label: str) -> str:
    lower = label.lower()
    if "week" in lower:
        return "weekly"
    if "day" in lower or "daily" in lower or "24" in lower:
        return "daily"
    if "five" in lower or "5h" in lower or "5-hour" in lower or "hour" in lower:
        return "5h"
    return "rolling"


def _parse_summary_pools(summary_body: dict[str, Any]) -> list[dict[str, Any]]:
    pools: list[dict[str, Any]] = []
    groups = summary_body.get("groups")
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_name = str(group.get("displayName") or group.get("description") or "Models")[:80]
            buckets = group.get("buckets")
            if not isinstance(buckets, list):
                continue
            for bucket in buckets:
                if not isinstance(bucket, dict) or bucket.get("disabled") is True:
                    continue
                frac = _bounded_fraction(bucket.get("remainingFraction"))
                if frac is None:
                    continue
                label = str(
                    bucket.get("displayName") or bucket.get("description") or "Quota Remaining"
                )[:80]
                reset_raw = bucket.get("resetTime")
                reset_time = (
                    reset_raw[:40] if isinstance(reset_raw, str) and reset_raw else None
                )
                pools.append(
                    {
                        "id": f"{_slug(group_name)}:{_slug(label)}",
                        "group": group_name,
                        "label": label,
                        "window": _classify_window(label),
                        "remaining_fraction": round(frac, 4),
                        "remaining_pct": int(round(frac * 100)),
                        "reset_time": reset_time,
                    }
                )
    return pools


def _parse_models_and_gemini_pool(
    models_body: dict[str, Any],
    existing_pools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    raw_models = models_body.get("models")
    if not isinstance(raw_models, dict):
        return [], None

    parsed_models: list[dict[str, Any]] = []
    for model_id, info in raw_models.items():
        if not isinstance(model_id, str) or not isinstance(info, dict):
            continue
        quota = info.get("quotaInfo")
        if not isinstance(quota, dict):
            continue
        frac = _bounded_fraction(quota.get("remainingFraction"))
        if frac is None:
            continue
        display = str(info.get("displayName") or model_id)[:80]
        reset_raw = quota.get("resetTime")
        reset_time = reset_raw[:40] if isinstance(reset_raw, str) and reset_raw else None
        parsed_models.append(
            {
                "id": model_id[:80],
                "display_name": display,
                "remaining_fraction": round(frac, 4),
                "remaining_pct": int(round(frac * 100)),
                "reset_time": reset_time,
            }
        )

    parsed_models.sort(key=lambda m: (not m["id"].startswith("gemini-"), m["display_name"]))

    has_gemini_group = any("gemini" in str(p.get("group", "")).lower() for p in existing_pools)
    gemini_pool: dict[str, Any] | None = None
    if not has_gemini_group:
        gemini_entries = [m for m in parsed_models if m["id"].startswith("gemini-")]
        if gemini_entries:
            # Prefer flagship agent models (3.1 Pro / 3 Flash / 2.5 Pro) if present.
            preferred = [
                m
                for m in gemini_entries
                if any(
                    k in m["id"]
                    for k in ("gemini-3.1-pro", "gemini-3-flash", "gemini-2.5-pro")
                )
            ]
            chosen = min(preferred or gemini_entries, key=lambda m: m["remaining_fraction"])
            gemini_pool = {
                "id": "gemini-models:quota-remaining",
                "group": "Gemini Models",
                "label": "5h Limit Remaining",
                "window": "5h",
                "remaining_fraction": chosen["remaining_fraction"],
                "remaining_pct": chosen["remaining_pct"],
                "reset_time": chosen["reset_time"],
            }

    return parsed_models[:32], gemini_pool


def fetch_antigravity_usage_sync(*, force_token_refresh: bool = False) -> dict[str, Any]:
    """Fetch Google Antigravity tier + remaining quota synchronously (run off-loop)."""
    settings = _read_antigravity_store_json("settings.json") or {}
    auth_cfg = settings.get("auth") if isinstance(settings.get("auth"), dict) else {}
    auth_type = str(auth_cfg.get("type") or "")[:40]

    token_doc = _read_antigravity_store_json("acp_token.json")
    if not token_doc:
        if auth_type == "api-key" or os.environ.get("GEMINI_API_KEY"):
            return {
                "available": False,
                "auth_state": AUTH_API_KEY,
                "auth_type": auth_type or "api-key",
            }
        return {
            "available": False,
            "auth_state": AUTH_SIGNIN_REQUIRED,
            "auth_type": auth_type or "oauth-personal",
        }

    client_id = token_doc.get("client_id")
    client_secret = token_doc.get("client_secret")
    refresh_token = token_doc.get("refresh_token")
    stored_project = token_doc.get("project_id")
    if not (
        isinstance(client_id, str)
        and client_id
        and isinstance(client_secret, str)
        and client_secret
        and isinstance(refresh_token, str)
        and refresh_token
    ):
        return {
            "available": False,
            "auth_state": AUTH_SIGNIN_REQUIRED,
            "auth_type": auth_type or "oauth-personal",
        }

    access_token, state = _exchange_access_token(
        client_id,
        client_secret,
        refresh_token,
        force_refresh=force_token_refresh,
    )
    if not access_token:
        return {
            "available": False,
            "auth_state": state,
            "auth_type": auth_type or "oauth-personal",
        }

    refresh_digest = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:24]

    load_payload: dict[str, Any] = {
        "metadata": {
            "ideType": "ANTIGRAVITY",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }
    }
    if isinstance(stored_project, str) and stored_project:
        load_payload["cloudaicompanionProject"] = stored_project

    try:
        load_resp = _ccpa_post(access_token, "loadCodeAssist", load_payload)
        if load_resp.status_code == 401 and not force_token_refresh:
            access_token, state = _exchange_access_token(
                client_id, client_secret, refresh_token, force_refresh=True
            )
            if not access_token:
                return {
                    "available": False,
                    "auth_state": state,
                    "auth_type": auth_type or "oauth-personal",
                }
            load_resp = _ccpa_post(access_token, "loadCodeAssist", load_payload)
    except _RequestError:
        return {
            "available": False,
            "auth_state": AUTH_OTHER,
            "auth_type": auth_type or "oauth-personal",
        }

    if load_resp.status_code == 401:
        return {
            "available": False,
            "auth_state": AUTH_SIGNIN_REQUIRED,
            "auth_type": auth_type or "oauth-personal",
        }

    load_body = load_resp.json() if load_resp.status_code == 200 else {}
    project_id = (
        str(load_body.get("cloudaicompanionProject") or stored_project or "").strip()
    )
    tier_obj = load_body.get("paidTier") or load_body.get("currentTier")
    tier_dict = tier_obj if isinstance(tier_obj, dict) else {}
    tier_id = str(tier_dict.get("id") or "")[:64] or None
    tier_name = str(tier_dict.get("name") or "")[:80] or None
    tier_desc = str(tier_dict.get("description") or "")[:160] or None
    upgrade_uri = _safe_upgrade_uri(tier_dict.get("upgradeSubscriptionUri"))

    rpc_req: dict[str, Any] = {"project": project_id} if project_id else {}
    summary_body: dict[str, Any] = {}
    models_body: dict[str, Any] = {}
    try:
        summary_resp = _ccpa_post(access_token, "retrieveUserQuotaSummary", rpc_req)
        if summary_resp.status_code == 200:
            summary_body = summary_resp.json()
    except (_RequestError, ValueError):
        summary_body = {}

    try:
        models_resp = _ccpa_post(access_token, "fetchAvailableModels", rpc_req)
        if models_resp.status_code == 200:
            models_body = models_resp.json()
    except (_RequestError, ValueError):
        models_body = {}

    summary_pools = _parse_summary_pools(summary_body)
    models_list, gemini_pool = _parse_models_and_gemini_pool(models_body, summary_pools)
    pools = ([gemini_pool] if gemini_pool is not None else []) + summary_pools

    if not pools and not models_list and not tier_name:
        return {
            "available": False,
            "auth_state": AUTH_OTHER,
            "auth_type": auth_type or "oauth-personal",
        }

    email = _fetch_account_email(access_token, refresh_digest)
    result: dict[str, Any] = {
        "available": True,
        "auth_state": AUTH_OK,
        "auth_type": auth_type or "oauth-personal",
        "email": email,
        "tier_id": tier_id,
        "tier_name": tier_name,
        "tier_description": tier_desc,
        "upgrade_uri": upgrade_uri,
        "pools": pools,
        "models": models_list,
        "fetched_at": time.time(),
        "stale": False,
    }
    redacted = _redact_obj(result)
    return redacted if isinstance(redacted, dict) else result


def clear_antigravity_usage_cache() -> None:
    """Clear all in-memory token, email and quota caches (for tests)."""
    global _CACHED_ACCESS_TOKEN, _USAGE_CACHE, _USAGE_CACHE_TS
    with _TOKEN_LOCK:
        _CACHED_ACCESS_TOKEN = None
        _EMAIL_CACHE.clear()
    _USAGE_CACHE = None
    _USAGE_CACHE_TS = 0.0


async def _refresh_usage(*, force: bool = False) -> dict[str, Any]:
    global _USAGE_CACHE, _USAGE_CACHE_TS, _REFRESH_IN_FLIGHT
    now = time.time()
    if not force and _USAGE_CACHE is not None and (now - _USAGE_CACHE_TS) < _CACHE_TTL_SECS:
        return dict(_USAGE_CACHE)

    async with _REFRESH_LOCK:
        now = time.time()
        if not force and _USAGE_CACHE is not None and (now - _USAGE_CACHE_TS) < _CACHE_TTL_SECS:
            return dict(_USAGE_CACHE)
        _REFRESH_IN_FLIGHT = True
        try:
            fresh = await asyncio.to_thread(fetch_antigravity_usage_sync)
        finally:
            _REFRESH_IN_FLIGHT = False

        if fresh.get("available") is True:
            _USAGE_CACHE = fresh
            _USAGE_CACHE_TS = time.time()
            return dict(fresh)

        # On transient transport failure, serve any prior valid reading marked stale.
        if (
            fresh.get("auth_state") == AUTH_OTHER
            and _USAGE_CACHE is not None
            and _USAGE_CACHE.get("available") is True
        ):
            stale_copy = {**_USAGE_CACHE, "stale": True}
            return stale_copy

        _USAGE_CACHE = fresh
        _USAGE_CACHE_TS = time.time()
        return dict(fresh)


async def api_antigravity_usage(request: web.Request) -> web.Response:
    """GET /api/usage/antigravity -- cached Google Antigravity remaining quota."""
    usage = await _refresh_usage(force=False)
    return web.json_response({"usage": usage})


async def api_antigravity_usage_refresh(request: web.Request) -> web.Response:
    """POST /api/usage/antigravity/refresh -- refresh Antigravity quota immediately."""
    if _REFRESH_IN_FLIGHT:
        return web.json_response({"error": "refresh_in_flight"}, status=409)
    usage = await _refresh_usage(force=True)
    return web.json_response({"usage": usage})
