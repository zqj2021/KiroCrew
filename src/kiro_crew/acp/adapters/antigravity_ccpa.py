"""Google Antigravity OAuth (CCPA) credentials, model catalog, and loopback proxy.

Enables ``kirocrew-antigravity-acp`` (the pure-Python ``google-antigravity`` SDK
adapter) to authenticate directly with the user's signed-in Google Antigravity
OAuth credential (``~/.gemini/antigravity-acp/acp_token.json``) in addition to
``GEMINI_API_KEY`` and Vertex AI:

1. **OAuth token exchange & onboarding**: refreshes the stored Google OAuth token
   over TLS (no redirects) and resolves the user's Cloud Code project + tier
   endpoint (``daily-cloudcode-pa.googleapis.com`` for consumer Free/Pro/Ultra
   tiers; ``cloudcode-pa.googleapis.com`` for GCP ToS enterprise tiers).
2. **Grouped CCPA model catalog & deterministic thinking control**: fetches the
   account's live ``fetchAvailableModels`` list, groups ``-<level>`` variants into
   family IDs (``gemini-3.8-flash``, ``gemini-4-argon``, …) matching standalone
   Antigravity ACP, and defaults every session and unpinned family switch to
   ``low`` thinking level rather than auto-escalating into ``high``.
3. **Loopback CCPA reverse proxy with live thought injection**: bridges
   ``localharness``'s Gemini REST calls onto CCPA ``v1internal``, injecting
   ``generationConfig.thinkingConfig.includeThoughts = True`` so CCPA streams
   real ``{"thought": true, "text": "..."}`` chain-of-thought deltas across all
   reasoning steps while tracking upstream in-flight liveness per session.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import logging
import os
import platform
import re
import secrets
import socketserver
import ssl
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
_PROD_CCPA_ENDPOINT = "https://cloudcode-pa.googleapis.com"
_DAILY_CCPA_ENDPOINT = "https://daily-cloudcode-pa.googleapis.com"
_CCPA_BASE_URL_ENV = "AGY_ACP_CCPA_BASE_URL"
_DEFAULT_ACP_MODEL_ENV = "AGY_ACP_DEFAULT_MODEL"

#: Default CCPA concrete model and family (explicitly ``low``, never auto-high).
DEFAULT_CCPA_FAMILY_ID = "gemini-3.8-flash"
DEFAULT_CCPA_THINKING_LEVEL = "low"
DEFAULT_CCPA_CONCRETE_MODEL_ID = "gemini-3.8-flash-low"

_TOKEN_EXPIRY_SKEW_SECS = 60.0
_CATALOG_TTL_SECS = 300.0
_HTTP_TIMEOUT_SECS = 20.0
_STREAM_TIMEOUT_SECS = 300.0
_MAX_JSON_BYTES = 2_000_000

_GOARCH = {"x86_64": "amd64", "aarch64": "arm64"}
_MODEL_PATH_RE = re.compile(r"/v1(?:beta)?/models/([^/:]+)")

#: Legacy CCPA model IDs whose suffix does not spell the level directly.
_CCPA_LEGACY_VARIANTS: dict[str, tuple[str, str]] = {
    "gemini-3.5-flash-extra-low": ("gemini-3.5-flash", "low"),
    "gemini-3.5-flash-low": ("gemini-3.5-flash", "medium"),
    "gemini-3-flash-agent": ("gemini-3.5-flash", "high"),
    "gemini-3.1-pro-low": ("gemini-3.1-pro", "low"),
    "gemini-pro-agent": ("gemini-3.1-pro", "high"),
}

#: Ordered from lowest thinking cost to highest so unpinned selection picks ``low``.
_LEVEL_ORDER: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")
_LEVEL_RANK: dict[str, int] = {lvl: idx for idx, lvl in enumerate(_LEVEL_ORDER)}
_CCPA_LEVEL_LABELS: dict[str, str] = {
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "Extra",
    "max": "Max",
}

#: Fallback CCPA catalog used only if ``fetchAvailableModels`` is temporarily unreachable.
_FALLBACK_CCPA_MODELS: tuple[tuple[str, str, int], ...] = (
    ("gemini-3.8-flash-low", "Gemini 3.8 Flash (Low)", 1_048_576),
    ("gemini-3.8-flash-medium", "Gemini 3.8 Flash (Medium)", 1_048_576),
    ("gemini-3.8-flash-high", "Gemini 3.8 Flash (High)", 1_048_576),
    ("gemini-4-argon-low", "Gemini 4 Argon (Low)", 1_048_576),
    ("gemini-4-argon-medium", "Gemini 4 Argon (Medium)", 1_048_576),
    ("gemini-4-argon-high", "Gemini 4 Argon (High)", 1_048_576),
    ("gemini-3-flash", "Gemini 3 Flash", 1_048_576),
    ("gemini-2.5-pro", "Gemini 2.5 Pro", 1_048_576),
    ("gemini-2.5-flash", "Gemini 2.5 Flash", 1_048_576),
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Disable HTTP redirects so OAuth bearer/refresh tokens never leave Google hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _build_https_opener() -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()
    return urllib.request.build_opener(_NoRedirect(), urllib.request.HTTPSHandler(context=ctx))


def build_aidev_user_agent(
    client_name: str = "kiro-crew",
    client_version: str = "0.13.0",
    *,
    acp_version: str = "1.4.4",
) -> str:
    """Build the structured ``antigravity/acp`` User-Agent required by CCPA."""
    os_type = platform.system().lower() or "linux"
    raw_arch = platform.machine().lower() or "amd64"
    arch = _GOARCH.get(raw_arch, raw_arch)
    safe_name = re.sub(r"[^a-z0-9-]", "-", (client_name or "unknown").lower())
    safe_ver = re.sub(r"[^a-z0-9.-]", "-", (client_version or "unknown").lower())
    dims = (
        f"aidev_client; os_type={os_type}; arch={arch}; "
        f"host_path={safe_name}/{safe_ver}; proxy_client=antigravity/sdk"
    )
    return f"antigravity/acp/{acp_version} ({dims})"


def resolve_acp_token_path(env: Mapping[str, str] | None = None) -> Path | None:
    """Locate ``acp_token.json``, respecting isolated test ``env`` mappings."""
    if env is not None:
        cli_home = env.get("GEMINI_CLI_HOME", "").strip()
        if cli_home:
            return Path(cli_home).expanduser() / ".gemini" / "antigravity-acp" / "acp_token.json"
        gem_home = env.get("GEMINI_HOME", "").strip()
        if gem_home:
            return Path(gem_home).expanduser() / "antigravity-acp" / "acp_token.json"
        home = env.get("HOME", "").strip() or env.get("USERPROFILE", "").strip()
        if not home:
            # Explicit isolated mapping (e.g. unit test passing env={}): never touch host ~.
            return None
        return Path(home) / ".gemini" / "antigravity-acp" / "acp_token.json"

    cli_home = os.environ.get("GEMINI_CLI_HOME", "").strip()
    if cli_home:
        return Path(cli_home).expanduser() / ".gemini" / "antigravity-acp" / "acp_token.json"
    gem_home = os.environ.get("GEMINI_HOME", "").strip()
    if gem_home:
        return Path(gem_home).expanduser() / "antigravity-acp" / "acp_token.json"
    return Path.home() / ".gemini" / "antigravity-acp" / "acp_token.json"


def has_valid_oauth_token_file(env: Mapping[str, str] | None = None) -> bool:
    """True when ``acp_token.json`` exists and holds OAuth refresh credentials."""
    data = _read_token_json(env)
    if not data:
        return False
    return bool(
        isinstance(data.get("refresh_token"), str)
        and data["refresh_token"].strip()
        and isinstance(data.get("client_id"), str)
        and data["client_id"].strip()
        and isinstance(data.get("client_secret"), str)
        and data["client_secret"].strip()
    )


def _read_token_json(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    path = resolve_acp_token_path(env)
    if path is None:
        return None
    try:
        st = path.lstat()
        if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_JSON_BYTES:
            return None
        parsed = json.loads(path.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def parse_ccpa_variant(model_id: str) -> tuple[str, str] | None:
    """Return ``(family_id, level)`` for a leveled CCPA model id, else ``None``."""
    if model_id in _CCPA_LEGACY_VARIANTS:
        return _CCPA_LEGACY_VARIANTS[model_id]
    trimmed = model_id.removesuffix("-effort")
    family_id, sep, level = trimmed.rpartition("-")
    if not sep or not family_id or level not in _CCPA_LEVEL_LABELS:
        return None
    return family_id, level


def _strip_level_suffix(display_name: str, level: str) -> str:
    label = _CCPA_LEVEL_LABELS.get(level, level)
    for candidate in (
        f" ({level})",
        f" ({label})",
        f" ({level} thinking)",
        f" ({label} thinking)",
    ):
        if len(display_name) > len(candidate) and display_name.lower().endswith(candidate.lower()):
            return display_name[: -len(candidate)].strip()
    return display_name


@dataclass(frozen=True)
class CcpaModelInfo:
    """One concrete model served by CCPA."""

    model_id: str
    name: str
    description: str
    max_tokens: int = 1_048_576


@dataclass(frozen=True)
class CcpaFamilyEntry:
    """Grouped family entry advertised in ``configOptions[id='model']``."""

    family_id: str
    name: str
    #: Concrete ``(level, model_id)`` pairs ordered from ``low`` -> ``high``.
    variants: tuple[tuple[str, str], ...]

    def model_id_for_level(self, level: str) -> str | None:
        for lvl, mid in self.variants:
            if lvl == level:
                return mid
        return None

    def nearest_model_id(self, level: str) -> str:
        exact = self.model_id_for_level(level)
        if exact is not None:
            return exact
        target_rank = _LEVEL_RANK.get(level, _LEVEL_RANK[DEFAULT_CCPA_THINKING_LEVEL])
        ranked = [(_LEVEL_RANK.get(lvl, 99), mid) for lvl, mid in self.variants]
        ranked.sort(key=lambda item: (abs(item[0] - target_rank), item[0]))
        return ranked[0][1]

    def low_default_model_id(self) -> str:
        """Always pick ``low`` (or the lowest available effort), never auto-high."""
        return self.nearest_model_id(DEFAULT_CCPA_THINKING_LEVEL)


@dataclass(frozen=True)
class CcpaCatalog:
    """Parsed CCPA model catalog with family groupings and per-model context windows."""

    available_models: tuple[CcpaModelInfo, ...]
    picker_entries: tuple[CcpaFamilyEntry | CcpaModelInfo, ...]
    families_by_id: Mapping[str, CcpaFamilyEntry]
    variant_to_family: Mapping[str, tuple[str, str]]
    max_tokens_by_model: Mapping[str, int]

    def context_window_for(self, model_id: str, default: int = 1_048_576) -> int:
        return self.max_tokens_by_model.get(model_id, default)

    def resolve_initial_model(self, requested_default: str = "") -> tuple[str, str]:
        """Return ``(concrete_model_id, thinking_level)`` defaulting strictly to ``low``."""
        available_ids = {m.model_id for m in self.available_models}
        candidate = requested_default.strip()
        if candidate:
            if candidate in available_ids:
                fam_lvl = self.variant_to_family.get(candidate)
                return candidate, (fam_lvl[1] if fam_lvl else DEFAULT_CCPA_THINKING_LEVEL)
            if candidate in self.families_by_id:
                fam = self.families_by_id[candidate]
                concrete = fam.low_default_model_id()
                lvl = self.variant_to_family.get(concrete, ("", DEFAULT_CCPA_THINKING_LEVEL))[1]
                return concrete, lvl

        if DEFAULT_CCPA_FAMILY_ID in self.families_by_id:
            fam = self.families_by_id[DEFAULT_CCPA_FAMILY_ID]
            concrete = fam.low_default_model_id()
            lvl = self.variant_to_family.get(concrete, ("", DEFAULT_CCPA_THINKING_LEVEL))[1]
            return concrete, lvl

        for entry in self.picker_entries:
            if isinstance(entry, CcpaFamilyEntry):
                concrete = entry.low_default_model_id()
                lvl = self.variant_to_family.get(concrete, ("", DEFAULT_CCPA_THINKING_LEVEL))[1]
                return concrete, lvl
            return entry.model_id, DEFAULT_CCPA_THINKING_LEVEL

        return DEFAULT_CCPA_CONCRETE_MODEL_ID, DEFAULT_CCPA_THINKING_LEVEL

    def resolve_model_switch(
        self,
        requested_value: str,
        *,
        explicit_thinking_level: str | None,
    ) -> tuple[str, str] | None:
        """Resolve a ``model`` config write to ``(concrete_model_id, thinking_level)``.

        When *explicit_thinking_level* is ``None`` (the user has not manually chosen a
        thinking level), switching to any family ALWAYS lands on ``low`` rather than
        inheriting or auto-routing to ``high`` / ``medium``.
        """
        available_ids = {m.model_id for m in self.available_models}
        if requested_value in self.families_by_id:
            fam = self.families_by_id[requested_value]
            desired_level = explicit_thinking_level or DEFAULT_CCPA_THINKING_LEVEL
            concrete = fam.nearest_model_id(desired_level)
            actual_level = self.variant_to_family.get(concrete, ("", desired_level))[1]
            return concrete, actual_level

        if requested_value in available_ids:
            fam_lvl = self.variant_to_family.get(requested_value)
            if fam_lvl is not None:
                return requested_value, fam_lvl[1]
            return requested_value, (explicit_thinking_level or DEFAULT_CCPA_THINKING_LEVEL)

        return None

    def resolve_thinking_switch(
        self,
        current_concrete_model: str,
        requested_level: str,
    ) -> tuple[str, str] | None:
        """Resolve a ``thinking_level`` config write on the current model's family."""
        fam_lvl = self.variant_to_family.get(current_concrete_model)
        if fam_lvl is None:
            return None
        fam = self.families_by_id.get(fam_lvl[0])
        if fam is None:
            return None
        concrete = fam.model_id_for_level(requested_level)
        if concrete is None:
            return None
        return concrete, requested_level

    def build_session_envelope(
        self,
        *,
        concrete_model_id: str,
        mode: str,
        session_modes: Sequence[tuple[str, str, str]],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the ACP ``session/new`` / ``set_config_option`` envelope."""
        fam_lvl = self.variant_to_family.get(concrete_model_id)
        current_family = self.families_by_id.get(fam_lvl[0]) if fam_lvl else None
        current_picker_value = (
            current_family.family_id if current_family is not None else concrete_model_id
        )

        model_options: list[dict[str, str]] = []
        for entry in self.picker_entries:
            if isinstance(entry, CcpaFamilyEntry):
                model_options.append({"value": entry.family_id, "name": entry.name})
            else:
                model_options.append({"value": entry.model_id, "name": entry.name})
        if all(opt["value"] != current_picker_value for opt in model_options):
            model_options.insert(0, {"value": current_picker_value, "name": current_picker_value})

        config_options: list[dict[str, Any]] = [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": current_picker_value,
                "options": model_options,
            }
        ]
        if current_family is not None:
            current_lvl = fam_lvl[1] if fam_lvl else DEFAULT_CCPA_THINKING_LEVEL
            config_options.append(
                {
                    "id": "thinking_level",
                    "name": "Thinking Level",
                    "category": "thought_level",
                    "type": "select",
                    "currentValue": current_lvl,
                    "options": [
                        {"value": lvl, "name": _CCPA_LEVEL_LABELS.get(lvl, lvl.title())}
                        for lvl, _mid in current_family.variants
                    ],
                }
            )
        config_options.append(
            {
                "id": "mode",
                "name": "Session Mode",
                "category": "mode",
                "type": "select",
                "currentValue": mode,
                "options": [
                    {"value": mid, "name": mname, "description": mdesc}
                    for mid, mname, mdesc in session_modes
                ],
            }
        )

        out: dict[str, Any] = {
            "modes": {
                "currentModeId": mode,
                "availableModes": [
                    {"id": mid, "name": mname, "description": mdesc}
                    for mid, mname, mdesc in session_modes
                ],
            },
            "configOptions": config_options,
            "models": {
                "availableModels": [
                    {
                        "modelId": m.model_id,
                        "name": m.name,
                        "description": m.description,
                    }
                    for m in self.available_models
                ],
                "currentModelId": concrete_model_id,
            },
        }
        if session_id is not None:
            out["sessionId"] = session_id
        return out


def build_ccpa_catalog_from_models(models: Sequence[CcpaModelInfo]) -> CcpaCatalog:
    """Group concrete CCPA models into families with ``low -> high`` variant ordering."""
    if not models:
        models = [
            CcpaModelInfo(model_id=mid, name=name, description=name, max_tokens=toks)
            for mid, name, toks in _FALLBACK_CCPA_MODELS
        ]

    # Put gemini-3.8-flash first in picker display when present, preserving CCPA order after.
    raw_families: dict[str, list[tuple[str, str]]] = {}
    family_names: dict[str, str] = {}
    variant_to_family: dict[str, tuple[str, str]] = {}
    max_tokens_by_model: dict[str, int] = {}

    for m in models:
        max_tokens_by_model[m.model_id] = m.max_tokens
        parsed = parse_ccpa_variant(m.model_id)
        if parsed is None:
            continue
        fam_id, level = parsed
        variant_to_family[m.model_id] = (fam_id, level)
        raw_families.setdefault(fam_id, []).append((level, m.model_id))
        family_names.setdefault(fam_id, _strip_level_suffix(m.name, level) or fam_id)

    families_by_id: dict[str, CcpaFamilyEntry] = {}
    for fam_id, pairs in raw_families.items():
        dedup: dict[str, str] = {}
        for lvl, mid in pairs:
            dedup.setdefault(lvl, mid)
        ordered = tuple(
            sorted(dedup.items(), key=lambda item: _LEVEL_RANK.get(item[0], 99))
        )
        families_by_id[fam_id] = CcpaFamilyEntry(
            family_id=fam_id,
            name=family_names[fam_id],
            variants=ordered,
        )

    picker_entries: list[CcpaFamilyEntry | CcpaModelInfo] = []
    seen_families: set[str] = set()
    for m in models:
        fam_lvl = variant_to_family.get(m.model_id)
        if fam_lvl is not None:
            fam_id = fam_lvl[0]
            if fam_id not in seen_families:
                seen_families.add(fam_id)
                picker_entries.append(families_by_id[fam_id])
        else:
            picker_entries.append(m)

    return CcpaCatalog(
        available_models=tuple(models),
        picker_entries=tuple(picker_entries),
        families_by_id=families_by_id,
        variant_to_family=variant_to_family,
        max_tokens_by_model=max_tokens_by_model,
    )


def parse_fetch_available_models_response(data: Mapping[str, Any]) -> CcpaCatalog:
    """Parse CCPA ``fetchAvailableModels`` JSON into a :class:`CcpaCatalog`."""
    raw_models = data.get("models")
    models_map: Mapping[str, Any] = raw_models if isinstance(raw_models, Mapping) else {}

    ordered_ids: list[str] = []
    seen: set[str] = set()
    sorts = data.get("agentModelSorts")
    if isinstance(sorts, list):
        for group in sorts:
            if not isinstance(group, Mapping):
                continue
            groups_inner = group.get("groups")
            if isinstance(groups_inner, list):
                for sub in groups_inner:
                    if isinstance(sub, Mapping) and isinstance(sub.get("modelIds"), list):
                        for mid in sub["modelIds"]:
                            if isinstance(mid, str) and mid not in seen:
                                seen.add(mid)
                                ordered_ids.append(mid)
            if isinstance(group.get("modelIds"), list):
                for mid in group["modelIds"]:
                    if isinstance(mid, str) and mid not in seen:
                        seen.add(mid)
                        ordered_ids.append(mid)

    if not ordered_ids:
        ordered_ids = [str(k) for k in models_map.keys() if isinstance(k, str)]

    parsed_list: list[CcpaModelInfo] = []
    for mid in ordered_ids:
        if not mid.startswith("gemini"):
            continue
        meta = models_map.get(mid)
        if not isinstance(meta, Mapping):
            continue
        display_name = str(meta.get("displayName") or mid).strip() or mid
        desc = str(meta.get("description") or display_name).strip()
        raw_max = meta.get("maxTokens") or meta.get("inputTokenLimit") or 1_048_576
        try:
            max_tokens = int(raw_max)
            if max_tokens <= 0:
                max_tokens = 1_048_576
        except (TypeError, ValueError):
            max_tokens = 1_048_576
        parsed_list.append(
            CcpaModelInfo(
                model_id=mid,
                name=display_name,
                description=desc,
                max_tokens=max_tokens,
            )
        )

    return build_ccpa_catalog_from_models(parsed_list)


@dataclass
class SessionStreamActivity:
    """Live telemetry for one ACP session's upstream CCPA HTTP stream(s)."""

    in_flight: int = 0
    started_mono: float = 0.0
    last_sse_mono: float = 0.0
    thoughts_token_count: int = 0
    saw_thought_text: bool = False
    active_model: str = ""


class CcpaOAuthManager:
    """Manages OAuth token refresh, CCPA onboarding, and cached model catalog."""

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        user_agent: str | None = None,
    ) -> None:
        self._env = env
        self.user_agent = user_agent or build_aidev_user_agent()
        self._lock = threading.Lock()
        self._cached_token: tuple[str, str, float] | None = None
        self._resolved_project_id: str = ""
        self._resolved_endpoint_url: str = ""
        self._cached_catalog: tuple[CcpaCatalog, float] | None = None

    def _effective_env_value(self, key: str) -> str:
        if self._env is not None:
            return self._env.get(key, "").strip()
        return os.environ.get(key, "").strip()

    def default_model_preference(self) -> str:
        return self._effective_env_value(_DEFAULT_ACP_MODEL_ENV) or DEFAULT_CCPA_CONCRETE_MODEL_ID

    def get_session_target(self) -> tuple[str, str, str]:
        """Return ``(access_token, project_id, endpoint_url)``."""
        raw = _read_token_json(self._env)
        if not raw:
            raise PermissionError("Google Antigravity OAuth token file is missing or unreadable.")

        client_id = str(raw.get("client_id") or "").strip()
        client_secret = str(raw.get("client_secret") or "").strip()
        refresh_token = str(raw.get("refresh_token") or "").strip()
        if not (client_id and client_secret and refresh_token):
            raise PermissionError("Google Antigravity OAuth token is incomplete; sign in again.")

        access_token = self._exchange_token(client_id, client_secret, refresh_token)

        env_endpoint = self._effective_env_value(_CCPA_BASE_URL_ENV)
        stored_project = str(raw.get("project_id") or "").strip()
        stored_endpoint = str(raw.get("endpoint_url") or "").strip()

        with self._lock:
            project_id = self._resolved_project_id or stored_project
            endpoint_url = env_endpoint or self._resolved_endpoint_url or stored_endpoint

        if not project_id or not endpoint_url:
            project_id, resolved_ep = self._run_load_code_assist(access_token)
            endpoint_url = env_endpoint or resolved_ep
            with self._lock:
                self._resolved_project_id = project_id
                self._resolved_endpoint_url = resolved_ep

        return access_token, project_id, endpoint_url.rstrip("/")

    def _exchange_token(self, client_id: str, client_secret: str, refresh_token: str) -> str:
        digest = hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:24]
        now = time.monotonic()
        with self._lock:
            if (
                self._cached_token is not None
                and self._cached_token[0] == digest
                and now < self._cached_token[2]
            ):
                return self._cached_token[1]

        form = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            _OAUTH_TOKEN_URL,
            data=form,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        opener = _build_https_opener()
        try:
            with opener.open(req, timeout=_HTTP_TIMEOUT_SECS) as resp:
                payload = json.loads(resp.read(_MAX_JSON_BYTES).decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 401):
                raise PermissionError(
                    "Google Antigravity OAuth session expired or revoked; please sign in again."
                ) from exc
            raise RuntimeError(f"OAuth token refresh failed with HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError(f"OAuth token refresh failed: {exc}") from exc

        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise PermissionError("OAuth token response contained no access_token.")
        expires_in = payload.get("expires_in", 3600)
        ttl = max(60.0, float(expires_in) - _TOKEN_EXPIRY_SKEW_SECS)
        with self._lock:
            self._cached_token = (digest, token, time.monotonic() + ttl)
        return token

    def _run_load_code_assist(self, access_token: str) -> tuple[str, str]:
        url = f"{_PROD_CCPA_ENDPOINT}/v1internal:loadCodeAssist"
        body = json.dumps({"metadata": {"ideType": "ANTIGRAVITY"}}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
        )
        opener = _build_https_opener()
        with opener.open(req, timeout=_HTTP_TIMEOUT_SECS) as resp:
            data = json.loads(resp.read(_MAX_JSON_BYTES).decode("utf-8", "replace"))
        paid_tier = data.get("paidTier") if isinstance(data.get("paidTier"), Mapping) else {}
        uses_gcp_tos = bool(paid_tier.get("usesGcpTos", False))
        endpoint = _PROD_CCPA_ENDPOINT if uses_gcp_tos else _DAILY_CCPA_ENDPOINT
        project_id = str(data.get("cloudaicompanionProject") or "aicode-consumers").strip()
        return project_id, endpoint

    def get_catalog(self) -> CcpaCatalog:
        """Fetch and cache the account's CCPA model catalog (with resilient fallback)."""
        now = time.monotonic()
        with self._lock:
            if self._cached_catalog is not None and now < self._cached_catalog[1]:
                return self._cached_catalog[0]

        try:
            access_token, project_id, endpoint_url = self.get_session_target()
            catalog = self._fetch_catalog_http(access_token, project_id, endpoint_url)
        except PermissionError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("Falling back to built-in CCPA model catalog", exc_info=True)
            catalog = build_ccpa_catalog_from_models(())

        with self._lock:
            self._cached_catalog = (catalog, time.monotonic() + _CATALOG_TTL_SECS)
        return catalog

    def _fetch_catalog_http(
        self, access_token: str, project_id: str, endpoint_url: str
    ) -> CcpaCatalog:
        # Try resolved tier endpoint first, then prod CCPA endpoint.
        opener = _build_https_opener()
        body = json.dumps({"project": project_id}).encode("utf-8")
        endpoints = [endpoint_url]
        if _PROD_CCPA_ENDPOINT not in endpoints:
            endpoints.append(_PROD_CCPA_ENDPOINT)

        last_err: Exception | None = None
        for base in endpoints:
            req = urllib.request.Request(
                f"{base.rstrip('/')}/v1internal:fetchAvailableModels",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "User-Agent": self.user_agent,
                },
            )
            try:
                with opener.open(req, timeout=_HTTP_TIMEOUT_SECS) as resp:
                    payload = json.loads(resp.read(_MAX_JSON_BYTES).decode("utf-8", "replace"))
                if isinstance(payload, Mapping) and payload.get("models"):
                    return parse_fetch_available_models_response(payload)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if last_err is not None:
            raise last_err
        return build_ccpa_catalog_from_models(())


def inject_include_thoughts(incoming_json: dict[str, Any]) -> dict[str, Any]:
    """Ensure ``generationConfig.thinkingConfig.includeThoughts`` defaults to ``True``.

    ``localharness`` omits ``includeThoughts`` on CCPA requests, which suppresses
    streamed ``{"thought": true, "text": "..."}`` summaries even while the model
    spends seconds generating internal reasoning tokens. Preserves an explicit
    ``False`` if a caller intentionally disabled thoughts.
    """
    gen_cfg = incoming_json.get("generationConfig")
    if not isinstance(gen_cfg, dict):
        gen_cfg = {}
        incoming_json["generationConfig"] = gen_cfg
    think_cfg = gen_cfg.get("thinkingConfig")
    if not isinstance(think_cfg, dict):
        think_cfg = {}
        gen_cfg["thinkingConfig"] = think_cfg
    if "includeThoughts" not in think_cfg and "include_thoughts" not in think_cfg:
        think_cfg["includeThoughts"] = True
    return incoming_json


class _CcpaThreadingHttpServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[http.server.BaseHTTPRequestHandler],
        oauth_mgr: CcpaOAuthManager,
        secret_token: str,
        activity_lock: threading.Lock,
        activities: dict[str, SessionStreamActivity],
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.oauth_mgr = oauth_mgr
        self.secret_token = secret_token
        self.activity_lock = activity_lock
        self.activities = activities


class _CcpaProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forwards ``localharness`` Gemini REST calls to CCPA with OAuth + thought streaming."""

    def log_message(self, format_str: str, *args: Any) -> None:  # noqa: A002
        del format_str, args

    def _drain_body(self) -> None:
        try:
            rem = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return
        while rem > 0:
            try:
                chunk = self.rfile.read(min(rem, 65536))
            except OSError:
                break
            if not chunk:
                break
            rem -= len(chunk)

    def _reply_error(self, code: int, msg: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(msg)))
        self.end_headers()
        self.wfile.write(msg)

    def do_POST(self) -> None:  # noqa: N802
        assert isinstance(self.server, _CcpaThreadingHttpServer)
        tok = self.headers.get("X-ACP-Proxy-Token") or ""
        if not tok or not secrets.compare_digest(tok, self.server.secret_token):
            self._drain_body()
            self._reply_error(401, b"Unauthorized")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length)
            incoming = json.loads(raw_body.decode("utf-8"))
            if not isinstance(incoming, dict):
                raise ValueError("Expected JSON object")
        except (OSError, UnicodeDecodeError, ValueError):
            self._reply_error(400, b"Invalid JSON request body")
            return

        match = _MODEL_PATH_RE.search(self.path)
        if not match:
            self._reply_error(400, b"Model missing in path")
            return
        model = match.group(1)

        try:
            access_token, project_id, endpoint_url = self.server.oauth_mgr.get_session_target()
        except Exception as exc:  # noqa: BLE001
            self._reply_error(500, f"OAuth error: {exc}".encode("utf-8", "replace"))
            return

        inject_include_thoughts(incoming)
        is_stream = "streamGenerateContent" in self.path
        acp_session_id = (self.headers.get("X-ACP-Session-Id") or "").strip()

        now = time.monotonic()
        if acp_session_id:
            with self.server.activity_lock:
                act = self.server.activities.setdefault(acp_session_id, SessionStreamActivity())
                act.in_flight += 1
                act.started_mono = now
                act.last_sse_mono = 0.0
                act.thoughts_token_count = 0
                act.saw_thought_text = False
                act.active_model = model

        ccpa_action = "streamGenerateContent?alt=sse" if is_stream else "generateContent"
        url = f"{endpoint_url}/v1internal:{ccpa_action}"
        wrapped = json.dumps(
            {"project": project_id, "model": model, "request": incoming}
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=wrapped,
            method="POST",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "User-Agent": self.server.oauth_mgr.user_agent,
            },
        )
        opener = _build_https_opener()

        try:
            with opener.open(req, timeout=_STREAM_TIMEOUT_SECS) as resp:
                status = getattr(resp, "status", 200) or 200
                if is_stream:
                    self.send_response(status)
                    for k, v in resp.headers.items():
                        kl = k.lower()
                        if kl == "content-length" or (
                            kl == "transfer-encoding" and v.lower() == "chunked"
                        ):
                            continue
                        self.send_header(k, v)
                    self.end_headers()
                    for raw_line in resp:
                        stripped = raw_line.strip()
                        if not stripped:
                            self.wfile.write(b"\n")
                            self.wfile.flush()
                            continue
                        if not stripped.startswith(b"data:"):
                            self.wfile.write(stripped + b"\n")
                            self.wfile.flush()
                            continue
                        data_part = stripped[5:].strip()
                        try:
                            ccpa_obj = json.loads(data_part.decode("utf-8"))
                            inner = ccpa_obj.get("response", ccpa_obj)
                            if acp_session_id and isinstance(inner, Mapping):
                                self._record_sse_chunk(acp_session_id, inner)
                            out_line = b"data: " + json.dumps(inner).encode("utf-8") + b"\n"
                        except (UnicodeDecodeError, ValueError, TypeError):
                            out_line = stripped + b"\n"
                        self.wfile.write(out_line)
                        self.wfile.flush()
                else:
                    body = resp.read()
                    try:
                        ccpa_obj = json.loads(body.decode("utf-8"))
                        unwrapped = json.dumps(ccpa_obj.get("response", ccpa_obj)).encode("utf-8")
                    except (UnicodeDecodeError, ValueError, TypeError):
                        unwrapped = body
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(unwrapped)))
                    self.end_headers()
                    self.wfile.write(unwrapped)
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read()
            except Exception:  # noqa: BLE001
                err_body = f"Upstream HTTP {exc.code}".encode("utf-8")
            self.send_response(exc.code)
            self.send_header(
                "Content-Type", exc.headers.get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            try:
                self._reply_error(502, f"CCPA upstream error: {exc}".encode("utf-8", "replace"))
            except OSError:
                pass
        finally:
            if acp_session_id:
                with self.server.activity_lock:
                    act = self.server.activities.get(acp_session_id)
                    if act is not None:
                        act.in_flight = max(0, act.in_flight - 1)

    def _record_sse_chunk(self, acp_session_id: str, inner: Mapping[str, Any]) -> None:
        assert isinstance(self.server, _CcpaThreadingHttpServer)
        usage = inner.get("usageMetadata")
        thoughts_tok = 0
        if isinstance(usage, Mapping):
            raw_tok = usage.get("thoughtsTokenCount")
            if isinstance(raw_tok, int) and raw_tok > 0:
                thoughts_tok = raw_tok
        saw_thought = False
        cands = inner.get("candidates")
        if isinstance(cands, list) and cands and isinstance(cands[0], Mapping):
            content = cands[0].get("content")
            if isinstance(content, Mapping) and isinstance(content.get("parts"), list):
                for part in content["parts"]:
                    if isinstance(part, Mapping) and part.get("thought") and part.get("text"):
                        saw_thought = True
                        break
        with self.server.activity_lock:
            act = self.server.activities.get(acp_session_id)
            if act is not None:
                act.last_sse_mono = time.monotonic()
                if thoughts_tok > act.thoughts_token_count:
                    act.thoughts_token_count = thoughts_tok
                if saw_thought:
                    act.saw_thought_text = True


@dataclass
class CcpaLoopbackProxy:
    """Lifecycle wrapper around the local loopback CCPA proxy server."""

    oauth_mgr: CcpaOAuthManager
    secret_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    _server: _CcpaThreadingHttpServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _activity_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _activities: dict[str, SessionStreamActivity] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("CCPA loopback proxy is not started")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def is_alive(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    def start(self) -> str:
        if self.is_alive:
            return self.base_url
        self.stop()
        srv = _CcpaThreadingHttpServer(
            ("127.0.0.1", 0),
            _CcpaProxyHandler,
            self.oauth_mgr,
            self.secret_token,
            self._activity_lock,
            self._activities,
        )
        t = threading.Thread(target=srv.serve_forever, name="antigravity-ccpa-proxy", daemon=True)
        t.start()
        self._server = srv
        self._thread = t
        return self.base_url

    def snapshot_activity(self, session_id: str) -> SessionStreamActivity | None:
        with self._activity_lock:
            act = self._activities.get(session_id)
            if act is None:
                return None
            return SessionStreamActivity(
                in_flight=act.in_flight,
                started_mono=act.started_mono,
                last_sse_mono=act.last_sse_mono,
                thoughts_token_count=act.thoughts_token_count,
                saw_thought_text=act.saw_thought_text,
                active_model=act.active_model,
            )

    def clear_session_activity(self, session_id: str) -> None:
        with self._activity_lock:
            self._activities.pop(session_id, None)

    def stop(self) -> None:
        srv, self._server = self._server, None
        t, self._thread = self._thread, None
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:  # noqa: BLE001
                pass
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
