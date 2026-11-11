"""Google Antigravity ACP runtime resolution and launch (standalone + Python SDK).

Supports two execution modes behind one per-session ``ProcessAdapter``:

1. **Standalone ACP runtime** (``mode="standalone"``): Google's two-file distribution
   ``agy_acp_server`` + ``localharness_external`` (pointed to via
   ``ANTIGRAVITY_HARNESS_PATH``).
2. **In-tree Python SDK adapter** (``mode="sdk"``): ``kirocrew-antigravity-acp`` backed by
   the open-source ``google-antigravity`` Python wheel and its bundled ``localharness``
   binary, wiring ``policy.ask_user`` into ``LocalAgentConfig.policies`` for pre-execution
   ``session/request_permission`` gating.

``ANTIGRAVITY_ACP_MODE`` selects ``"auto"`` (default: standalone pair when present, else
Python SDK adapter when ``google-antigravity`` is installed), ``"sdk"``, or
``"standalone"``.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import shutil
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from kiro_crew import acp_tool_gate, platform_compat
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.harness.base import ProcessAdapter, SpawnContext, SpawnPlan
from kiro_crew.acp.transport_errors import AcpError
from kiro_crew.agent_sdk.backends import ACP_BACKEND_ANTIGRAVITY
from kiro_crew.env import describe_search_path

logger = logging.getLogger(__name__)

__all__ = [
    "ANTIGRAVITY_ACP_BIN_ENV",
    "ANTIGRAVITY_ACP_MODE_ENV",
    "ANTIGRAVITY_HARNESS_PATH_ENV",
    "ANTIGRAVITY_PROTOCOL_VERSION",
    "ANTIGRAVITY_SDK_ADAPTER_BIN",
    "ANTIGRAVITY_SDK_HARNESS_BIN",
    "ANTIGRAVITY_SDK_VERSION_PREFIX",
    "ANTIGRAVITY_STRICT_VERSION_PIN_ENV",
    "ANTIGRAVITY_VERIFIED_VERSION",
    "AntigravityLaunch",
    "AntigravityResolution",
    "antigravity_cached_resolution",
    "antigravity_not_found_message",
    "forget_antigravity_resolution",
    "is_accepted_antigravity_version",
    "probe_antigravity_runtime",
    "resolve_antigravity_runtime",
]

# The ACP protocol and baseline verified standalone release.
ANTIGRAVITY_PROTOCOL_VERSION = 1
ANTIGRAVITY_VERIFIED_VERSION = "1.4.4"
ANTIGRAVITY_SDK_VERSION_PREFIX = "sdk-"

# The two published standalone archive members. Windows adds ``.exe`` to both.
ANTIGRAVITY_ACP_BIN = "agy_acp_server.exe" if platform_compat.IS_WINDOWS else "agy_acp_server.par"
ANTIGRAVITY_HARNESS_BIN = (
    "localharness_external.exe" if platform_compat.IS_WINDOWS else "localharness_external"
)

# The Python SDK console wrapper and the wheel-bundled Go harness binary.
ANTIGRAVITY_SDK_ADAPTER_BIN = (
    "kirocrew-antigravity-acp.exe" if platform_compat.IS_WINDOWS else "kirocrew-antigravity-acp"
)
ANTIGRAVITY_SDK_HARNESS_BIN = "localharness.exe" if platform_compat.IS_WINDOWS else "localharness"

#: Operator override for the standalone main executable.
ANTIGRAVITY_ACP_BIN_ENV = "ANTIGRAVITY_ACP_BIN"

#: Runtime selection: ``auto`` (default), ``sdk``, or ``standalone``.
ANTIGRAVITY_ACP_MODE_ENV = "ANTIGRAVITY_ACP_MODE"

#: Set to ``1`` / ``true`` to refuse standalone 1.x versions other than 1.4.4.
ANTIGRAVITY_STRICT_VERSION_PIN_ENV = "ANTIGRAVITY_STRICT_VERSION_PIN"

#: The helper variable Antigravity's executable (and SDK) reads.
ANTIGRAVITY_HARNESS_PATH_ENV = "ANTIGRAVITY_HARNESS_PATH"

ANTIGRAVITY_INSTALL_HINT = (
    "Install the Google Antigravity Python SDK with `pip install google-antigravity` "
    "(used automatically by `kirocrew-antigravity-acp`), or install the standalone "
    f"release (baseline verified on {ANTIGRAVITY_VERIFIED_VERSION}): download "
    f"agy-acp-server-{ANTIGRAVITY_VERIFIED_VERSION}-PLATFORM.zip from "
    "https://dl.google.com/agy-extensions/releases/OS/ (OS is macos, linux or windows; "
    "PLATFORM is darwin-arm64, darwin-x86_64, linux-x86_64, linux-arm64, windows-x86_64 "
    "or windows-arm64), or set ANTIGRAVITY_ACP_BIN to its agy_acp_server executable."
)

_STANDALONE_COMPAT_RE = re.compile(r"1\.\d+\.\d+(?:[-+][\w.]+)?\Z")
_warned_unverified_versions: set[str] = set()


def is_accepted_antigravity_version(
    version: str | None, *, env: Mapping[str, str] | None = None
) -> bool:
    """True when *version* is an accepted SDK adapter or standalone 1.x runtime."""
    if not version:
        return False
    if version == ANTIGRAVITY_VERIFIED_VERSION:
        return True
    if (
        version.startswith(ANTIGRAVITY_SDK_VERSION_PREFIX)
        and len(version) > len(ANTIGRAVITY_SDK_VERSION_PREFIX)
        and version[len(ANTIGRAVITY_SDK_VERSION_PREFIX)].isdigit()
    ):
        return True

    environ = env if env is not None else os.environ
    strict = environ.get(ANTIGRAVITY_STRICT_VERSION_PIN_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not strict and _STANDALONE_COMPAT_RE.match(version):
        if version not in _warned_unverified_versions:
            _warned_unverified_versions.add(version)
            logger.warning(
                "Google Antigravity ACP %s differs from baseline verified release %s; "
                "proceeding in 1.x compatibility mode (set %s=1 to enforce exact pin).",
                version,
                ANTIGRAVITY_VERIFIED_VERSION,
                ANTIGRAVITY_STRICT_VERSION_PIN_ENV,
            )
        return True
    return False


@dataclass(frozen=True)
class AntigravityResolution:
    """The two executables one Antigravity install contributes."""

    server: str | None
    helper: str | None
    search_path: str
    mode: str = "standalone"


_cache: AntigravityResolution | None = None
_cache_lock = threading.Lock()
_cache_generation = 0


def _normalize_executable(path: str | None) -> str | None:
    """Return *path* as an absolute executable path, or ``None`` when it is not executable."""
    if not path or not platform_compat.is_executable_file(path):
        return None
    return os.path.realpath(path) if platform_compat.IS_WINDOWS else os.path.abspath(path)


def _find_main_executable(env: Mapping[str, str], search_path: str) -> str | None:
    override = env.get(ANTIGRAVITY_ACP_BIN_ENV, "")
    if override:
        return _normalize_executable(os.path.expanduser(override))

    from kiro_crew.acp.client import _mise_which

    mise_resolved = _mise_which(ANTIGRAVITY_ACP_BIN)
    if mise_resolved:
        return _normalize_executable(mise_resolved)
    return _normalize_executable(shutil.which(ANTIGRAVITY_ACP_BIN, path=search_path))


def _find_helper(env: Mapping[str, str], server: str | None, search_path: str) -> str | None:
    explicit = env.get(ANTIGRAVITY_HARNESS_PATH_ENV, "")
    if explicit:
        return _normalize_executable(os.path.expanduser(explicit))

    if server:
        sibling = _normalize_executable(str(Path(server).with_name(ANTIGRAVITY_HARNESS_BIN)))
        if sibling:
            return sibling
    return _normalize_executable(shutil.which(ANTIGRAVITY_HARNESS_BIN, path=search_path))


def _find_sdk_adapter_executable(search_path: str) -> str | None:
    """Locate ``kirocrew-antigravity-acp`` beside ``sys.executable`` or on *search_path*."""
    sibling = _normalize_executable(
        str(Path(sys.executable).with_name(ANTIGRAVITY_SDK_ADAPTER_BIN))
    )
    if sibling:
        return sibling
    return _normalize_executable(shutil.which(ANTIGRAVITY_SDK_ADAPTER_BIN, path=search_path))


def _find_sdk_bundled_helper(env: Mapping[str, str]) -> str | None:
    """Locate ``localharness`` inside the installed ``google-antigravity`` wheel."""
    explicit = env.get(ANTIGRAVITY_HARNESS_PATH_ENV, "")
    if explicit:
        return _normalize_executable(os.path.expanduser(explicit))
    try:
        spec = importlib.util.find_spec("google.antigravity")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = _normalize_executable(str(Path(location) / "bin" / ANTIGRAVITY_SDK_HARNESS_BIN))
        if candidate:
            return candidate
    return None


def _resolve_sdk_runtime(env: Mapping[str, str], search_path: str) -> AntigravityResolution:
    return AntigravityResolution(
        server=_find_sdk_adapter_executable(search_path),
        helper=_find_sdk_bundled_helper(env),
        search_path=search_path,
        mode="sdk",
    )


def _resolve_antigravity_runtime() -> AntigravityResolution:
    """Resolve Antigravity ACP components against one environment snapshot. Blocking."""
    from kiro_crew.acp.client import augmented_path

    env = dict(os.environ)
    search_path = augmented_path(env.get("PATH", ""))
    requested_mode = env.get(ANTIGRAVITY_ACP_MODE_ENV, "auto").strip().lower() or "auto"

    if requested_mode == "sdk":
        return _resolve_sdk_runtime(env, search_path)

    server = _find_main_executable(env, search_path)
    helper = _find_helper(env, server, search_path)
    standalone = AntigravityResolution(
        server=server,
        helper=helper,
        search_path=search_path,
        mode="standalone",
    )
    if (
        requested_mode == "standalone"
        or env.get(ANTIGRAVITY_ACP_BIN_ENV, "")
        or server is not None
        or helper is not None
    ):
        return standalone

    sdk_res = _resolve_sdk_runtime(env, search_path)
    if sdk_res.server and sdk_res.helper:
        return sdk_res
    return standalone


def probe_antigravity_runtime() -> AntigravityResolution:
    """Resolve both components without changing what the next spawn will use."""
    return _resolve_antigravity_runtime()


def resolve_antigravity_runtime() -> AntigravityResolution:
    """Return both Antigravity executables, cached for this gateway process."""
    global _cache
    with _cache_lock:
        cached = _cache
        generation = _cache_generation
    if cached is not None:
        return cached

    resolved = _resolve_antigravity_runtime()
    with _cache_lock:
        if _cache_generation == generation and _cache is None:
            _cache = resolved
        return _cache or resolved


def antigravity_cached_resolution() -> AntigravityResolution | None:
    """The resolution this process has already cached, if any."""
    with _cache_lock:
        return _cache


def forget_antigravity_resolution() -> None:
    """Drop the process-lifetime resolution after an owner-gated re-check."""
    global _cache, _cache_generation
    with _cache_lock:
        _cache = None
        _cache_generation += 1


def antigravity_not_found_message(resolution: AntigravityResolution) -> str:
    """The actionable error for a half-installed or absent Antigravity pair."""
    searched = describe_search_path(resolution.search_path)
    if resolution.mode == "sdk":
        missing_sdk = []
        if not resolution.server:
            missing_sdk.append(ANTIGRAVITY_SDK_ADAPTER_BIN)
        if not resolution.helper:
            missing_sdk.append(f"google-antigravity ({ANTIGRAVITY_SDK_HARNESS_BIN})")
        return (
            f"Google Antigravity Python SDK runtime is not fully installed: missing "
            f"{', '.join(missing_sdk)} ({searched}). {ANTIGRAVITY_INSTALL_HINT}"
        )
    missing = []
    if not resolution.server:
        missing.append(ANTIGRAVITY_ACP_BIN)
    if not resolution.helper:
        missing.append(ANTIGRAVITY_HARNESS_BIN)
    return (
        f"Google Antigravity ACP is not fully installed: missing {', '.join(missing)} "
        f"({searched}). The main executable and its localharness_external helper are "
        f"distributed together and must stay in the same directory. "
        f"{ANTIGRAVITY_INSTALL_HINT}"
    )


def _standalone_needs_empty_uid_flag() -> bool:
    """True on POSIX hosts without a ``nobody`` group (Debian/Ubuntu/WSL2).

    Google's standalone ``agy_acp_server.par`` embeds ``init_google.cc`` with default
    ``--uid=nobody`` / ``--gid==uid`` and aborts at startup when ``getgrnam("nobody")``
    fails (Debian/Ubuntu name GID 65534 ``nogroup``). Passing ``--uid=`` disables that
    lookup without requiring root or ``/etc/group`` edits.
    """
    if platform_compat.IS_WINDOWS:
        return False
    try:
        import grp

        grp.getgrnam("nobody")
    except KeyError:
        return True
    except Exception:  # noqa: BLE001
        return False
    return False


class AntigravityLaunch(ProcessAdapter):
    """Launch Google Antigravity's standalone ACP server or Python SDK ACP adapter."""

    backend = ACP_BACKEND_ANTIGRAVITY

    def __init__(self) -> None:
        self._helper_path = ""

    async def resolve_spawn(self, ctx: SpawnContext) -> SpawnPlan:
        """Resolve the pair, warm MCP, then apply the enforced-host credential mask."""
        session = ctx.session
        assert session is not None, "antigravity is launched for one session"
        resolution = await asyncio.to_thread(resolve_antigravity_runtime)
        if not resolution.server or not resolution.helper:
            raise AcpError(antigravity_not_found_message(resolution))

        await session._prepare_session_mcp()
        hidden = await launch_mod._run_preflight_bounded(
            launch_mod._sandbox_preflight, self.backend, ctx.sandbox_mode
        )
        self._helper_path = resolution.helper
        label = "kirocrew-antigravity-acp" if resolution.mode == "sdk" else "agy_acp_server"
        argv = [resolution.server]
        if resolution.mode == "standalone" and _standalone_needs_empty_uid_flag():
            argv.append("--uid=")
        return SpawnPlan(
            argv=argv,
            spawn_label=label,
            stderr_label=label,
            extra_hidden_dirs=hidden,
            extra_expose_files=acp_tool_gate.adapter_expose_files(self.backend, hidden),
        )

    def apply_spawn_env(self, env: dict[str, str], *, spawned_binary: str | None = None) -> None:
        """Point the server at the exact helper resolved for this launch."""
        if self._helper_path:
            env[ANTIGRAVITY_HARNESS_PATH_ENV] = self._helper_path
