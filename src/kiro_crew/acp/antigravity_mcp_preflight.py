"""Sandboxed MCP ``initialize`` preflight for Google Antigravity sessions.

Why Antigravity needs an ``initialize`` preflight beyond ``prune_unstartable``
------------------------------------------------------------------------------
Google Antigravity's ``localharness`` / ``localharness_external`` runs inside
Kiro Crew's OS sandbox where :func:`kiro_crew.acp.launch._sandbox_preflight`
masks foreign credential homes (such as ``~/.aws``). Unlike ``kiro-cli`` or
``goose`` (which drop a single failing MCP server and continue), Antigravity's
harness aborts the **entire chat turn** whenever *any* MCP server in
``session/new``'s ``mcpServers`` array fails its ``initialize`` handshake
(for example ``aws-mcp`` via ``uvx mcp-proxy-for-aws``, which raises
``ValueError: No AWS credentials available`` inside the ``~/.aws``-masked
sandbox when not routed through Crew's host MCP broker).

:func:`kiro_crew.acp.mcp_startable.prune_unstartable` only verifies that
``uvx`` exists on ``PATH``, so such a server still reaches ``session/new`` and
breaks every turn. Probing third-party stdio MCP servers inside the exact same
sandbox mask before ``session/new`` prunes only deterministic ``initialize``
failures, logs a clear warning, and records the server in the session's
``McpSessionReport`` while keeping every healthy server (and any slow-starting
server still alive at the probe deadline).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from kiro_crew import acp_tool_gate, agent_scratch, platform_compat
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp.mcp_startable import PrunedServer
from kiro_crew.agent_sdk.backends import ACP_BACKEND_ANTIGRAVITY
from kiro_crew.env import append_mcp_path_dirs, augmented_path
from kiro_crew.sandbox import scrub_agent_subprocess_env
from kiro_crew.sandbox_tiers import configured_sandbox_mode
from kiro_crew.sandbox_wrap import wrap_argv

logger = logging.getLogger("kiro_crew.acp.client")

#: Environment toggle: ``"0"`` disables preflight; ``"1"`` forces it even under pytest.
ANTIGRAVITY_MCP_PREFLIGHT_ENV = "ANTIGRAVITY_MCP_PREFLIGHT"

#: Per-server wall-clock budget for the MCP ``initialize`` handshake.
_PROBE_TIMEOUT_SECS = 8.0

#: Cache TTL and cap for settled preflight verdicts (healthy or deterministic failure).
_CACHE_TTL_SECS = 120.0
_CACHE_MAX_ENTRIES = 64
_PREFLIGHT_CACHE: dict[str, tuple[float, str]] = {}
_CACHE_LOCK = threading.Lock()

_INIT_REQUEST = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kirocrew-mcp-preflight", "version": "1.0"},
            },
        },
        separators=(",", ":"),
    )
    + "\n"
).encode("utf-8")


def preflight_enabled_for_live_session() -> bool:
    """Whether live Antigravity sessions should run the sandboxed MCP preflight.

    Enabled by default in production. Skipped under ``pytest`` unless
    ``ANTIGRAVITY_MCP_PREFLIGHT=1`` so synthetic session tests using dummy stdio
    commands (e.g. bare ``sys.executable``) do not wait on an MCP handshake.
    """
    flag = os.environ.get(ANTIGRAVITY_MCP_PREFLIGHT_ENV, "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return "PYTEST_CURRENT_TEST" not in os.environ


def clear_preflight_cache() -> None:
    """Drop all cached MCP preflight verdicts (used by tests and re-checks)."""
    with _CACHE_LOCK:
        _PREFLIGHT_CACHE.clear()


def _is_crew_managed_server(element: Mapping[str, Any]) -> bool:
    """True for Crew's own control-plane servers and host-brokered stubs."""
    name = str(element.get("name") or "")
    if name.startswith("kirocrew-"):
        return True
    args = element.get("args")
    if isinstance(args, list) and any(
        isinstance(arg, str) and "kiro_crew.mcp_gateway.stub" in arg for arg in args
    ):
        return True
    return False


def _element_env_dict(element: Mapping[str, Any]) -> dict[str, str]:
    """Extract ``[{"name": k, "value": v}]`` or ``{k: v}`` env pairs from *element*."""
    raw = element.get("env")
    out: dict[str, str] = {}
    if isinstance(raw, Mapping):
        for k, v in raw.items():
            if isinstance(k, str) and v is not None:
                out[k] = str(v)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping):
                k = item.get("name")
                v = item.get("value")
                if isinstance(k, str) and k and v is not None:
                    out[k] = str(v)
    return out


def _cache_key(
    element: Mapping[str, Any],
    *,
    sandbox_mode: str,
    hidden: tuple[str, ...],
) -> str:
    canonical = json.dumps(
        {
            "name": str(element.get("name") or ""),
            "command": str(element.get("command") or ""),
            "args": [str(a) for a in (element.get("args") or ()) if a is not None],
            "env": sorted(_element_env_dict(element).items()),
            "mode": sandbox_mode,
            "hidden": list(hidden),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _get_cached_verdict(key: str) -> str | None:
    now = time.monotonic()
    with _CACHE_LOCK:
        entry = _PREFLIGHT_CACHE.get(key)
        if entry is None:
            return None
        ts, reason = entry
        if now - ts > _CACHE_TTL_SECS:
            _PREFLIGHT_CACHE.pop(key, None)
            return None
        return reason


def _store_cached_verdict(key: str, reason: str) -> None:
    now = time.monotonic()
    with _CACHE_LOCK:
        if key not in _PREFLIGHT_CACHE and len(_PREFLIGHT_CACHE) >= _CACHE_MAX_ENTRIES:
            oldest = next(iter(_PREFLIGHT_CACHE), None)
            if oldest is not None:
                _PREFLIGHT_CACHE.pop(oldest, None)
        _PREFLIGHT_CACHE[key] = (now, reason)


def _summarize_stderr(stderr_lines: list[str], exit_code: int | None) -> str:
    """Pick the most informative failure line from captured stderr."""
    cleaned = [line.strip() for line in stderr_lines if line.strip()]
    for line in reversed(cleaned):
        lower = line.lower()
        if any(
            marker in lower
            for marker in (
                "valueerror:",
                "runtimeerror:",
                "mcperror:",
                "permissionerror:",
                "credential",
                "error:",
                "failed",
            )
        ) and not line.startswith("File "):
            return f"MCP initialize failed in sandbox: {line[:200]}"
    if cleaned:
        return f"MCP initialize failed in sandbox: {cleaned[-1][:200]}"
    if exit_code is not None:
        return f"MCP server exited before initialize completed (code {exit_code})"
    return "MCP server closed stdout before answering initialize"


def _terminate_probe_process(proc: subprocess.Popen[bytes]) -> None:
    """Terminate the probe process group cleanly without leaking children."""
    if proc.poll() is not None:
        return
    if platform_compat.IS_POSIX:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    if platform_compat.IS_POSIX:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
    else:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _probe_stdio_server_sync(
    element: Mapping[str, Any],
    *,
    sandbox_mode: str,
    hidden: tuple[str, ...],
    expose: tuple[str, ...],
    timeout: float = _PROBE_TIMEOUT_SECS,
) -> tuple[str, bool]:
    """Run one stdio MCP ``initialize`` inside Antigravity's sandbox.

    Returns ``(failure_reason, cacheable)``. ``failure_reason == ""`` means keep
    the server. Deterministic failures (JSON-RPC ``error`` response or premature
    EOF / non-zero exit) return a non-empty reason with ``cacheable=True``. A
    server still alive at *timeout* (e.g. ``npx`` downloading a cold package) is
    kept with ``cacheable=False``.
    """
    command = element.get("command")
    if not isinstance(command, str) or not command or element.get("url"):
        return ("", True)

    raw_args = element.get("args")
    args = [str(a) for a in raw_args if a is not None] if isinstance(raw_args, list) else []
    argv = [command, *args]

    scratch_dir = None
    cleanup_path: str | None = None
    try:
        try:
            scratch_dir = agent_scratch.allocate_scratch("agy-mcp-probe")
        except (OSError, agent_scratch.ScratchBoundaryError):
            scratch_dir = None
        private_dirs = (str(scratch_dir),) if scratch_dir is not None else ()

        wrapped_argv, cleanup_path = wrap_argv(
            argv,
            mode=sandbox_mode,
            strip_python_env=True,
            extra_hidden_dirs=hidden,
            extra_private_dirs=private_dirs,
            extra_expose_files=expose,
        )

        env = dict(os.environ)
        launch_mod.drop_foreign_inherited_port(env)
        agent_scratch.drop_inherited_scratch_env(env)
        env["PATH"] = append_mcp_path_dirs(augmented_path(env.get("PATH", "")))
        env = scrub_agent_subprocess_env(env, forward_ssh_auth_sock=False)
        env.update(_element_env_dict(element))
        if scratch_dir is not None:
            env.update(agent_scratch.scratch_env(scratch_dir))

        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": env,
        }
        if platform_compat.IS_POSIX:
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = (
                platform_compat.CREATE_NEW_PROCESS_GROUP | platform_compat._SUBPROCESS_NO_WINDOW
            )

        try:
            proc = subprocess.Popen(wrapped_argv, **popen_kwargs)
        except OSError as exc:
            return (f"MCP command failed to spawn: {exc}", True)

        stdout_q: queue.Queue[bytes | None] = queue.Queue()
        stderr_lines: list[str] = []

        def _read_stdout() -> None:
            assert proc.stdout is not None
            try:
                for raw_line in iter(proc.stdout.readline, b""):
                    stdout_q.put(raw_line)
            except OSError:
                pass
            finally:
                stdout_q.put(None)

        def _read_stderr() -> None:
            assert proc.stderr is not None
            try:
                for raw_line in iter(proc.stderr.readline, b""):
                    if len(stderr_lines) < 64:
                        stderr_lines.append(raw_line.decode("utf-8", "replace"))
            except OSError:
                pass

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_out.start()
        t_err.start()

        try:
            assert proc.stdin is not None
            try:
                proc.stdin.write(_INIT_REQUEST)
                proc.stdin.flush()
            except OSError:
                pass

            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if proc.poll() is None:
                        # Still alive (e.g. cold package download): do not prune.
                        return ("", False)
                    t_err.join(timeout=0.3)
                    return (_summarize_stderr(stderr_lines, proc.returncode), True)

                try:
                    item = stdout_q.get(timeout=min(remaining, 0.25))
                except queue.Empty:
                    if proc.poll() is not None and stdout_q.empty():
                        t_err.join(timeout=0.3)
                        return (_summarize_stderr(stderr_lines, proc.returncode), True)
                    continue

                if item is None:
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
                    t_err.join(timeout=0.3)
                    return (_summarize_stderr(stderr_lines, proc.returncode), True)

                text = item.decode("utf-8", "replace").strip()
                if not text or not text.startswith("{"):
                    continue
                try:
                    payload = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(payload, dict) or payload.get("id") != 1:
                    continue
                if isinstance(payload.get("result"), dict):
                    return ("", True)
                if "error" in payload:
                    err_obj = payload["error"]
                    msg = (
                        str(err_obj.get("message") or err_obj)
                        if isinstance(err_obj, Mapping)
                        else str(err_obj)
                    )
                    return (f"MCP initialize error in sandbox: {msg[:200]}", True)
        finally:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            _terminate_probe_process(proc)
    except Exception:
        logger.debug("Antigravity MCP preflight encountered an internal error; keeping server", exc_info=True)
        return ("", False)
    finally:
        if cleanup_path:
            launch_mod._unlink_readback_launcher(cleanup_path)
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)


def _prepare_sandbox_spec(sandbox_mode: str | None) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    mode = sandbox_mode or configured_sandbox_mode()
    hidden = launch_mod._sandbox_preflight(ACP_BACKEND_ANTIGRAVITY, mode)
    expose = acp_tool_gate.adapter_expose_files(ACP_BACKEND_ANTIGRAVITY, hidden)
    return mode, hidden, expose


async def probe_antigravity_mcp_servers(
    servers: Sequence[Any],
    *,
    sandbox_mode: str | None = None,
    timeout: float = _PROBE_TIMEOUT_SECS,
) -> tuple[list[Any], list[PrunedServer]]:
    """Probe third-party stdio MCP servers inside Antigravity's sandbox in parallel.

    Returns ``(kept_servers, pruned_servers)`` in original order. Never raises:
    any internal error keeps the server unchanged.
    """
    if not servers:
        return [], []
    try:
        mode, hidden, expose = await asyncio.to_thread(_prepare_sandbox_spec, sandbox_mode)
    except Exception:
        logger.debug("Could not prepare Antigravity MCP preflight sandbox spec", exc_info=True)
        return list(servers), []

    async def _judge(element: Any) -> str:
        if not isinstance(element, Mapping):
            return ""
        if element.get("url") or not element.get("command") or _is_crew_managed_server(element):
            return ""
        key = _cache_key(element, sandbox_mode=mode, hidden=hidden)
        cached = _get_cached_verdict(key)
        if cached is not None:
            return cached
        reason, cacheable = await asyncio.to_thread(
            _probe_stdio_server_sync,
            element,
            sandbox_mode=mode,
            hidden=hidden,
            expose=expose,
            timeout=timeout,
        )
        if cacheable:
            _store_cached_verdict(key, reason)
        return reason

    reasons = await asyncio.gather(*(_judge(element) for element in servers))
    kept: list[Any] = []
    pruned: list[PrunedServer] = []
    for element, reason in zip(servers, reasons):
        if reason and isinstance(element, Mapping):
            pruned.append(PrunedServer(str(element.get("name") or ""), reason))
        else:
            kept.append(element)
    return kept, pruned
