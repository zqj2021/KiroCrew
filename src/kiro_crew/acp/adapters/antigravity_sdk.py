"""ACP v1 JSON-RPC 2.0 stdio server backed by the ``google-antigravity`` Python SDK.

Bridges Kiro Crew's per-session ``AcpClient`` transport directly onto Google's official
open-source ``google-antigravity`` SDK (``Agent`` / ``LocalAgentConfig`` / ``Conversation``):

* **Zero external archive setup**: uses the ``localharness`` binary bundled inside the
  ``google-antigravity`` Python wheel (``pip install google-antigravity``).
* **True pre-execution tool gating**: wires ``google.antigravity.hooks.policy.ask_user``
  into ``LocalAgentConfig.policies`` so privileged built-in tools (``run_command``,
  ``edit_file``, ``create_file``, ``read_url_content``, ``search_web``, etc.), out-of-workspace
  file reads, and all MCP server tools emit an ACP ``tool_call`` notification followed by
  ``session/request_permission`` *before* ``localharness`` executes a single instruction.
* **Live session tuning & 1M-context telemetry**: supports ``model``, ``thinking_level``,
  and ``mode`` updates via ``session/set_config_option`` (preserving conversation trajectory
  across reconnections via ``CREATE_OR_RESUME``) and emits ``usage_update`` notifications
  with the 1,048,576-token context window size.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.metadata
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

#: Protocol version and context capacity advertised on the ACP wire.
SDK_ACP_PROTOCOL_VERSION = 1
ANTIGRAVITY_CONTEXT_WINDOW_TOKENS = 1_048_576
ANTIGRAVITY_SDK_ADAPTER_BIN = "kirocrew-antigravity-acp"
ANTIGRAVITY_SDK_VERSION_PREFIX = "sdk-"

#: Canonical Gemini models offered in the ``model`` session select.
DEFAULT_ANTIGRAVITY_MODELS: tuple[tuple[str, str], ...] = (
    ("gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
    ("gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
)
DEFAULT_MODEL_ID = "gemini-2.5-flash"

#: Thinking levels offered on ``thinking_level``.
THINKING_OPTIONS: tuple[tuple[str, str], ...] = (
    ("low", "Low"),
    ("medium", "Medium"),
    ("high", "High"),
)
DEFAULT_THINKING_LEVEL = "high"

#: Session modes matching standalone Antigravity ACP so ``Routing.SESSION_CONFIG``
#: and ``ACP_BACKEND_FORBIDDEN_MODES`` work identically across both runtimes.
SESSION_MODES: tuple[tuple[str, str, str], ...] = (
    ("default", "Default", "Default permission prompt flow"),
    ("auto_edit", "Auto Edit", "Auto-approve file edit tools"),
    ("yolo", "YOLO", "Auto-approve all tools"),
)

#: SDK built-in tool name -> ACP ``kind`` classification.
_SDK_TOOL_KIND: dict[str, str] = {
    "run_command": "execute",
    "shell": "execute",
    "edit_file": "edit",
    "create_file": "edit",
    "write_file": "edit",
    "write_to_file": "edit",
    "view_file": "read",
    "read_file": "read",
    "list_directory": "read",
    "list_dir": "read",
    "find_file": "search",
    "find_by_name": "search",
    "search_directory": "search",
    "grep_search": "search",
    "read_url_content": "fetch",
    "search_web": "fetch",
}

#: Built-in tools that only inspect files/directories when confined inside the workspace.
_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "view_file",
        "read_file",
        "list_directory",
        "list_dir",
        "find_file",
        "find_by_name",
        "search_directory",
        "grep_search",
    }
)

#: Path-carrying keys inspected to ensure un-prompted read tools stay strictly inside ``cwd``.
_PATH_ARG_KEYS: tuple[str, ...] = (
    "path",
    "file_path",
    "filePath",
    "AbsolutePath",
    "absolute_path",
    "TargetFile",
    "target_file",
    "DirectoryPath",
    "directory_path",
    "SearchPath",
    "search_path",
)


def installed_sdk_version() -> str:
    """Return the installed ``google-antigravity`` package version, or ``""`` when absent."""
    try:
        return importlib.metadata.version("google-antigravity")
    except importlib.metadata.PackageNotFoundError:
        return ""


def sdk_agent_version_string() -> str:
    """The ``agentInfo.version`` reported at ACP ``initialize``."""
    ver = installed_sdk_version() or "0.0.0"
    return f"{ANTIGRAVITY_SDK_VERSION_PREFIX}{ver}"


def _kv_pairs_to_dict(raw: object) -> dict[str, str]:
    """Normalize an ACP ``[{"name": k, "value": v}]`` array or plain mapping to ``dict``."""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if isinstance(k, str) and v is not None}
    if isinstance(raw, list):
        out: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict):
                name = item.get("name")
                value = item.get("value")
                if isinstance(name, str) and name and isinstance(value, str):
                    out[name] = value
        return out
    return {}


def _is_strictly_inside_workspace(target: str, workspace: str) -> bool:
    """True when *target* resolves inside *workspace* and touches no hidden credential home."""
    if not target or not workspace:
        return False
    try:
        ws_real = Path(workspace).expanduser().resolve()
        candidate = Path(target).expanduser()
        if not candidate.is_absolute():
            candidate = ws_real / candidate
        cand_real = candidate.resolve()
        rel = cand_real.relative_to(ws_real)
    except (OSError, ValueError):
        return False
    # Dot-prefixed credential directories (e.g. .ssh, .aws, .gemini) still ask.
    protected = {".ssh", ".aws", ".gnupg", ".gemini", ".kiro", ".kube"}
    return not any(part in protected for part in rel.parts)


def _read_call_stays_in_workspace(args: dict[str, Any], workspace: str) -> bool:
    """True when every path argument of a read-only tool resolves inside *workspace*."""
    inspected = 0
    for key in _PATH_ARG_KEYS:
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            inspected += 1
            if not _is_strictly_inside_workspace(val, workspace):
                return False
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    inspected += 1
                    if not _is_strictly_inside_workspace(item, workspace):
                        return False
    return inspected > 0 or bool(workspace)


def _load_gemini_settings_fallback() -> dict[str, Any]:
    """Read optional ``settings.json`` under ``$GEMINI_HOME/antigravity-acp`` if present."""
    gemini_home = os.environ.get("GEMINI_HOME") or os.path.expanduser("~/.gemini")
    settings_path = Path(gemini_home) / "antigravity-acp" / "settings.json"
    try:
        if settings_path.is_file():
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:  # noqa: BLE001
        logger.debug("Could not parse optional %s", settings_path, exc_info=True)
    return {}


def resolve_sdk_auth_config(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve API-key or Vertex AI auth settings from environment / settings.json.

    Raises ``PermissionError`` containing Antigravity's ``signed_out_signature`` phrase
    when neither an API key nor a Vertex AI project is configured.
    """
    environ = env if env is not None else dict(os.environ)
    api_key = environ.get("GEMINI_API_KEY") or environ.get("GOOGLE_API_KEY") or ""
    use_vertex_raw = environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    use_vertex = use_vertex_raw in ("1", "true", "yes")
    project = environ.get("GOOGLE_CLOUD_PROJECT") or environ.get("CLOUD_ML_PROJECT_ID") or ""
    location = (
        environ.get("GOOGLE_CLOUD_LOCATION") or environ.get("CLOUD_ML_REGION") or "us-central1"
    )

    if not api_key and not (use_vertex and project):
        settings = _load_gemini_settings_fallback()
        raw_gcp = settings.get("gcp")
        gcp: dict[str, Any] = raw_gcp if isinstance(raw_gcp, dict) else {}
        raw_auth = settings.get("auth")
        auth_cfg: dict[str, Any] = raw_auth if isinstance(raw_auth, dict) else {}
        proj_val = gcp.get("project")
        if not project and isinstance(proj_val, str):
            project = proj_val.strip()
        loc_val = gcp.get("location")
        if isinstance(loc_val, str) and loc_val.strip():
            location = loc_val.strip()
        if auth_cfg.get("type") == "vertex-ai" and project:
            use_vertex = True

    if use_vertex and (project or api_key):
        return {
            "vertex": True,
            "project": project or None,
            "location": location or "us-central1",
            "api_key": api_key or None,
        }
    if api_key:
        return {
            "vertex": False,
            "project": None,
            "location": None,
            "api_key": api_key,
        }

    raise PermissionError(
        "No authentication method selected: set GEMINI_API_KEY in the environment "
        "(or configure GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT for Vertex AI)."
    )


@dataclass
class _SdkSessionState:
    """Per-session mutable state held by ``AntigravitySdkAcpServer``."""

    session_id: str
    cwd: str
    mcp_servers_raw: list[dict[str, Any]] = field(default_factory=list)
    model_id: str = DEFAULT_MODEL_ID
    thinking_level: str = DEFAULT_THINKING_LEVEL
    mode: str = "default"
    dirty_config: bool = True
    strategy_cm: Any = None
    conversation: Any = None
    call_seq: int = 0
    gated_step_call_ids: dict[str, str] = field(default_factory=dict)
    ungated_step_call_ids: dict[str, str] = field(default_factory=dict)

    def next_call_id(self) -> str:
        self.call_seq += 1
        return f"call-{self.call_seq}"

    async def close_runtime(self) -> None:
        cm = self.strategy_cm
        self.strategy_cm = None
        self.conversation = None
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Error closing Antigravity SDK session %s", self.session_id, exc_info=True
                )


StrategyFactory = Callable[
    [_SdkSessionState, Callable[[Any, str], Awaitable[bool]]],
    Any,
]


class AntigravitySdkAcpServer:
    """Async newline-delimited JSON-RPC 2.0 ACP server over ``stdin`` / ``stdout``."""

    def __init__(
        self,
        *,
        send_line: Callable[[str], Awaitable[None]] | None = None,
        strategy_factory: StrategyFactory | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._send_line = send_line or self._default_stdout_write
        self._strategy_factory = strategy_factory or self._build_sdk_conversation_cm
        self._env = env
        self._write_lock = asyncio.Lock()
        self._next_rpc_id = 1
        self._pending_client_rpcs: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self._sessions: dict[str, _SdkSessionState] = {}
        self._in_flight_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    async def _default_stdout_write(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    async def _write_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        async with self._write_lock:
            await self._send_line(encoded)

    async def _send_result(self, req_id: Any, result: dict[str, Any]) -> None:
        await self._write_json({"jsonrpc": "2.0", "id": req_id, "result": result})

    async def _send_error(self, req_id: Any, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        await self._write_json({"jsonrpc": "2.0", "id": req_id, "error": err})

    async def _send_session_update(self, session_id: str, update: dict[str, Any]) -> None:
        await self._write_json(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    async def _call_client(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = f"srv-{self._next_rpc_id}"
        self._next_rpc_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_client_rpcs[req_id] = fut
        try:
            await self._write_json(
                {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
            )
            return await fut
        finally:
            self._pending_client_rpcs.pop(req_id, None)

    # ── Session config option envelopes ──

    @staticmethod
    def _session_envelope(session: _SdkSessionState, *, include_id: bool) -> dict[str, Any]:
        model_options = [{"value": mid, "name": mname} for mid, mname in DEFAULT_ANTIGRAVITY_MODELS]
        if all(opt["value"] != session.model_id for opt in model_options):
            model_options.insert(0, {"value": session.model_id, "name": session.model_id})

        modes_block = {
            "currentModeId": session.mode,
            "availableModes": [
                {"id": mid, "name": mname, "description": mdesc}
                for mid, mname, mdesc in SESSION_MODES
            ],
        }
        config_options = [
            {
                "id": "model",
                "name": "Model",
                "category": "model",
                "type": "select",
                "currentValue": session.model_id,
                "options": model_options,
            },
            {
                "id": "thinking_level",
                "name": "Thinking Level",
                "category": "thought_level",
                "type": "select",
                "currentValue": session.thinking_level,
                "options": [{"value": tid, "name": tname} for tid, tname in THINKING_OPTIONS],
            },
            {
                "id": "mode",
                "name": "Session Mode",
                "category": "mode",
                "type": "select",
                "currentValue": session.mode,
                "options": [
                    {"value": mid, "name": mname, "description": mdesc}
                    for mid, mname, mdesc in SESSION_MODES
                ],
            },
        ]
        models_block = {
            "availableModels": [
                {
                    "modelId": opt["value"],
                    "name": opt["name"],
                    "description": f"{opt['name']} via Google Antigravity SDK",
                }
                for opt in model_options
            ],
            "currentModelId": session.model_id,
        }
        out: dict[str, Any] = {
            "modes": modes_block,
            "configOptions": config_options,
            "models": models_block,
        }
        if include_id:
            out["sessionId"] = session.session_id
        return out

    # ── SDK Strategy Construction ──

    def _build_sdk_conversation_cm(
        self,
        session: _SdkSessionState,
        approval_handler: Callable[[Any, str], Awaitable[bool]],
    ) -> Any:
        from google.antigravity import Agent, LocalAgentConfig, types
        from google.antigravity.hooks import policy

        auth = resolve_sdk_auth_config(self._env)

        thinking_map = {
            "minimal": types.ThinkingLevel.MINIMAL,
            "low": types.ThinkingLevel.LOW,
            "medium": types.ThinkingLevel.MEDIUM,
            "high": types.ThinkingLevel.HIGH,
            "extra_high": types.ThinkingLevel.EXTRA_HIGH,
        }
        thinking_enum = thinking_map.get(session.thinking_level, types.ThinkingLevel.HIGH)
        model_opts = types.GeminiModelOptions(thinking_level=thinking_enum)

        if auth["vertex"]:
            endpoint: Any = types.VertexEndpoint(
                project=auth["project"],
                location=auth["location"],
                api_key=auth["api_key"],
                options=model_opts,
            )
        else:
            endpoint = types.GeminiAPIEndpoint(
                api_key=auth["api_key"],
                options=model_opts,
            )
        model_target = types.ModelTarget(name=session.model_id, endpoint=endpoint)

        mcp_configs: list[Any] = []
        for entry in session.mcp_servers_raw:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            transport_type = str(entry.get("type") or "").lower()
            url = entry.get("url")
            command = entry.get("command")
            if (
                isinstance(url, str)
                and url
                and transport_type in ("http", "sse", "streamable_http", "")
            ):
                mcp_configs.append(
                    types.McpStreamableHttpServer(
                        name=name,
                        url=url,
                        headers=_kv_pairs_to_dict(entry.get("headers")),
                    )
                )
            elif isinstance(command, str) and command:
                raw_args = entry.get("args")
                args_list = [str(a) for a in raw_args] if isinstance(raw_args, list) else []
                mcp_configs.append(
                    types.McpStdioServer(
                        name=name,
                        command=command,
                        args=args_list,
                        env=_kv_pairs_to_dict(entry.get("env")),
                    )
                )

        # Build pre-execution policies:
        # - Safe workspace-confined read tools run directly; out-of-workspace reads ask.
        # - Every mutating, shell, network, subagent, and MCP tool MUST ask Kiro Crew
        #   before execution when session.mode == "default".
        rules: list[Any] = []
        if session.mode == "yolo":
            rules.append(policy.allow("*"))
        else:
            for read_tool in sorted(_READ_ONLY_TOOLS):
                rules.append(
                    policy.allow(
                        read_tool,
                        when=lambda args, ws=session.cwd: _read_call_stays_in_workspace(
                            args if isinstance(args, dict) else {}, ws
                        ),
                    )
                )
            if session.mode == "auto_edit":
                rules.append(policy.allow("edit_file"))
                rules.append(policy.allow("create_file"))
            for mcp_cfg in mcp_configs:
                rules.extend(policy.ask_user(mcp_cfg, handler=approval_handler))
            rules.append(policy.ask_user("*", handler=approval_handler))

        gemini_home = (self._env or os.environ).get("GEMINI_HOME") or os.path.expanduser(
            "~/.gemini"
        )
        save_dir = str(Path(gemini_home) / "antigravity-sdk-acp" / "trajectories")
        app_data_dir = str(Path(gemini_home) / "antigravity-sdk-acp" / "appdata")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(app_data_dir, exist_ok=True)

        conv_id = (
            session.session_id
            if len(session.session_id) >= 32
            else str(uuid.uuid5(uuid.NAMESPACE_URL, f"kirocrew-antigravity:{session.session_id}"))
        )
        cfg = LocalAgentConfig(
            model=model_target,
            workspaces=[session.cwd] if session.cwd else None,
            mcp_servers=mcp_configs or None,
            policies=rules,
            conversation_id=conv_id,
            session_continuation_mode=types.SessionContinuationMode.CREATE_OR_RESUME,
            save_dir=save_dir,
            app_data_dir=app_data_dir,
            env=dict(self._env) if self._env is not None else None,
        )
        agent = Agent(config=cfg)

        @contextlib.asynccontextmanager
        async def _agent_conversation_cm() -> Any:
            async with agent as started:
                yield started.conversation

        return _agent_conversation_cm()

    async def _ensure_conversation(self, session: _SdkSessionState) -> Any:
        if session.conversation is not None and not session.dirty_config:
            return session.conversation
        await session.close_runtime()

        async def _on_tool_ask(tc: Any, _reason: str | None = None) -> bool:
            return await self._request_tool_permission(session, tc)

        cm = self._strategy_factory(session, _on_tool_ask)
        conv = await cm.__aenter__()
        session.strategy_cm = cm
        session.conversation = conv
        session.dirty_config = False
        return conv

    # ── Pre-execution ACP Permission Bridge ──

    @staticmethod
    def _format_tool_wire_fields(tc: Any) -> tuple[str, str, dict[str, Any], dict[str, Any] | None]:
        """Return ``(kind, title, raw_input, meta)`` matching Antigravity ACP frames."""
        tool_name = str(getattr(tc, "name", "") or "")
        server_name = str(getattr(tc, "server_name", "") or "")
        raw_args = getattr(tc, "args", None)
        raw_input: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}

        if server_name:
            meta: dict[str, Any] | None = {
                "mcp": {"server": server_name, "tool": tool_name},
                "is_mcp_tool_call": True,
            }
            return "other", tool_name or "mcp_tool", raw_input, meta

        kind = _SDK_TOOL_KIND.get(tool_name, "other")
        if kind == "execute":
            cmd = (
                raw_input.get("CommandLine")
                or raw_input.get("command_line")
                or raw_input.get("command")
                or ""
            )
            if isinstance(cmd, str) and cmd:
                raw_input.setdefault("CommandLine", cmd)
                raw_input.setdefault("command_line", cmd)
                title = cmd
            else:
                title = "run_command"
        else:
            # Matches tool_call_identity.harness_tool_name(..., backend="antigravity"),
            # which unwraps ``Run <name>?`` on native non-shell tools.
            title = f"Run {tool_name}?" if tool_name else "tool"
        return kind, title, raw_input, None

    async def _request_tool_permission(self, session: _SdkSessionState, tc: Any) -> bool:
        call_id = session.next_call_id()
        step_id = str(getattr(tc, "step_id", "") or getattr(tc, "id", "") or "")
        if step_id:
            session.gated_step_call_ids[step_id] = call_id
        else:
            # Fallback FIFO correlation by tool name when policy_decision_request omits step_id.
            fifo_key = (
                f"__pending__:{getattr(tc, 'server_name', '') or ''}:{getattr(tc, 'name', '')}"
            )
            session.gated_step_call_ids[fifo_key] = call_id

        kind, title, raw_input, meta = self._format_tool_wire_fields(tc)
        tool_update: dict[str, Any] = {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": title,
            "kind": kind,
            "status": "pending",
            "rawInput": raw_input,
        }
        if meta is not None:
            tool_update["_meta"] = meta
        await self._send_session_update(session.session_id, tool_update)

        options: list[dict[str, str]] = [
            {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
            {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
        ]
        if meta is not None:
            options.insert(
                0, {"optionId": "allow_always", "name": "Allow Always", "kind": "allow_always"}
            )

        resp = await self._call_client(
            "session/request_permission",
            {
                "sessionId": session.session_id,
                "toolCall": {
                    "toolCallId": call_id,
                    "title": title,
                    "kind": kind,
                    "status": "pending",
                    "rawInput": raw_input,
                },
                "options": options,
            },
        )
        raw_res = resp.get("result")
        result: dict[str, Any] = raw_res if isinstance(raw_res, dict) else {}
        raw_out = result.get("outcome")
        outcome: dict[str, Any] = raw_out if isinstance(raw_out, dict) else {}
        opt_id = outcome.get("optionId")
        approved = (
            outcome.get("outcome") == "selected"
            and isinstance(opt_id, str)
            and opt_id in ("allow", "allow_always")
        )
        if approved:
            await self._send_session_update(
                session.session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "in_progress",
                },
            )
            return True

        await self._send_session_update(
            session.session_id,
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "failed",
                "rawOutput": "Permission denied by host policy.",
            },
        )
        return False

    # ── Prompt Conversion & Step Streaming ──

    @staticmethod
    def _extract_prompt_parts(blocks: object) -> tuple[list[Any], str]:
        """Convert ACP prompt blocks into SDK ``Content`` parts + combined plain text."""
        if isinstance(blocks, str):
            return [blocks], blocks
        if not isinstance(blocks, list):
            return [""], ""

        parts: list[Any] = []
        texts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
                parts.append(block["text"])
            elif btype == "image" and isinstance(block.get("data"), str):
                mime = str(block.get("mimeType") or "image/png")
                try:
                    from google.antigravity import types

                    raw_bytes = base64.b64decode(block["data"])
                    parts.append(types.Image(data=raw_bytes, mime_type=mime))
                except Exception:  # noqa: BLE001
                    texts.append("[image decode failed]")
            elif btype == "resource" and isinstance(block.get("resource"), dict):
                res = block["resource"]
                if isinstance(res.get("text"), str):
                    uri = res.get("uri", "")
                    chunk = f"\n<resource uri={uri!r}>\n{res['text']}\n</resource>\n"
                    texts.append(chunk)
                    parts.append(chunk)

        if not parts:
            parts.append("")
        return parts, "\n".join(texts).strip()

    async def _emit_usage_if_present(self, session: _SdkSessionState, conv: Any) -> None:
        conn = getattr(conv, "connection", None)
        usage = getattr(conn, "cumulative_usage", None)
        if usage is None:
            return
        used = getattr(usage, "total_token_count", None)
        if used is None:
            prompt_toks = getattr(usage, "prompt_token_count", 0) or 0
            cand_toks = getattr(usage, "candidates_token_count", 0) or 0
            thought_toks = getattr(usage, "thoughts_token_count", 0) or 0
            used = prompt_toks + cand_toks + thought_toks
        await self._send_session_update(
            session.session_id,
            {
                "sessionUpdate": "usage_update",
                "used": int(used or 0),
                "size": ANTIGRAVITY_CONTEXT_WINDOW_TOKENS,
            },
        )

    async def _handle_slash_usage(self, req_id: Any, session: _SdkSessionState) -> None:
        used = 0
        if session.conversation is not None:
            conn = getattr(session.conversation, "connection", None)
            usage = getattr(conn, "cumulative_usage", None)
            used = int(getattr(usage, "total_token_count", 0) or 0)
        pct = (used / ANTIGRAVITY_CONTEXT_WINDOW_TOKENS) * 100.0
        summary = (
            f"Google Antigravity SDK context usage: {used:,} / "
            f"{ANTIGRAVITY_CONTEXT_WINDOW_TOKENS:,} tokens ({pct:.2f}%) "
            f"[model={session.model_id}, thinking={session.thinking_level}]"
        )
        await self._send_session_update(
            session.session_id,
            {
                "sessionUpdate": "usage_update",
                "used": used,
                "size": ANTIGRAVITY_CONTEXT_WINDOW_TOKENS,
            },
        )
        await self._send_session_update(
            session.session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": summary},
            },
        )
        await self._send_result(req_id, {"stopReason": "end_turn"})

    # ── RPC Method Handlers ──

    async def _on_initialize(self, req_id: Any, _params: dict[str, Any]) -> None:
        await self._send_result(
            req_id,
            {
                "protocolVersion": SDK_ACP_PROTOCOL_VERSION,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {
                        "image": True,
                        "audio": True,
                        "embeddedContext": True,
                    },
                    "mcpCapabilities": {"http": True, "sse": True},
                    "sessionCapabilities": {
                        "list": {},
                        "additionalDirectories": {},
                        "resume": {},
                    },
                },
                "authMethods": [
                    {
                        "id": "gemini-api-key",
                        "name": "Gemini API key",
                        "description": "Set GEMINI_API_KEY in the gateway or agent environment",
                    },
                    {
                        "id": "vertex-ai",
                        "name": "Vertex AI",
                        "description": "Set GOOGLE_GENAI_USE_VERTEXAI=true and GOOGLE_CLOUD_PROJECT",
                    },
                ],
                "agentInfo": {
                    "name": "antigravity-acp",
                    "title": "Google Antigravity (Python SDK)",
                    "version": sdk_agent_version_string(),
                },
            },
        )

    async def _on_session_new(self, req_id: Any, params: dict[str, Any]) -> None:
        try:
            resolve_sdk_auth_config(self._env)
        except PermissionError as exc:
            await self._send_error(req_id, -32000, str(exc))
            return

        cwd = str(params.get("cwd") or os.getcwd())
        mcp_servers = params.get("mcpServers")
        mcp_list = list(mcp_servers) if isinstance(mcp_servers, list) else []
        session_id = f"antigravity-sdk-{uuid.uuid4()}"
        session = _SdkSessionState(
            session_id=session_id,
            cwd=cwd,
            mcp_servers_raw=mcp_list,
        )
        self._sessions[session_id] = session
        await self._send_result(req_id, self._session_envelope(session, include_id=True))

    async def _on_session_load(self, req_id: Any, params: dict[str, Any]) -> None:
        try:
            resolve_sdk_auth_config(self._env)
        except PermissionError as exc:
            await self._send_error(req_id, -32000, str(exc))
            return

        session_id = str(params.get("sessionId") or "")
        if not session_id:
            await self._send_error(req_id, -32602, "sessionId is required")
            return
        cwd = str(params.get("cwd") or os.getcwd())
        mcp_servers = params.get("mcpServers")
        mcp_list = list(mcp_servers) if isinstance(mcp_servers, list) else []
        session = self._sessions.get(session_id)
        if session is None:
            session = _SdkSessionState(
                session_id=session_id,
                cwd=cwd,
                mcp_servers_raw=mcp_list,
            )
            self._sessions[session_id] = session
        else:
            session.cwd = cwd
            if mcp_list:
                session.mcp_servers_raw = mcp_list
                session.dirty_config = True
        await self._send_result(req_id, self._session_envelope(session, include_id=False))

    async def _on_set_config_option(self, req_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        session = self._sessions.get(session_id)
        if session is None:
            await self._send_error(req_id, -32602, f"Session not found: {session_id}")
            return

        config_id = str(params.get("configId") or params.get("id") or "")
        value = str(params.get("value") or "")
        if config_id == "mode":
            valid_modes = {m[0] for m in SESSION_MODES}
            if value not in valid_modes:
                await self._send_error(req_id, -32602, f"Unsupported mode: {value!r}")
                return
            if session.mode != value:
                session.mode = value
                session.dirty_config = True
                await self._send_session_update(
                    session.session_id,
                    {"sessionUpdate": "current_mode_update", "currentModeId": value},
                )
        elif config_id == "model":
            if not value:
                await self._send_error(req_id, -32602, "Model value cannot be empty")
                return
            if session.model_id != value:
                session.model_id = value
                session.dirty_config = True
        elif config_id == "thinking_level":
            valid_thinking = {t[0] for t in THINKING_OPTIONS} | {"minimal", "extra_high"}
            if value not in valid_thinking:
                await self._send_error(req_id, -32602, f"Unsupported thinking_level: {value!r}")
                return
            if session.thinking_level != value:
                session.thinking_level = value
                session.dirty_config = True
        else:
            await self._send_error(req_id, -32602, f"Unknown config option: {config_id!r}")
            return

        envelope = self._session_envelope(session, include_id=False)
        await self._send_result(req_id, {"configOptions": envelope["configOptions"]})

    async def _on_set_mode(self, req_id: Any, params: dict[str, Any]) -> None:
        mode_id = str(params.get("modeId") or "")
        await self._on_set_config_option(
            req_id,
            {"sessionId": params.get("sessionId"), "configId": "mode", "value": mode_id},
        )

    async def _on_session_prompt(self, req_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        session = self._sessions.get(session_id)
        if session is None:
            await self._send_error(req_id, -32602, f"Session not found: {session_id}")
            return

        parts, plain_text = self._extract_prompt_parts(params.get("prompt"))
        if plain_text.strip() == "/usage":
            await self._handle_slash_usage(req_id, session)
            return

        try:
            conv = await self._ensure_conversation(session)
        except PermissionError as exc:
            await self._send_error(req_id, -32000, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            await self._send_error(
                req_id, -32603, f"Failed to start Antigravity SDK session: {exc}"
            )
            return

        await self._emit_usage_if_present(session, conv)

        prompt_payload: Any = parts[0] if len(parts) == 1 and isinstance(parts[0], str) else parts
        try:
            await conv.send(prompt_payload)
            async for step in conv.receive_steps():
                source_val = getattr(
                    getattr(step, "source", None), "value", getattr(step, "source", "")
                )
                target_val = getattr(
                    getattr(step, "target", None), "value", getattr(step, "target", "")
                )
                status_val = getattr(
                    getattr(step, "status", None), "value", getattr(step, "status", "")
                )

                is_model = str(source_val).upper().endswith("MODEL")
                is_user_target = str(target_val).upper().endswith("USER") or not target_val
                is_done = str(status_val).upper().endswith("DONE")
                is_error = str(status_val).upper().endswith("ERROR")

                if is_model and is_user_target and not is_error:
                    thought_delta = getattr(step, "thinking_delta", "") or ""
                    if thought_delta:
                        await self._send_session_update(
                            session.session_id,
                            {
                                "sessionUpdate": "agent_thought_chunk",
                                "content": {"type": "text", "text": str(thought_delta)},
                            },
                        )
                    text_delta = getattr(step, "content_delta", "") or ""
                    if text_delta:
                        await self._send_session_update(
                            session.session_id,
                            {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": str(text_delta)},
                            },
                        )

                tool_calls = getattr(step, "tool_calls", None) or []
                for tc in tool_calls:
                    step_key = str(
                        getattr(tc, "step_id", "")
                        or getattr(tc, "id", "")
                        or getattr(step, "id", "")
                    )
                    fifo_key = f"__pending__:{getattr(tc, 'server_name', '') or ''}:{getattr(tc, 'name', '')}"
                    gated_id = session.gated_step_call_ids.get(
                        step_key
                    ) or session.gated_step_call_ids.get(fifo_key)

                    if gated_id:
                        if is_done or is_error:
                            session.gated_step_call_ids.pop(step_key, None)
                            session.gated_step_call_ids.pop(fifo_key, None)
                            raw_out = (
                                getattr(step, "error", "")
                                if is_error
                                else (getattr(step, "content", "") or "ok")
                            )
                            await self._send_session_update(
                                session.session_id,
                                {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": gated_id,
                                    "status": "failed" if is_error else "completed",
                                    "rawOutput": raw_out,
                                },
                            )
                    else:
                        ungated_id = session.ungated_step_call_ids.get(step_key)
                        if ungated_id is None:
                            ungated_id = session.next_call_id()
                            if step_key:
                                session.ungated_step_call_ids[step_key] = ungated_id
                            kind, title, raw_input, meta = self._format_tool_wire_fields(tc)
                            update_frame: dict[str, Any] = {
                                "sessionUpdate": "tool_call",
                                "toolCallId": ungated_id,
                                "title": title,
                                "kind": kind,
                                "status": "pending",
                                "rawInput": raw_input,
                            }
                            if meta is not None:
                                update_frame["_meta"] = meta
                            await self._send_session_update(session.session_id, update_frame)
                        if is_done or is_error:
                            session.ungated_step_call_ids.pop(step_key, None)
                            await self._send_session_update(
                                session.session_id,
                                {
                                    "sessionUpdate": "tool_call_update",
                                    "toolCallId": ungated_id,
                                    "status": "failed" if is_error else "completed",
                                    "rawOutput": (
                                        getattr(step, "error", "")
                                        if is_error
                                        else (getattr(step, "content", "") or "ok")
                                    ),
                                },
                            )

            await self._emit_usage_if_present(session, conv)
            await self._send_result(req_id, {"stopReason": "end_turn"})
        except Exception as exc:  # noqa: BLE001
            exc_name = type(exc).__name__
            if "Cancelled" in exc_name or "cancelled" in str(exc).lower():
                await self._send_result(req_id, {"stopReason": "cancelled"})
                return
            await self._send_error(req_id, -32603, str(exc))

    async def _on_session_cancel(self, req_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        session = self._sessions.get(session_id)
        if session is not None and session.conversation is not None:
            conn = getattr(session.conversation, "connection", None)
            cancel_fn = getattr(conn, "cancel", None)
            if callable(cancel_fn):
                try:
                    await cancel_fn()
                except Exception:  # noqa: BLE001
                    logger.debug("Cancel failed for %s", session_id, exc_info=True)
        if req_id is not None:
            await self._send_result(req_id, {})

    async def _on_session_close(self, req_id: Any, params: dict[str, Any]) -> None:
        session_id = str(params.get("sessionId") or "")
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.close_runtime()
        if req_id is not None:
            await self._send_result(req_id, {})

    # ── Message Dispatch ──

    async def handle_raw_line(self, line: str) -> None:
        """Parse and dispatch one JSON-RPC line without blocking concurrent responses."""
        stripped = line.strip()
        if not stripped:
            return
        try:
            msg = json.loads(stripped)
        except json.JSONDecodeError:
            await self._send_error(None, -32700, "Parse error")
            return
        if not isinstance(msg, dict):
            await self._send_error(None, -32600, "Invalid Request")
            return

        method = msg.get("method")
        msg_id = msg.get("id")

        # Client response to a server-initiated request (e.g. session/request_permission).
        if method is None and msg_id in self._pending_client_rpcs:
            fut = self._pending_client_rpcs.get(msg_id)
            if fut is not None and not fut.done():
                fut.set_result(msg)
            return

        if not isinstance(method, str):
            if msg_id is not None:
                await self._send_error(msg_id, -32600, "Invalid Request")
            return

        raw_params = msg.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        task = asyncio.create_task(self._dispatch_request(msg_id, method, params))
        self._in_flight_tasks.add(task)
        task.add_done_callback(self._in_flight_tasks.discard)

    async def _dispatch_request(self, req_id: Any, method: str, params: dict[str, Any]) -> None:
        handlers: dict[str, Callable[[Any, dict[str, Any]], Awaitable[None]]] = {
            "initialize": self._on_initialize,
            "session/new": self._on_session_new,
            "session/load": self._on_session_load,
            "session/set_config_option": self._on_set_config_option,
            "session/set_mode": self._on_set_mode,
            "session/prompt": self._on_session_prompt,
            "session/cancel": self._on_session_cancel,
            "session/close": self._on_session_close,
        }
        handler = handlers.get(method)
        if handler is None:
            if req_id is not None:
                await self._send_error(req_id, -32601, f"Method not found: {method}")
            return
        try:
            await handler(req_id, params)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled error in ACP method %s", method)
            if req_id is not None:
                await self._send_error(req_id, -32603, f"Internal error: {exc}")

    async def close_all(self) -> None:
        tasks = list(self._in_flight_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for session in list(self._sessions.values()):
            await session.close_runtime()
        self._sessions.clear()


async def serve_stdio() -> None:
    """Run ``AntigravitySdkAcpServer`` over process ``stdin`` / ``stdout`` until EOF."""
    server = AntigravitySdkAcpServer()
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    try:
        while True:
            raw = await reader.readline()
            if not raw:
                break
            await server.handle_raw_line(raw.decode("utf-8", errors="replace"))
        if server._in_flight_tasks:
            await asyncio.gather(*list(server._in_flight_tasks), return_exceptions=True)
    finally:
        await server.close_all()


def main() -> None:
    """Console entry point for ``kirocrew-antigravity-acp``."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        asyncio.run(serve_stdio())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
