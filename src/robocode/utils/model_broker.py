"""Host-owned inference broker for network-disconnected Apptainer agents.

Only this process holds provider credentials. The Unix socket exposes a small
HTTP API, not CONNECT or an arbitrary destination proxy. Request bodies are
validated before forwarding to fixed HTTPS endpoints; redirects are never followed.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import json
import os
import socketserver
import ssl
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from robocode.utils.backends import (
    ANTHROPIC_API_HOST,
    CODEX_CHATGPT_HOST,
    OPENAI_API_HOST,
)
from robocode.utils.claude_auth import host_claude_config_dir
from robocode.utils.codex_auth import host_codex_home

MAX_BODY = 32 * 1024 * 1024
MODEL_PORT = 18080
BROKER_DIR = "/run/robocode-broker"


class BrokerPolicyError(ValueError):
    """A request is outside the explicitly supported inference protocol."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BrokerPolicyError("duplicate JSON key")
        result[key] = value
    return result


def _local_content(value: Any) -> None:
    """Reject provider-side URL/file retrieval in actual content (not prose)."""
    if isinstance(value, list):
        for child in value:
            _local_content(child)
    elif isinstance(value, dict):
        kind = value.get("type", "")
        if isinstance(kind, str) and (
            kind.startswith(
                (
                    "web_",
                    "mcp_",
                    "server_",
                    "computer_",
                    "code_interpreter",
                    "file_search",
                )
            )
            or kind
            in {
                "tool_search_call",
                "tool_search_output",
                "item_reference",
                "input_file",
                "document",
            }
        ):
            raise BrokerPolicyError("server-side content operation")
        for key, child in value.items():
            if key in {"url", "image_url"}:
                if not isinstance(child, str):
                    raise BrokerPolicyError("remote content URL")
                prefix, sep, encoded = child.partition(",")
                if not sep or prefix not in {
                    "data:image/png;base64",
                    "data:image/jpeg;base64",
                    "data:image/webp;base64",
                    "data:image/gif;base64",
                }:
                    raise BrokerPolicyError("only inline raster images are supported")
                try:
                    base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise BrokerPolicyError("invalid inline image") from exc
            if key in {"file_url", "file_id", "container_id", "server_url"}:
                raise BrokerPolicyError("remote content reference")
            if key == "source" and isinstance(child, dict):
                if child.get("type") not in {"base64", "text"}:
                    raise BrokerPolicyError("remote content source")
                if child.get("type") == "base64" and child.get("media_type") not in {
                    "image/png",
                    "image/jpeg",
                    "image/webp",
                    "image/gif",
                }:
                    raise BrokerPolicyError("only inline raster images are supported")
            _local_content(child)


def _schemas(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref" and (
                not isinstance(child, str) or not child.startswith("#")
            ):
                raise BrokerPolicyError("remote schema reference")
            _schemas(child)
    elif isinstance(value, list):
        for child in value:
            _schemas(child)


def _openai_tools(tools: Any) -> None:
    if not isinstance(tools, list):
        raise BrokerPolicyError("tools must be a list")
    for tool in tools:
        if not isinstance(tool, dict):
            raise BrokerPolicyError("invalid tool")
        kind = tool.get("type")
        if kind == "namespace":
            if set(tool) - {"type", "name", "description", "tools"}:
                raise BrokerPolicyError("unsupported namespace fields")
            _openai_tools(tool.get("tools"))
        elif kind == "function":
            if set(tool) - {
                "type",
                "name",
                "description",
                "parameters",
                "strict",
                "defer_loading",
            }:
                raise BrokerPolicyError("unsupported function fields")
            _schemas(tool)
        elif kind == "custom":
            if set(tool) - {"type", "name", "description", "format", "defer_loading"}:
                raise BrokerPolicyError("unsupported custom tool fields")
        else:
            raise BrokerPolicyError("server-side tools are forbidden")


def validate_request(protocol: str, path: str, raw: bytes) -> dict[str, Any]:
    """Parse a bounded request and fail closed on unsupported API operations."""
    if len(raw) > MAX_BODY:
        raise BrokerPolicyError("body too large")
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicates)
    except (ValueError, RecursionError) as exc:
        raise BrokerPolicyError("invalid JSON") from exc
    if not isinstance(data, dict):
        raise BrokerPolicyError("body must be an object")
    if protocol == "responses":
        if path not in {"/v1/responses", "/v1/responses/compact"}:
            raise BrokerPolicyError("endpoint forbidden")
        allowed = {
            "model",
            "instructions",
            "input",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "stream",
            "store",
            "reasoning",
            "text",
            "include",
            "prompt_cache_key",
            "service_tier",
            "max_output_tokens",
            "temperature",
            "top_p",
            "metadata",
            "truncation",
            "prompt_cache_retention",
            "safety_identifier",
            "client_metadata",
        }
        if set(data) - allowed:
            raise BrokerPolicyError(
                "unsupported fields: " + ",".join(sorted(set(data) - allowed))
            )
        data.pop("client_metadata", None)  # do not grant authority via client hints
        _openai_tools(data.get("tools", []))
        choice = data.get("tool_choice", "auto")
        if not (
            choice in ("auto", "none", "required")
            if isinstance(choice, str)
            else isinstance(choice, dict)
            and choice.get("type") in {"function", "custom"}
        ):
            raise BrokerPolicyError("unsupported tool choice")
        if any(
            item != "reasoning.encrypted_content" for item in data.get("include", [])
        ):
            raise BrokerPolicyError("unsupported include")
        _local_content(data.get("input"))
        _schemas(data.get("text"))
    elif protocol == "messages":
        if path not in {
            "/v1/messages",
            "/v1/messages?beta=true",
            "/v1/messages/count_tokens",
            "/v1/messages/count_tokens?beta=true",
        }:
            raise BrokerPolicyError("endpoint forbidden")
        allowed = {
            "model",
            "messages",
            "system",
            "tools",
            "tool_choice",
            "max_tokens",
            "stream",
            "temperature",
            "top_p",
            "top_k",
            "thinking",
            "output_config",
            "metadata",
            "stop_sequences",
            "service_tier",
            "context_management",
        }
        if set(data) - allowed:
            raise BrokerPolicyError(
                "unsupported fields: " + ",".join(sorted(set(data) - allowed))
            )
        tools = data.get("tools", [])
        if not isinstance(tools, list):
            raise BrokerPolicyError("tools must be a list")
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type", "custom") != "custom":
                raise BrokerPolicyError("server-side tools are forbidden")
            if set(tool) - {
                "type",
                "name",
                "description",
                "input_schema",
                "cache_control",
                "defer_loading",
                "strict",
                "input_examples",
            }:
                raise BrokerPolicyError("unsupported custom tool fields")
            _schemas(tool)
        _local_content(data.get("messages"))
        _local_content(data.get("system"))
        context = data.get("context_management", {})
        if not isinstance(context, dict) or set(context) - {"edits"}:
            raise BrokerPolicyError("unsupported context management")
        for edit in context.get("edits", []):
            if not isinstance(edit, dict) or edit.get("type") not in {
                "clear_thinking_20251015",
                "clear_tool_uses_20250919",
            }:
                raise BrokerPolicyError("unsupported context operation")
    else:
        raise BrokerPolicyError("unsupported protocol")
    if not isinstance(data.get("model"), str) or not data["model"]:
        raise BrokerPolicyError("model required")
    return data


@dataclass(frozen=True)
class BrokerUpstream:
    """Trusted upstream selection; never populated from a container request."""

    protocol: str
    host: str
    base_path: str
    headers: dict[str, str] = field(repr=False)
    chatgpt: bool = False


def load_broker_upstream(backend: str) -> BrokerUpstream:
    """Load credentials on the host without copying them into the container."""
    if backend == "codex":
        key = os.environ.get("CODEX_API_KEY")
        if key:
            return BrokerUpstream(
                "responses", OPENAI_API_HOST, "/v1", {"Authorization": "Bearer " + key}
            )
        auth = json.loads((host_codex_home() / "auth.json").read_text(encoding="utf-8"))
        if auth.get("auth_mode") == "chatgpt":
            tokens = auth["tokens"]
            return BrokerUpstream(
                "responses",
                CODEX_CHATGPT_HOST,
                "/backend-api/codex",
                {
                    "Authorization": "Bearer " + tokens["access_token"],
                    "ChatGPT-Account-ID": tokens["account_id"],
                    "OpenAI-Beta": "responses=experimental",
                    "originator": "codex_cli_rs",
                },
                chatgpt=True,
            )
        key = auth.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if key:
            return BrokerUpstream(
                "responses", OPENAI_API_HOST, "/v1", {"Authorization": "Bearer " + key}
            )
        raise RuntimeError("No supported Codex credentials for the isolated broker")
    if backend == "claude":
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not token and not key:
            creds = json.loads(
                (host_claude_config_dir() / ".credentials.json").read_text(
                    encoding="utf-8"
                )
            )
            token = creds.get("claudeAiOauth", {}).get("accessToken")
        headers = {"anthropic-version": "2023-06-01"}
        if token:
            headers.update(
                {
                    "Authorization": "Bearer " + token,
                    "anthropic-beta": "oauth-2025-04-20,context-management-2025-06-27",
                }
            )
        elif key:
            headers["x-api-key"] = key
        else:
            raise RuntimeError("No Claude credentials for the isolated broker")
        return BrokerUpstream("messages", ANTHROPIC_API_HOST, "/v1", headers)
    raise RuntimeError(
        f"Isolated Apptainer model transport does not support {backend!r}; "
        "refusing host networking"
    )


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, path: Path, provider: BrokerUpstream, log_path: Path):
        self.provider = provider
        self.log_path = log_path
        self.log_lock = threading.Lock()
        super().__init__(str(path), _Handler)

    def record(self, path: str, status: int, reason: str) -> None:
        """Retain decisions, never credentials or request/response bodies."""
        with self.log_lock, self.log_path.open("a", encoding="utf-8") as log:
            log.write(
                json.dumps({"path": path[:200], "status": status, "reason": reason})
                + "\n"
            )


class _Handler(BaseHTTPRequestHandler):
    server: _Server
    close_connection: bool
    protocol_version = "HTTP/1.0"  # one framed request per connection

    def setup(self) -> None:
        self.request.settimeout(120)
        super().setup()

    def log_message(  # pylint: disable=redefined-builtin
        self, format: str, *args: Any
    ) -> None:
        """Suppress the standard HTTP logger; use body-free policy audit records."""

    def _reject(self, status: int, reason: str) -> None:
        self.server.record(self.path, status, reason)
        body = json.dumps(
            {
                "error": {
                    "message": "Robocode broker: " + reason,
                    "type": "broker_policy",
                }
            }
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_CONNECT(self) -> None:  # pylint: disable=invalid-name
        """Never expose a TCP tunnel."""
        self._reject(403, "CONNECT forbidden")

    def do_GET(self) -> None:  # pylint: disable=invalid-name
        """Do not expose discovery, search, retrieval, or websocket upgrades."""
        self._reject(403, "GET and websocket upgrades forbidden")

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Validate, authenticate on the host, and stream a fixed upstream."""
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if (
                len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                raise BrokerPolicyError("single Content-Length required")
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY:
                raise BrokerPolicyError("invalid body size")
            if self.headers.get("Transfer-Encoding") or self.headers.get("Upgrade"):
                raise BrokerPolicyError("transfer encoding and upgrades forbidden")
            if self.headers.get("Content-Encoding", "identity") != "identity":
                raise BrokerPolicyError("compressed requests unsupported")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise BrokerPolicyError("incomplete body")
            provider = self.server.provider
            data = validate_request(provider.protocol, self.path, raw)
            # The ChatGPT Codex endpoint only accepts streaming, unstored inference.
            if provider.protocol == "responses":
                data["store"] = False
            if provider.chatgpt:
                if self.path == "/v1/responses":
                    data["stream"] = True
            raw = json.dumps(data, allow_nan=False).encode()
        except (
            BrokerPolicyError,
            ValueError,
            RecursionError,
            TypeError,
            AttributeError,
        ) as exc:
            self._reject(403, str(exc))
            return
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **provider.headers,
        }
        conn = http.client.HTTPSConnection(
            provider.host, timeout=120, context=ssl.create_default_context()
        )
        started = False
        try:
            conn.request(
                "POST",
                provider.base_path + self.path[len("/v1") :],
                body=raw,
                headers=headers,
            )
            response = conn.getresponse()
            if 300 <= response.status < 400:
                self._reject(502, "upstream redirect forbidden")
                return
            self.server.record(self.path, response.status, "forwarded")
            self.send_response(response.status)
            self.send_header(
                "Content-Type", response.getheader("Content-Type", "application/json")
            )
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            if not started:
                self._reject(502, "upstream unavailable")
        finally:
            conn.close()
            self.close_connection = True


@contextmanager
def model_broker(
    directory: Path, provider: BrokerUpstream, log_path: Path
) -> Iterator[Path]:
    """Expose only the per-run Unix endpoint; the private log stays outside binds."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    path = directory / "model.sock"
    with _Server(path, provider, log_path) as server:
        path.chmod(0o600)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield path
        finally:
            server.shutdown()
            thread.join(timeout=5)
            path.unlink(missing_ok=True)
