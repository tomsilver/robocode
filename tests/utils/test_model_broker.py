"""Adversarial policy and HTTP framing checks for the trusted inference broker."""

# pylint: disable=redefined-outer-name

import http.client
import io
import json
import socket
from pathlib import Path
from typing import Any

import pytest

from robocode.utils.model_broker import (
    BrokerPolicyError,
    BrokerUpstream,
    load_broker_upstream,
    model_broker,
    validate_request,
)


def _body(**extra: Any) -> bytes:
    return json.dumps({"model": "test-model", "input": "hello", **extra}).encode()


@pytest.mark.parametrize(
    "tool",
    [
        "web_search",
        "web_search_preview",
        "file_search",
        "mcp",
        "code_interpreter",
        "computer_use_preview",
        "image_generation",
        "tool_search",
        "future_server_tool",
    ],
)
def test_hosted_tools_rejected(tool):
    """New or known provider-executed tools never reach the upstream API."""
    with pytest.raises(BrokerPolicyError):
        validate_request("responses", "/v1/responses", _body(tools=[{"type": tool}]))


def test_nested_namespace_cannot_hide_hosted_tool():
    """Namespaces contain only client-executed function/custom declarations."""
    tools = [{"type": "namespace", "name": "a", "tools": [{"type": "web_search"}]}]
    with pytest.raises(BrokerPolicyError):
        validate_request("responses", "/v1/responses", _body(tools=tools))


@pytest.mark.parametrize(
    "path",
    [
        "https://example.com/v1/responses",
        "//example.com/v1/responses",
        "/v1/models",
        "/v1/responses?url=https://example.com",
        "/v1/responses/../search",
        "/v1/responses%2f..%2fsearch",
        "/v1/files",
        "/v1/responses/123",
    ],
)
def test_only_exact_inference_paths_allowed(path):
    """Absolute URLs, redirects, retrieval endpoints and encoded paths fail."""
    with pytest.raises(BrokerPolicyError):
        validate_request("responses", path, _body())


@pytest.mark.parametrize(
    "content",
    [
        {"type": "input_image", "image_url": "https://example.com/image.png"},
        {"type": "input_file", "file_url": "https://example.com/f"},
        {"type": "input_file", "file_id": "file-123"},
        {"source": {"type": "url", "url": "https://example.com"}},
    ],
)
def test_remote_content_retrieval_rejected(content):
    """An inference endpoint cannot be used as a URL fetcher."""
    with pytest.raises(BrokerPolicyError):
        validate_request("responses", "/v1/responses", _body(input=[content]))


def test_inline_image_and_literal_url_text_allowed():
    """Locally supplied pixels and ordinary URL mentions are not network fetches."""
    data = _body(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "https://example.com"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                ],
            }
        ],
        tools=[{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
    )
    assert validate_request("responses", "/v1/responses", data)["model"] == "test-model"


@pytest.mark.parametrize(
    "extra",
    [
        {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
        {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
        {"mcp_servers": [{"url": "https://example.com"}]},
        {"container": {"skills": []}},
        {
            "messages": [
                {"content": [{"source": {"type": "url", "url": "https://example.com"}}]}
            ]
        },
    ],
)
def test_claude_server_capabilities_rejected(extra):
    """Only client-executed Claude tools and inline message content are allowed."""
    data = json.dumps({"model": "claude", "messages": [], **extra}).encode()
    with pytest.raises(BrokerPolicyError):
        validate_request("messages", "/v1/messages?beta=true", data)


def test_duplicate_keys_and_unknown_fields_fail_closed():
    """Neither ambiguous JSON nor future API switches can silently expand access."""
    for data in (
        b'{"model":"a","tools":[],"tools":[{"type":"web_search"}]}',
        _body(new_network_feature=True),
    ):
        with pytest.raises(BrokerPolicyError):
            validate_request("responses", "/v1/responses", data)


class _UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(self.host)


class _Response(io.BytesIO):
    status = 200

    def getheader(self, name, default=None):
        """Return the minimal streaming response headers."""
        return "text/event-stream" if name == "Content-Type" else default


@pytest.fixture
def broker(tmp_path: Path, monkeypatch):
    """A real Unix HTTP server with a fake, observable HTTPS upstream."""
    requests = []
    response = _Response(b'data: {"ok":true}\n\n')

    class Connection:
        """Record upstream operations without performing network I/O."""

        def __init__(self, host, **kwargs):
            assert host == "api.openai.com"
            assert kwargs["context"].check_hostname

        def request(self, method, path, body, headers):
            """Record exactly what would be sent to the provider."""
            requests.append((method, path, body, headers))

        def getresponse(self):
            """Return the fixture response."""
            return response

        def close(self):
            """No real upstream connection needs closing."""

    monkeypatch.setattr(
        "robocode.utils.model_broker.http.client.HTTPSConnection", Connection
    )
    provider = BrokerUpstream(
        "responses", "api.openai.com", "/v1", {"Authorization": "Bearer host-secret"}
    )
    with model_broker(tmp_path, provider, tmp_path / "audit.jsonl") as path:
        yield path, requests, response


@pytest.mark.parametrize(
    "method,path,body,headers",
    [
        ("CONNECT", "example.com:443", None, {}),
        ("GET", "/v1/responses", None, {"Upgrade": "websocket"}),
        ("POST", "/v1/responses", _body(), {"Content-Encoding": "gzip"}),
        ("POST", "/v1/responses", _body(), {"Transfer-Encoding": "chunked"}),
        ("POST", "/v1/responses", _body(tools=[{"type": "web_search"}]), {}),
    ],
)
def test_http_denials_never_open_upstream(broker, method, path, body, headers):
    """Framing, tunneling and tool bypasses are rejected before HTTPS starts."""
    address, requests, _ = broker
    conn = _UnixHTTP(str(address))
    conn.request(method, path, body=body, headers=headers)
    assert conn.getresponse().status == 403
    conn.close()
    assert not requests


def test_valid_stream_uses_fixed_host_path_and_host_credentials(broker):
    """An agent's Host, Authorization and forwarding headers carry no authority."""
    address, requests, _ = broker
    conn = _UnixHTTP(str(address))
    conn.request(
        "POST",
        "/v1/responses",
        body=_body(),
        headers={
            "Host": "evil.invalid",
            "Authorization": "Bearer attacker",
            "X-Forwarded-Host": "evil.invalid",
        },
    )
    response = conn.getresponse()
    assert response.status == 200
    assert response.read() == b'data: {"ok":true}\n\n'
    conn.close()
    assert requests[0][0:2] == ("POST", "/v1/responses")
    assert requests[0][3]["Authorization"] == "Bearer host-secret"
    assert "X-Forwarded-Host" not in requests[0][3]
    assert "host-secret" not in (address.parent / "audit.jsonl").read_text(
        encoding="utf-8"
    )


def test_upstream_redirect_is_never_followed(broker):
    """Even a redirect from the approved provider cannot change destination."""
    address, requests, upstream = broker
    upstream.status = 302
    conn = _UnixHTTP(str(address))
    conn.request("POST", "/v1/responses", body=_body())
    assert conn.getresponse().status == 502
    conn.close()
    assert len(requests) == 1


def test_duplicate_content_length_rejected(broker):
    """Conflicting framing cannot smuggle a second request to the provider."""
    address, requests, _ = broker
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(str(address))
        sock.sendall(
            b"POST /v1/responses HTTP/1.1\r\nHost: local\r\n"
            b"Content-Length: 2\r\nContent-Length: 3\r\n\r\n{}"
        )
        assert b"403" in sock.recv(4096)
    assert not requests


@pytest.mark.parametrize(
    "content",
    [
        {"type": "input_image", "image_url": "data:image/svg+xml;base64,AAAA"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "AAAA",
            },
        },
        {"type": "input_file", "file_data": "AAAA"},
        {"type": "mcp_approval_response", "approval_request_id": "x", "approve": True},
    ],
)
def test_indirect_document_and_server_operation_channels_rejected(content):
    """Opaque document formats and inherited hosted operations stay unsupported."""
    with pytest.raises(BrokerPolicyError):
        validate_request("responses", "/v1/responses", _body(input=[content]))


def test_cannot_inherit_tools_from_stored_response():
    """A caller cannot continue an unrelated stored response with hosted tools."""
    with pytest.raises(BrokerPolicyError):
        validate_request(
            "responses", "/v1/responses", _body(previous_response_id="resp_other")
        )


@pytest.mark.parametrize(
    "backend,env_name,expected_host,header,prefix",
    [
        ("codex", "CODEX_API_KEY", "api.openai.com", "Authorization", "Bearer "),
        (
            "claude",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "api.anthropic.com",
            "Authorization",
            "Bearer ",
        ),
        ("claude", "ANTHROPIC_API_KEY", "api.anthropic.com", "x-api-key", ""),
    ],
)
def test_credentials_belong_to_host_upstream(
    monkeypatch, backend, env_name, expected_host, header, prefix
):
    """The broker resolves host auth without manufacturing a container auth mount."""
    for key in ("CODEX_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(env_name, "host-only-test-secret")
    upstream = load_broker_upstream(backend)
    assert upstream.host == expected_host
    assert upstream.headers[header] == prefix + "host-only-test-secret"
    assert "host-only-test-secret" not in repr(upstream)


def test_codex_session_auth_stays_on_host(tmp_path, monkeypatch):
    """ChatGPT account routing is loaded from host auth, not a container hint."""
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.setattr("robocode.utils.model_broker.host_codex_home", lambda: tmp_path)
    (tmp_path / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "host-session-secret",
                    "account_id": "trusted-account",
                },
            }
        ),
        encoding="utf-8",
    )
    upstream = load_broker_upstream("codex")
    assert (upstream.host, upstream.base_path) == ("chatgpt.com", "/backend-api/codex")
    assert upstream.headers["ChatGPT-Account-ID"] == "trusted-account"
    assert "host-session-secret" not in repr(upstream)


@pytest.fixture
def chatgpt_broker(broker, tmp_path):
    """Serve the ChatGPT transport using the existing fake upstream."""
    _, requests, response = broker
    directory = tmp_path / "chatgpt"
    directory.mkdir()
    provider = BrokerUpstream(
        "responses",
        "api.openai.com",
        "/v1",
        {"Authorization": "Bearer host-secret"},
        chatgpt=True,
    )
    with model_broker(directory, provider, directory / "audit.jsonl") as path:
        yield path, requests, response


def test_cache_session_only_forwarded(chatgpt_broker):
    """Preserve cache affinity without forwarding arbitrary client authority."""
    address, requests, _ = chatgpt_broker
    session = "01a0bb44-3526-7a93-bc54-037d443037f0"
    conn = _UnixHTTP(str(address))
    conn.request(
        "POST",
        "/v1/responses",
        body=_body(prompt_cache_key=session),
        headers={
            "Session-ID": session,
            "Authorization": "Bearer attacker",
            "Host": "evil.invalid",
            "X-Forwarded-Host": "evil.invalid",
            "thread-id": session,
            "x-codex-turn-state": "untrusted-opaque",
        },
    )
    assert conn.getresponse().status == 200
    conn.close()
    headers = requests[0][3]
    assert headers["session-id"] == session
    assert headers["Authorization"] == "Bearer host-secret"
    assert set(headers) == {"Content-Type", "Accept", "Authorization", "session-id"}
    assert json.loads(requests[0][2])["prompt_cache_key"] == session


@pytest.mark.parametrize(
    "session",
    [
        "",
        "https://evil.invalid/",
        "x" * 8192,
        "01a0bb4435267a93bc54037d443037f0",
        "01a0bb44-3526-7a93-bc54-037d443037fz",
    ],
)
def test_invalid_cache_session_denied(chatgpt_broker, session):
    """Malformed session identifiers fail before contacting the upstream."""
    address, requests, _ = chatgpt_broker
    conn = _UnixHTTP(str(address))
    conn.request("POST", "/v1/responses", body=_body(), headers={"session-id": session})
    assert conn.getresponse().status == 403
    conn.close()
    assert not requests


def test_duplicate_cache_session_denied(chatgpt_broker):
    """Ambiguous duplicate session headers fail closed."""
    address, requests, _ = chatgpt_broker
    conn = _UnixHTTP(str(address))
    body = _body()
    conn.putrequest("POST", "/v1/responses")
    conn.putheader("Content-Length", str(len(body)))
    for _ in range(2):
        conn.putheader("session-id", "01a0bb44-3526-7a93-bc54-037d443037f0")
    conn.endheaders(body)
    assert conn.getresponse().status == 403
    conn.close()
    assert not requests


def test_api_key_transport_does_not_gain_session_header(broker):
    """Keep session metadata out of the separate API-key transport."""
    address, requests, _ = broker
    conn = _UnixHTTP(str(address))
    conn.request(
        "POST",
        "/v1/responses",
        body=_body(),
        headers={"session-id": "01a0bb44-3526-7a93-bc54-037d443037f0"},
    )
    assert conn.getresponse().status == 200
    conn.close()
    assert "session-id" not in requests[0][3]
