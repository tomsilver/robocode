"""The agent must never start when namespace setup or privilege dropping fails."""

# pylint: disable=redefined-outer-name

import pytest

from robocode.utils.apptainer_environment import clean_apptainer_env
from robocode.utils.isolated_transport import verify_namespace, verify_strict_runtime


@pytest.fixture
def namespace(monkeypatch):
    """Provide a kernel snapshot that models the required private namespace."""
    state = {
        "uid": 1013,
        "interfaces": [(1, "lo")],
        "route": "header\n",
        "caps": "0",
        "nnp": "1",
    }
    monkeypatch.setattr(
        "robocode.utils.isolated_transport.os.getuid", lambda: state["uid"]
    )
    monkeypatch.setattr(
        "robocode.utils.isolated_transport.socket.if_nameindex",
        lambda: state["interfaces"],
    )

    def read(path, **_kwargs):
        if str(path) == "/proc/net/route":
            return state["route"]
        return (
            "\n".join(
                f"{key}: {state['caps']}"
                for key in ("CapEff", "CapPrm", "CapBnd", "CapInh", "CapAmb")
            )
            + f"\nNoNewPrivs: {state['nnp']}\n"
        )

    monkeypatch.setattr("robocode.utils.isolated_transport.Path.read_text", read)
    return state


@pytest.mark.usefixtures("namespace")
def test_private_namespace_passes():
    """The required kernel state allows the supervisor to proceed."""
    verify_namespace()


@pytest.mark.parametrize(
    "key,value",
    [
        ("uid", 0),
        ("interfaces", [(1, "lo"), (2, "eth0")]),
        ("route", "header\nroute\n"),
        ("caps", "1000"),
        ("nnp", "0"),
    ],
)
def test_bad_namespace_cannot_fall_back(namespace, key, value):
    """A failed invariant aborts before any agent or relay is started."""
    namespace[key] = value
    with pytest.raises(RuntimeError):
        verify_namespace()


def test_child_environment_excludes_credentials_and_override_flags(monkeypatch):
    """Host auth and Apptainer special variables cannot leak to the child."""
    for key in (
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "APPTAINERENV_OPENAI_API_KEY",
        "APPTAINER_BINDPATH",
        "SINGULARITY_BINDPATH",
        "HTTPS_PROXY",
        "LD_PRELOAD",
    ):
        monkeypatch.setenv(key, "secret-or-override")
    env = clean_apptainer_env()
    assert "secret-or-override" not in env.values()


def test_strict_runtime_rejects_old_image(monkeypatch):
    """The previous image cannot silently remain in use after upgrading the runner."""
    monkeypatch.setattr(
        "pathlib.Path.exists", lambda self: str(self) == "/opt/robocode-mcp"
    )
    with pytest.raises(RuntimeError, match="legacy MCP"):
        verify_strict_runtime()
