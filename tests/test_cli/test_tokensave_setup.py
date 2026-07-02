"""Tests for tokensave coding-task compressor setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.cli import wrap as wrap_cli
from headroom.mcp_registry import build_tokensave_spec
from headroom.mcp_registry.base import RegisterResult, RegisterStatus, ServerSpec
from headroom.mcp_registry.ledger import headroom_installed_matching, record_install

_FAKE_BIN = Path("/usr/local/bin/tokensave")


def _equivalent(a: ServerSpec, b: ServerSpec) -> bool:
    return (a.command, tuple(a.args), dict(a.env)) == (b.command, tuple(b.args), dict(b.env))


class _FakeRegistrar:
    """Registrar mirroring real ``register_server`` overwrite semantics."""

    def __init__(self, name: str = "claude", *, detected: bool = True, server=None):
        self.name = name
        self.display_name = name.capitalize()
        self._detected = detected
        self._server = server
        self.force_calls: list[bool] = []
        self.unregistered: list[str] = []

    def detect(self) -> bool:
        return self._detected

    def get_server(self, server_name: str):
        return self._server if server_name == "tokensave" else None

    def register_server(self, spec: ServerSpec, *, force: bool = False) -> RegisterResult:
        self.force_calls.append(force)
        if self._server is not None and not _equivalent(self._server, spec) and not force:
            return RegisterResult(RegisterStatus.MISMATCH, "differs")
        self._server = spec
        return RegisterResult(RegisterStatus.REGISTERED, "ok")

    def unregister_server(self, server_name: str) -> bool:
        self.unregistered.append(server_name)
        self._server = None
        return True


@pytest.fixture(autouse=True)
def _workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    # Never touch the network or run the real binary during these unit tests.
    monkeypatch.setattr(wrap_cli, "_index_tokensave_project", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# _setup_tokensave_mcp
# ---------------------------------------------------------------------------


def test_setup_registers_and_records_when_binary_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wrap_cli, "_ensure_tokensave_binary", lambda verbose=False: _FAKE_BIN)
    registrar = _FakeRegistrar()

    assert wrap_cli._setup_tokensave_mcp(registrar) is True
    assert registrar._server is not None
    assert registrar._server.name == "tokensave"
    assert registrar._server.command == str(_FAKE_BIN)
    # Ledger now proves Headroom owns the entry.
    assert headroom_installed_matching("claude", registrar.get_server("tokensave"))


def test_setup_returns_false_when_binary_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli, "_ensure_tokensave_binary", lambda verbose=False: None)
    registrar = _FakeRegistrar()

    assert wrap_cli._setup_tokensave_mcp(registrar) is False
    assert registrar._server is None  # nothing registered


def test_setup_skips_when_agent_not_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = {"called": False}

    def _should_not_run(verbose=False):
        sentinel["called"] = True
        return _FAKE_BIN

    monkeypatch.setattr(wrap_cli, "_ensure_tokensave_binary", _should_not_run)
    registrar = _FakeRegistrar(detected=False)

    assert wrap_cli._setup_tokensave_mcp(registrar) is False
    assert sentinel["called"] is False  # never even fetched the binary


def test_setup_migrates_stale_headroom_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli, "_ensure_tokensave_binary", lambda verbose=False: _FAKE_BIN)
    # A stale Headroom-installed entry (different binary path) is on disk.
    stale = build_tokensave_spec("/old/path/tokensave")
    record_install("claude", stale)
    registrar = _FakeRegistrar(server=stale)

    assert wrap_cli._setup_tokensave_mcp(registrar) is True
    # Force-updated to the current spec.
    assert registrar.force_calls[-1] is True
    assert registrar._server.command == str(_FAKE_BIN)


def test_setup_preserves_user_managed_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli, "_ensure_tokensave_binary", lambda verbose=False: _FAKE_BIN)
    # User-managed entry (NOT in ledger) that differs from our spec.
    user = ServerSpec(name="tokensave", command="/custom/tokensave", args=("serve",))
    registrar = _FakeRegistrar(server=user)

    wrap_cli._setup_tokensave_mcp(registrar)
    # Never force-overwrote a user-managed entry.
    assert True not in registrar.force_calls
    assert registrar._server.command == "/custom/tokensave"


# ---------------------------------------------------------------------------
# _disable_tokensave_mcp
# ---------------------------------------------------------------------------


def test_disable_removes_headroom_installed(capsys: pytest.CaptureFixture[str]) -> None:
    spec = build_tokensave_spec(str(_FAKE_BIN))
    record_install("claude", spec)
    registrar = _FakeRegistrar(server=spec)

    wrap_cli._disable_tokensave_mcp(registrar, verbose=True)

    assert registrar.unregistered == ["tokensave"]
    assert "Removed previously-installed tokensave MCP" in capsys.readouterr().out


def test_disable_preserves_user_managed(capsys: pytest.CaptureFixture[str]) -> None:
    user = ServerSpec(name="tokensave", command="/custom/tokensave")
    registrar = _FakeRegistrar(server=user)

    wrap_cli._disable_tokensave_mcp(registrar, verbose=True)

    assert registrar.unregistered == []
    assert "user-managed" in capsys.readouterr().out


def test_disable_noop_when_absent(capsys: pytest.CaptureFixture[str]) -> None:
    registrar = _FakeRegistrar(server=None)
    wrap_cli._disable_tokensave_mcp(registrar, verbose=True)
    assert registrar.unregistered == []
    assert "Skipping tokensave MCP" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _setup_coding_compressor — primary/backup policy
# ---------------------------------------------------------------------------
