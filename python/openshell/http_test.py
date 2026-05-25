# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for openshell.http_client_for_sandbox."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch


def _make_fake_client(cluster_name: str | None) -> Any:
    """Return a minimal object that looks like SandboxClient."""

    class _FakeClient:
        _cluster_name = cluster_name

    return _FakeClient()


def _make_fake_session(cluster_name: str | None) -> Any:
    """Return a minimal object that looks like SandboxSession."""

    class _FakeSession:
        _client = _make_fake_client(cluster_name)

    return _FakeSession()


def _setup_mtls(tmp_path: Path, gateway_name: str) -> tuple[Path, Path]:
    """Write the active_gateway file and dummy cert/key; return (crt, key) paths."""
    config_dir = tmp_path / "openshell"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "active_gateway").write_text(gateway_name)

    mtls_dir = config_dir / "gateways" / gateway_name / "mtls"
    mtls_dir.mkdir(parents=True, exist_ok=True)

    crt = mtls_dir / "tls.crt"
    key = mtls_dir / "tls.key"
    crt.write_text("FAKE_CERT")
    key.write_text("FAKE_KEY")
    return crt, key


def test_http_client_for_sandbox_uses_active_gateway_certs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Session cert+key come from the active gateway's mtls dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    crt, key = _setup_mtls(tmp_path, "test-cluster")

    fake_client = _make_fake_client("test-cluster")

    from openshell._http import http_client_for_sandbox

    session = http_client_for_sandbox(fake_client)

    assert session.cert == (str(crt), str(key))
    assert session.verify is False


def test_http_client_for_sandbox_falls_back_to_resolve_active_cluster_when_client_has_no_name(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """When _cluster_name is None, resolve via active_gateway file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    crt, key = _setup_mtls(tmp_path, "fallback-cluster")

    fake_client = _make_fake_client(None)

    from openshell._http import http_client_for_sandbox

    session = http_client_for_sandbox(fake_client)

    assert session.cert == (str(crt), str(key))
    assert session.verify is False


def test_http_client_for_sandbox_raises_sandbox_error_when_mtls_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """SandboxError is raised when the mtls dir has no cert/key."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    # Write active_gateway but no mtls files.
    config_dir = tmp_path / "openshell"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "active_gateway").write_text("no-mtls-cluster")

    fake_client = _make_fake_client("no-mtls-cluster")

    from openshell._http import http_client_for_sandbox
    from openshell.sandbox import SandboxError

    import pytest

    with pytest.raises(SandboxError, match="no-mtls-cluster"):
        http_client_for_sandbox(fake_client)


def test_http_client_for_sandbox_raises_sandbox_error_when_requests_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """SandboxError with install-hint message when requests is not installed."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    _setup_mtls(tmp_path, "test-cluster")

    fake_client = _make_fake_client("test-cluster")

    # Simulate requests + urllib3 not being importable.
    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setitem(sys.modules, "urllib3", None)

    # Re-import the module fresh so the lazy import inside the function fires.
    import importlib

    import openshell._http as http_mod

    importlib.reload(http_mod)

    from openshell.sandbox import SandboxError

    import pytest

    with pytest.raises(SandboxError, match="pip install"):
        http_mod.http_client_for_sandbox(fake_client)

    # Restore so other tests are unaffected.
    monkeypatch.delitem(sys.modules, "requests", raising=False)
    monkeypatch.delitem(sys.modules, "urllib3", raising=False)
    importlib.reload(http_mod)


def test_http_client_for_sandbox_accepts_sandbox_session(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """SandboxSession (not SandboxClient) resolves cluster via _client._cluster_name."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    crt, key = _setup_mtls(tmp_path, "test-cluster")

    fake_session = _make_fake_session("test-cluster")

    from openshell._http import http_client_for_sandbox

    session = http_client_for_sandbox(fake_session)

    assert session.cert == (str(crt), str(key))
    assert session.verify is False


def test_http_client_for_sandbox_uses_OPENSHELL_GATEWAY_env_override(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """OPENSHELL_GATEWAY env var is used when _cluster_name is None and no active_gateway file."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENSHELL_GATEWAY", "other-cluster")

    # Set up mtls for "other-cluster" but NO active_gateway file.
    mtls_dir = tmp_path / "openshell" / "gateways" / "other-cluster" / "mtls"
    mtls_dir.mkdir(parents=True, exist_ok=True)
    crt = mtls_dir / "tls.crt"
    key = mtls_dir / "tls.key"
    crt.write_text("FAKE_CERT")
    key.write_text("FAKE_KEY")

    fake_client = _make_fake_client(None)

    from openshell._http import http_client_for_sandbox

    session = http_client_for_sandbox(fake_client)

    assert session.cert == (str(crt), str(key))
    assert session.verify is False
