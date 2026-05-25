# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from openshell._proto import openshell_pb2
from openshell.sandbox import (
    _PYTHON_CLOUDPICKLE_BOOTSTRAP,
    _SANDBOX_PYTHON_BIN,
    InferenceRouteClient,
    SandboxClient,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeStub:
    def __init__(self) -> None:
        self.request: openshell_pb2.ExecSandboxRequest | None = None

    def ExecSandbox(
        self,
        request: openshell_pb2.ExecSandboxRequest,
        timeout: float | None = None,
    ):
        self.request = request
        _ = timeout
        yield openshell_pb2.ExecSandboxEvent(
            exit=openshell_pb2.ExecSandboxExit(exit_code=0)
        )


class _FakeInferenceStub:
    def __init__(self) -> None:
        self.request = None

    def SetClusterInference(self, request: Any, timeout: float | None = None) -> Any:
        self.request = request
        _ = timeout

        class _Response:
            provider_name = request.provider_name
            model_id = request.model_id
            version = 1

        return _Response()


def _client_with_fake_stub(stub: _FakeStub) -> SandboxClient:
    client = cast("SandboxClient", object.__new__(SandboxClient))
    client._timeout = 30.0
    client._stub = cast("Any", stub)
    return client


def test_exec_sends_stdin_payload() -> None:
    stub = _FakeStub()
    client = _client_with_fake_stub(stub)

    result = client.exec("sandbox-1", ["python", "-c", "print('ok')"], stdin=b"payload")

    assert result.exit_code == 0
    assert stub.request is not None
    assert stub.request.stdin == b"payload"


def test_exec_python_serializes_callable_payload() -> None:
    stub = _FakeStub()
    client = _client_with_fake_stub(stub)

    def add(a: int, b: int) -> int:
        return a + b

    result = client.exec_python("sandbox-1", add, args=(2, 3))

    assert result.exit_code == 0
    assert stub.request is not None
    assert stub.request.command == [
        _SANDBOX_PYTHON_BIN,
        "-c",
        _PYTHON_CLOUDPICKLE_BOOTSTRAP,
    ]
    assert stub.request.environment["OPENSHELL_PYFUNC_B64"]
    assert stub.request.stdin == b""


def test_from_active_cluster_reads_gateway_metadata_layout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    gateway_name = "test-gateway"
    gateway_dir = tmp_path / "openshell" / "gateways" / gateway_name
    mtls_dir = gateway_dir / "mtls"
    mtls_dir.mkdir(parents=True)
    (tmp_path / "openshell" / "active_gateway").write_text(gateway_name)
    (gateway_dir / "metadata.json").write_text(
        json.dumps({"gateway_endpoint": "https://127.0.0.1:8443"})
    )
    (mtls_dir / "ca.crt").write_text("ca")
    (mtls_dir / "tls.crt").write_text("cert")
    (mtls_dir / "tls.key").write_text("key")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENSHELL_GATEWAY", raising=False)

    client = SandboxClient.from_active_cluster()
    try:
        assert client._cluster_name == gateway_name
    finally:
        client.close()


def test_from_active_cluster_prefers_openshell_gateway_env(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    gateway_name = "env-gateway"
    gateway_dir = tmp_path / "openshell" / "gateways" / gateway_name
    mtls_dir = gateway_dir / "mtls"
    mtls_dir.mkdir(parents=True)
    (gateway_dir / "metadata.json").write_text(
        json.dumps({"gateway_endpoint": "https://127.0.0.1:8443"})
    )
    (mtls_dir / "ca.crt").write_text("ca")
    (mtls_dir / "tls.crt").write_text("cert")
    (mtls_dir / "tls.key").write_text("key")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENSHELL_GATEWAY", gateway_name)

    client = SandboxClient.from_active_cluster()
    try:
        assert client._cluster_name == gateway_name
    finally:
        client.close()


class _FakeExposeStub:
    """Fake stub that records ExposeService calls and returns a canned response."""

    def __init__(self, url: str = "https://gateway.example/abc") -> None:
        self._url = url
        self.request: openshell_pb2.ExposeServiceRequest | None = None

    def ExposeService(
        self,
        request: openshell_pb2.ExposeServiceRequest,
        timeout: float | None = None,
    ) -> openshell_pb2.ServiceEndpointResponse:
        self.request = request
        _ = timeout
        return openshell_pb2.ServiceEndpointResponse(url=self._url)


def _client_with_expose_stub(stub: _FakeExposeStub) -> SandboxClient:
    client = cast("SandboxClient", object.__new__(SandboxClient))
    client._timeout = 30.0
    client._stub = cast("Any", stub)
    return client


def _make_sandbox_ref(name: str = "swift-koala", uid: str = "uid-1") -> Any:
    from openshell.sandbox import SandboxRef

    return SandboxRef(id=uid, name=name, phase=0)


def test_session_expose_http_returns_url_from_fake_stub() -> None:
    from openshell.sandbox import SandboxSession

    stub = _FakeExposeStub(url="https://gateway.example/abc")
    client = _client_with_expose_stub(stub)
    ref = _make_sandbox_ref()

    url = SandboxSession(client, ref).expose_http(8080)

    assert url == "https://gateway.example/abc"


def test_session_expose_http_passes_sandbox_name_and_port() -> None:
    from openshell.sandbox import SandboxSession

    stub = _FakeExposeStub()
    client = _client_with_expose_stub(stub)
    ref = _make_sandbox_ref(name="swift-koala", uid="uid-1")

    SandboxSession(client, ref).expose_http(8080)

    assert stub.request is not None
    assert stub.request.sandbox == "swift-koala"  # name, NOT id
    assert stub.request.target_port == 8080
    assert stub.request.service == "http"  # default service_name


def test_session_expose_http_uses_custom_service_name() -> None:
    from openshell.sandbox import SandboxSession

    stub = _FakeExposeStub()
    client = _client_with_expose_stub(stub)
    ref = _make_sandbox_ref()

    SandboxSession(client, ref).expose_http(9090, service_name="grpc")

    assert stub.request is not None
    assert stub.request.service == "grpc"


def test_sandbox_expose_http_raises_when_not_entered() -> None:
    import pytest

    from openshell.sandbox import Sandbox, SandboxError

    sb = Sandbox()
    with pytest.raises(SandboxError, match="context has not been entered"):
        sb.expose_http(8080)


def test_inference_set_cluster_forwards_no_verify_flag() -> None:
    stub = _FakeInferenceStub()
    client = cast("InferenceRouteClient", object.__new__(InferenceRouteClient))
    client._timeout = 30.0
    client._stub = cast("Any", stub)

    client.set_cluster(
        provider_name="openai-dev",
        model_id="gpt-4.1",
        no_verify=True,
    )

    assert stub.request is not None
    assert stub.request.no_verify is True


# ---------------------------------------------------------------------------
# exec_detached tests (Fix 4)
# ---------------------------------------------------------------------------

import threading
import time as _time

import pytest


class _BlockingExecStub:
    """Fake ExecSandbox stub that blocks until an event is set."""

    def __init__(self, block_event: threading.Event, delay: float = 0.0) -> None:
        self._block_event = block_event
        self._delay = delay
        self.request: openshell_pb2.ExecSandboxRequest | None = None

    def ExecSandbox(
        self,
        request: openshell_pb2.ExecSandboxRequest,
        timeout: float | None = None,
    ):
        self.request = request
        _ = timeout
        self._block_event.wait()
        if self._delay:
            _time.sleep(self._delay)
        yield openshell_pb2.ExecSandboxEvent(
            exit=openshell_pb2.ExecSandboxExit(exit_code=0)
        )


class _ErrorExecStub:
    """Fake ExecSandbox stub that raises an exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.request: openshell_pb2.ExecSandboxRequest | None = None

    def ExecSandbox(
        self,
        request: openshell_pb2.ExecSandboxRequest,
        timeout: float | None = None,
    ):
        self.request = request
        _ = timeout
        raise self._exc
        yield  # make it a generator


def _client_with_stub(stub: Any) -> SandboxClient:
    client = cast("SandboxClient", object.__new__(SandboxClient))
    client._timeout = 30.0
    client._stub = cast("Any", stub)
    return client


def test_session_exec_detached_returns_immediately() -> None:
    """exec_detached should return in well under the blocking time (~2 s)."""
    from openshell.sandbox import ExecHandle, SandboxSession

    block_event = threading.Event()
    stub = _BlockingExecStub(block_event)
    client = _client_with_stub(stub)
    ref = _make_sandbox_ref()
    session = SandboxSession(client, ref)

    start = _time.monotonic()
    handle = session.exec_detached(["sleep", "1"])
    elapsed = _time.monotonic() - start

    # Should return well before the 2-second block expires
    assert elapsed < 0.15, f"exec_detached blocked for {elapsed:.3f}s — expected <0.15s"
    assert isinstance(handle, ExecHandle)

    # Clean up: unblock the daemon thread so it can exit
    block_event.set()
    handle.join(timeout=2.0)


def test_session_exec_detached_passes_command_and_env() -> None:
    """exec_detached must forward command, env, and sandbox_id to ExecSandbox."""
    from openshell.sandbox import ExecHandle, SandboxSession

    block_event = threading.Event()
    stub = _BlockingExecStub(block_event)
    client = _client_with_stub(stub)
    ref = _make_sandbox_ref(name="swift-koala", uid="uid-42")
    session = SandboxSession(client, ref)

    handle = session.exec_detached(["python", "/app/agent.py"], env={"FOO": "bar"})

    # Give the daemon thread a moment to call ExecSandbox
    deadline = _time.monotonic() + 1.0
    while stub.request is None and _time.monotonic() < deadline:
        _time.sleep(0.01)

    assert stub.request is not None
    assert list(stub.request.command) == ["python", "/app/agent.py"]
    assert dict(stub.request.environment) == {"FOO": "bar"}
    assert stub.request.sandbox_id == "uid-42"  # exec uses id, not name

    block_event.set()
    handle.join(timeout=2.0)


def test_session_exec_detached_captures_exception_into_handle_error() -> None:
    """If the underlying exec raises, ExecHandle.error should hold the exception."""
    import grpc

    from openshell.sandbox import ExecHandle, SandboxSession

    exc = grpc.RpcError()
    stub = _ErrorExecStub(exc)
    client = _client_with_stub(stub)
    ref = _make_sandbox_ref()
    session = SandboxSession(client, ref)

    handle = session.exec_detached(["python", "/app/agent.py"])
    handle.join(timeout=1.0)

    assert handle.is_alive is False
    assert handle.error is exc


def test_session_exec_detached_handle_is_alive_while_running() -> None:
    """is_alive is True while the daemon thread is blocked; False after it exits."""
    from openshell.sandbox import ExecHandle, SandboxSession

    block_event = threading.Event()
    stub = _BlockingExecStub(block_event)
    client = _client_with_stub(stub)
    ref = _make_sandbox_ref()
    session = SandboxSession(client, ref)

    handle = session.exec_detached(["sleep", "forever"])

    # Give the thread a moment to start and enter the blocking wait
    deadline = _time.monotonic() + 1.0
    while not handle.is_alive and _time.monotonic() < deadline:
        _time.sleep(0.01)

    assert handle.is_alive is True

    block_event.set()
    handle.join(timeout=2.0)

    assert handle.is_alive is False


def test_sandbox_exec_detached_raises_when_not_entered() -> None:
    """Sandbox.exec_detached must raise SandboxError before context entry."""
    from openshell.sandbox import Sandbox, SandboxError

    sb = Sandbox()
    with pytest.raises(SandboxError, match="context has not been entered"):
        sb.exec_detached(["python", "/app/agent.py"])


# ---------------------------------------------------------------------------
# gateway config error handling tests (Fix 7)
# ---------------------------------------------------------------------------


def test_resolve_active_cluster_raises_sandbox_error_on_missing_file(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """_resolve_active_cluster should raise SandboxError with remediation hint when active_gateway is missing."""
    from openshell.sandbox import SandboxError, _resolve_active_cluster

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENSHELL_GATEWAY", raising=False)

    with pytest.raises(SandboxError) as exc_info:
        _resolve_active_cluster()

    err = exc_info.value
    assert "no active gateway configured" in str(err)
    assert "openshell gateway add" in str(err)
    assert str(tmp_path / "openshell" / "active_gateway") in str(err)
    # Verify the FileNotFoundError is chained
    assert isinstance(err.__cause__, FileNotFoundError)


def test_resolve_active_cluster_uses_env_override_before_filesystem(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """_resolve_active_cluster should use $OPENSHELL_GATEWAY env var and not access filesystem."""
    from openshell.sandbox import _resolve_active_cluster

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Do NOT create active_gateway file; filesystem is unavailable
    monkeypatch.setenv("OPENSHELL_GATEWAY", "test-cluster")

    result = _resolve_active_cluster()

    assert result == "test-cluster"
    # Verify active_gateway file was never created/accessed
    assert not (tmp_path / "openshell" / "active_gateway").exists()


def test_from_active_cluster_raises_sandbox_error_on_missing_metadata(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """SandboxClient.from_active_cluster should raise SandboxError with remediation hint when gateway metadata is missing."""
    from openshell.sandbox import SandboxClient, SandboxError

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("OPENSHELL_GATEWAY", raising=False)

    # Set up: create active_gateway but NOT the gateway directory
    (tmp_path / "openshell").mkdir()
    (tmp_path / "openshell" / "active_gateway").write_text("test-cluster")

    with pytest.raises(SandboxError) as exc_info:
        SandboxClient.from_active_cluster()

    err = exc_info.value
    assert "test-cluster" in str(err)
    assert "not registered" in str(err)
    assert "openshell gateway add" in str(err)
    assert "metadata.json" in str(err)
    # Verify the FileNotFoundError is chained
    assert isinstance(err.__cause__, FileNotFoundError)
