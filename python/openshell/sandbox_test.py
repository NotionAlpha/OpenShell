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
