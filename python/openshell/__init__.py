# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell - Agent execution and management SDK."""

from __future__ import annotations

from .sandbox import (
    ClusterInferenceConfig,
    ExecChunk,
    ExecHandle,
    ExecResult,
    InferenceRouteClient,
    Sandbox,
    SandboxClient,
    SandboxError,
    SandboxRef,
    SandboxSession,
    TlsConfig,
)

from ._proto import sandbox_pb2, openshell_pb2, datamodel_pb2  # noqa: F401
from .policy import policy_from_network_allow  # noqa: F401

try:
    from importlib.metadata import version

    __version__ = version("openshell")
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "ClusterInferenceConfig",
    "ExecChunk",
    "ExecHandle",
    "ExecResult",
    "InferenceRouteClient",
    "Sandbox",
    "SandboxClient",
    "SandboxError",
    "SandboxRef",
    "SandboxSession",
    "TlsConfig",
    "__version__",
    "sandbox_pb2",
    "openshell_pb2",
    "datamodel_pb2",
    "policy_from_network_allow",
]
