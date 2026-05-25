# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for constructing SandboxPolicy protos from user-friendly inputs."""
from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlparse

from ._proto import sandbox_pb2

_DEFAULT_EGRESS_RULE_NAME = "default_egress"


def _parse_destination(destination: str) -> tuple[str, int]:
    """Parse a destination spec into (host, port).

    Accepts:
      - URLs: "https://router.huggingface.co:443", "http://example.com/path"
      - host:port: "router.huggingface.co:443", "1.1.1.1:443", "[::1]:443"
      - bare hostnames: "router.huggingface.co" — defaults to 443
    """
    if not destination:
        raise ValueError("destination must be non-empty")

    # URL scheme handling
    if destination.startswith("https://") or destination.startswith("http://"):
        parsed = urlparse(destination)
        host = parsed.hostname
        if host is None:
            raise ValueError(f"could not parse host from URL: {destination!r}")
        port = parsed.port
        if port is None:
            port = 443 if destination.startswith("https://") else 80
        _validate_port(port, destination)
        return host, port

    # Bracketed IPv6 with port: [::1]:443
    # Use the urlparse "//<input>" trick to handle this correctly
    if destination.startswith("["):
        parsed = urlparse(f"//{destination}")
        host = parsed.hostname
        port = parsed.port
        if host is None:
            raise ValueError(f"could not parse bracketed IPv6 address: {destination!r}")
        if port is None:
            raise ValueError(
                f"bracketed IPv6 address must include an explicit port: {destination!r}"
            )
        _validate_port(port, destination)
        return host, port

    # host:port pair (including IPv4 with port)
    if ":" in destination:
        raw_host, raw_port = destination.rsplit(":", 1)
        try:
            port = int(raw_port)
        except ValueError:
            raise ValueError(
                f"invalid port {raw_port!r} in destination {destination!r}: must be an integer"
            )
        _validate_port(port, destination)
        return raw_host, port

    # Bare hostname — default to 443
    return destination, 443


def _validate_port(port: int, destination: str) -> None:
    if not (1 <= port <= 65535):
        raise ValueError(
            f"port {port} in destination {destination!r} is out of range 1–65535"
        )


def policy_from_network_allow(
    destinations: Sequence[str],
    *,
    rule_name: str = _DEFAULT_EGRESS_RULE_NAME,
    filesystem: sandbox_pb2.FilesystemPolicy | None = None,
    landlock: sandbox_pb2.LandlockPolicy | None = None,
) -> sandbox_pb2.SandboxPolicy:
    """Build a SandboxPolicy with a single egress rule allowing *destinations*.

    Each destination is parsed into a host+port NetworkEndpoint. The resulting
    rule is added under *rule_name* (default ``"default_egress"``). *filesystem*
    and *landlock* are passed through unchanged; if omitted, the policy has
    no filesystem entries and a default ``LandlockPolicy(compatibility="best_effort")``.

    If *destinations* is empty, the policy has no ``network_policies`` map entries
    (everything denied) — useful for explicitly air-gapped sandboxes.
    """
    if filesystem is None:
        filesystem = sandbox_pb2.FilesystemPolicy()
    if landlock is None:
        landlock = sandbox_pb2.LandlockPolicy(compatibility="best_effort")

    if not destinations:
        return sandbox_pb2.SandboxPolicy(
            filesystem=filesystem,
            landlock=landlock,
            network_policies={},
        )

    endpoints = []
    for dest in destinations:
        host, port = _parse_destination(dest)
        endpoints.append(sandbox_pb2.NetworkEndpoint(host=host, port=port))

    rule = sandbox_pb2.NetworkPolicyRule(name=rule_name, endpoints=endpoints)

    return sandbox_pb2.SandboxPolicy(
        filesystem=filesystem,
        landlock=landlock,
        network_policies={rule_name: rule},
    )
