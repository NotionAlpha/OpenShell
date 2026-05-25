# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for openshell.policy — policy_from_network_allow builder."""
from __future__ import annotations

import pytest

from openshell._proto import sandbox_pb2
from openshell.policy import _parse_destination, policy_from_network_allow


# ---------------------------------------------------------------------------
# _parse_destination tests
# ---------------------------------------------------------------------------


def test_parse_destination_https_with_port() -> None:
    """HTTPS URL with explicit port."""
    host, port = _parse_destination("https://router.huggingface.co:443")
    assert host == "router.huggingface.co"
    assert port == 443


def test_parse_destination_https_default_port() -> None:
    """HTTPS URL without explicit port defaults to 443."""
    host, port = _parse_destination("https://example.com")
    assert host == "example.com"
    assert port == 443


def test_parse_destination_http_default_port() -> None:
    """HTTP URL without explicit port defaults to 80."""
    host, port = _parse_destination("http://example.com")
    assert host == "example.com"
    assert port == 80


def test_parse_destination_https_with_path_stripped() -> None:
    """Path component is stripped; only host and port are returned."""
    host, port = _parse_destination("https://example.com/some/path")
    assert host == "example.com"
    assert port == 443


def test_parse_destination_host_port_pair() -> None:
    """Plain host:port pair is parsed correctly."""
    host, port = _parse_destination("router.huggingface.co:443")
    assert host == "router.huggingface.co"
    assert port == 443


def test_parse_destination_bare_hostname_defaults_to_443() -> None:
    """Bare hostname with no port defaults to 443."""
    host, port = _parse_destination("router.huggingface.co")
    assert host == "router.huggingface.co"
    assert port == 443


def test_parse_destination_ipv4_with_port() -> None:
    """IPv4 address with port."""
    host, port = _parse_destination("1.1.1.1:443")
    assert host == "1.1.1.1"
    assert port == 443


def test_parse_destination_ipv6_with_port_bracketed() -> None:
    """Bracketed IPv6 with explicit port."""
    host, port = _parse_destination("[::1]:443")
    assert host == "::1"
    assert port == 443


def test_parse_destination_rejects_empty() -> None:
    """Empty string raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        _parse_destination("")


def test_parse_destination_rejects_invalid_port() -> None:
    """Non-integer port in host:port form raises ValueError."""
    with pytest.raises(ValueError):
        _parse_destination("host:notanint")


def test_parse_destination_rejects_port_out_of_range() -> None:
    """Port outside 1–65535 raises ValueError."""
    with pytest.raises(ValueError):
        _parse_destination("host:70000")


# ---------------------------------------------------------------------------
# policy_from_network_allow tests
# ---------------------------------------------------------------------------


def test_policy_from_network_allow_builds_single_rule_with_all_endpoints() -> None:
    """Two destinations produce a single rule with two endpoints."""
    policy = policy_from_network_allow(
        ["https://router.huggingface.co:443", "1.1.1.1:443"]
    )
    assert len(policy.network_policies) == 1
    assert "default_egress" in policy.network_policies
    rule = policy.network_policies["default_egress"]
    assert len(rule.endpoints) == 2
    hosts = {ep.host for ep in rule.endpoints}
    ports = {ep.port for ep in rule.endpoints}
    assert "router.huggingface.co" in hosts
    assert "1.1.1.1" in hosts
    assert ports == {443}


def test_policy_from_network_allow_empty_destinations_yields_empty_map() -> None:
    """Empty destination list produces a policy with no network rules."""
    policy = policy_from_network_allow([])
    assert len(policy.network_policies) == 0


def test_policy_from_network_allow_custom_rule_name() -> None:
    """Custom rule_name kwarg is used as the map key."""
    policy = policy_from_network_allow(
        ["https://example.com"], rule_name="agent_egress"
    )
    assert "agent_egress" in policy.network_policies
    assert "default_egress" not in policy.network_policies


def test_policy_from_network_allow_passes_through_filesystem_and_landlock() -> None:
    """Custom filesystem and landlock objects appear unchanged on the policy."""
    fs = sandbox_pb2.FilesystemPolicy(read_only=["/data"])
    ll = sandbox_pb2.LandlockPolicy(compatibility="hard_requirement")
    policy = policy_from_network_allow(
        ["https://example.com"],
        filesystem=fs,
        landlock=ll,
    )
    assert list(policy.filesystem.read_only) == ["/data"]
    assert policy.landlock.compatibility == "hard_requirement"


def test_policy_from_network_allow_default_landlock_is_best_effort() -> None:
    """Omitting landlock kwarg results in compatibility='best_effort'."""
    policy = policy_from_network_allow(["https://example.com"])
    assert policy.landlock.compatibility == "best_effort"
