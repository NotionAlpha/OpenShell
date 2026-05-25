# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP client helper for talking to sandbox services through the gateway.

Optional — requires the ``[http]`` extra (which pulls in ``requests``).
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import requests

    from .sandbox import SandboxClient, SandboxSession


def http_client_for_sandbox(
    target: "SandboxSession | SandboxClient",
) -> "requests.Session":
    """Return a ``requests.Session`` configured for the target's active gateway.

    The returned session:

    - Loads the gateway's mTLS client cert+key from
      ``~/.config/openshell/gateways/<cluster>/mtls/``.
    - Sets ``verify=False`` (and silences the resulting urllib3 warning)
      because sandbox URLs use a routing hostname that doesn't match the
      gateway cert's SAN.

    Args:
        target: A :class:`~openshell.sandbox.SandboxClient` or
            :class:`~openshell.sandbox.SandboxSession`.

    Returns:
        A :class:`requests.Session` pre-configured with ``cert`` and
        ``verify=False``.

    Raises:
        SandboxError: if ``requests`` is not installed (install with the
            ``[http]`` extra), or if the active gateway has no mTLS material.
    """
    from .sandbox import SandboxError, _resolve_active_cluster, _xdg_config_home

    try:
        import requests
        import urllib3
    except ImportError as exc:
        raise SandboxError(
            "requests is required for http_client_for_sandbox; "
            "install with: pip install 'openshell[http]'"
        ) from exc

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Resolve cluster name: prefer direct _cluster_name, then delegate via _client.
    cluster: str | None = getattr(target, "_cluster_name", None)
    if cluster is None:
        inner_client = getattr(target, "_client", None)
        if inner_client is not None:
            cluster = getattr(inner_client, "_cluster_name", None)

    if cluster is None:
        cluster = _resolve_active_cluster()

    mtls_dir: pathlib.Path = (
        _xdg_config_home() / "openshell" / "gateways" / cluster / "mtls"
    )
    crt_path = mtls_dir / "tls.crt"
    key_path = mtls_dir / "tls.key"

    if not (crt_path.exists() and key_path.exists()):
        raise SandboxError(
            f"no mTLS material at {mtls_dir} — is gateway {cluster!r} registered?"
        )

    s = requests.Session()
    s.cert = (str(crt_path), str(key_path))
    s.verify = False
    return s
