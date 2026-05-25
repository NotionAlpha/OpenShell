# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for top-level proto module aliases on the openshell package.

Verifies that generated _pb2 modules are accessible directly as
``openshell.sandbox_pb2`` / ``openshell.openshell_pb2`` /
``openshell.datamodel_pb2`` without reaching into the private
``openshell._proto`` subpackage.
"""

import openshell
import openshell._proto
from openshell import sandbox_pb2, openshell_pb2, datamodel_pb2


def test_sandbox_pb2_importable() -> None:
    """from openshell import sandbox_pb2 must succeed."""
    assert sandbox_pb2 is not None


def test_openshell_pb2_importable() -> None:
    """from openshell import openshell_pb2 must succeed."""
    assert openshell_pb2 is not None


def test_datamodel_pb2_importable() -> None:
    """from openshell import datamodel_pb2 must succeed."""
    assert datamodel_pb2 is not None


def test_sandbox_pb2_identity() -> None:
    """openshell.sandbox_pb2 must be the same object as openshell._proto.sandbox_pb2."""
    assert openshell.sandbox_pb2 is openshell._proto.sandbox_pb2


def test_openshell_pb2_identity() -> None:
    """openshell.openshell_pb2 must be the same object as openshell._proto.openshell_pb2."""
    assert openshell.openshell_pb2 is openshell._proto.openshell_pb2


def test_datamodel_pb2_identity() -> None:
    """openshell.datamodel_pb2 must be the same object as openshell._proto.datamodel_pb2."""
    assert openshell.datamodel_pb2 is openshell._proto.datamodel_pb2


def test_sandbox_pb2_known_symbol() -> None:
    """SandboxPolicy must be accessible on the top-level sandbox_pb2 alias."""
    assert hasattr(openshell.sandbox_pb2, "SandboxPolicy"), (
        "SandboxPolicy not found in openshell.sandbox_pb2; "
        "check proto/sandbox.proto for the correct top-level message name"
    )


def test_pb2_modules_in_all() -> None:
    """All three _pb2 modules must appear in openshell.__all__."""
    assert "sandbox_pb2" in openshell.__all__
    assert "openshell_pb2" in openshell.__all__
    assert "datamodel_pb2" in openshell.__all__
