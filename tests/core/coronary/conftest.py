"""Shared fixtures of the CORONARY_CENTERLINE_001 suite.

``slice_fixture`` builds the whole chain once per session: the analytic
phantom volume -> voxel<->world transform -> seeded centerline derivation ->
CPR framing, with full §44 provenance.  The viewer tests build their own
session-bound copy (see ``viewer_support``) because the F11 governed identity
must resolve from a CLEAN checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_TESTS_DIR = _ROOT / "tests"

for _entry in (_ROOT, _TESTS_DIR):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))
if (
    str(_ROOT) in sys.path
    and str(_TESTS_DIR) in sys.path
    and sys.path.index(str(_ROOT)) > sys.path.index(str(_TESTS_DIR))
):
    sys.path.remove(str(_ROOT))
    sys.path.insert(0, str(_ROOT))

from tests.core.coronary.support import SliceFixture, build_slice_fixture  # noqa: E402


@pytest.fixture(scope="session")
def checkout_root() -> Path:
    """The executing code checkout root (F11 bound-checkout pattern)."""

    return _ROOT


@pytest.fixture(scope="session")
def slice_fixture(tmp_path_factory: pytest.TempPathFactory) -> SliceFixture:
    """The complete analytic-phantom slice chain, derived once."""

    return build_slice_fixture(tmp_path_factory.mktemp("coronary-slice"))
