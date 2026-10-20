"""Path hygiene: product roots must precede the shadow-prone ``tests/`` dir.

``tests/conftest.py`` prepends ``<ROOT>/tests`` to ``sys.path``; without this
reorder, ``tests/core`` would shadow the product ``core`` package for every
module imported from this directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TESTS_DIR = _ROOT / "tests"

for _entry in (_TESTS_DIR, _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

if (
    str(_ROOT) in sys.path
    and str(_TESTS_DIR) in sys.path
    and sys.path.index(str(_ROOT)) > sys.path.index(str(_TESTS_DIR))
):
    sys.path.remove(str(_ROOT))
    sys.path.insert(0, str(_ROOT))


@pytest.fixture(scope="session")
def clean_checkout() -> Path:
    """The executing code checkout root (r3 R3-01).

    The production root is derived exclusively from the executing code
    (__file__-derived) and alternate-root injection is rejected.  Tests
    running in the active test runner execute from this checkout root.
    """

    return _ROOT
