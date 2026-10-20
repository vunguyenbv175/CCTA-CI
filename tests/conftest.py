"""Test path setup + shared deterministic test support.

- Makes packages/services importable without installation.
- Provides FakeProcess, the shared scripted subprocess stand-in used by the
  Orchestrator state-machine tests (quota pause, routing integration). It
  starts "not done" (poll() -> None) and advances only when the test
  explicitly calls mark_done()/mark_failed(), so every tick outcome is
  deterministic.

Import in test modules as: from tests.conftest import FakeProcess
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _dir in (ROOT / "packages", ROOT / "services", ROOT / "tests"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))

# orchestration/state.json is a gitignored runtime checkpoint, but several
# kernel invariants (e.g. byte-stability of rejected validations) read it.
# Guarantee an empty initial checkpoint exists on every fresh clone.
_STATE_FILE = ROOT / "orchestration" / "state.json"
if not _STATE_FILE.exists():
    import json

    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(
        json.dumps({"agents": {}, "schema_version": 3, "checkpoint_revision": 0}, indent=2)
        + "\n",
        encoding="utf-8",
    )


class FakeProcess:
    """Scripted subprocess stand-in for orchestrator state-machine tests."""

    def __init__(self):
        self._code = None

    def mark_done(self):
        self._code = 0

    def mark_failed(self, code=1):
        self._code = code

    def poll(self):
        return self._code
