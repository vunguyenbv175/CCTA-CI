"""Public-export-only pytest harness guard.

The two frozen-digest assertions named below are host-calibrated: the private
tests read ``/proc/cpuinfo`` and accept only the owner's exact WSL2 fingerprint
(CPU model, NumPy, Python, glibc).  A public GitHub-hosted runner cannot
satisfy that fingerprint.  This public-only root conftest deselects exactly
those two nodes, with a printed reason, so the rest of their modules still
runs unchanged.  It does not evaluate, skip silently, or claim the host-parity
digests; the private gate remains BLOCKED.
"""

from __future__ import annotations

import os
import platform

OWNER_CPU = "AMD Ryzen 5 5600 6-Core Processor"
OWNER_NUMPY = "2.5.2"
OWNER_PYTHON = "3.12.15"
OWNER_GLIBC = "2.43"

HOST_PARITY_NODES = frozenset({
    "tests/core/coronary/test_evidence_digest_streaming.py::test_phantom48_evidence_identity_is_frozen",
    "tests/core/coronary/test_vesselness_blocked_parity.py::test_fixture_response_digest_is_frozen",
})


def _owner_host_profile() -> bool:
    try:
        import numpy as np

        cpu = ""
        with open("/proc/cpuinfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        confstr = getattr(os, "confstr", None)
        glibc = str(confstr("CS_GNU_LIBC_VERSION")).split()[-1] if confstr else ""
    except Exception:
        return False
    return (
        cpu == OWNER_CPU
        and str(np.__version__) == OWNER_NUMPY
        and platform.python_version() == OWNER_PYTHON
        and glibc == OWNER_GLIBC
    )


def pytest_collection_modifyitems(config, items):
    if _owner_host_profile():
        return
    deselected = [item for item in items if item.nodeid in HOST_PARITY_NODES]
    if not deselected:
        return
    items[:] = [item for item in items if item.nodeid not in HOST_PARITY_NODES]
    config.hook.pytest_deselected(items=deselected)
    print("PUBLIC-CLOSURE: deselected host-calibrated node(s); owner WSL2 fingerprint absent:")
    for item in deselected:
        print("  -", item.nodeid)
    print("  These assertions are not evaluated by the public cross-repo run; the private gate remains BLOCKED.")
