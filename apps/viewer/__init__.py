"""Viewer application package (VIEWER_001, FROZEN_V1.0).

The smallest physician-viewer vertical slice: two-layer frame identity over
ACCEPTED ``MPR_VIEW`` artifacts, the frozen binary32 window/level transfer
function, the frozen PNG transport encoder, the closed 7-action interaction
grammar, a loopback-only governed HTTP server, and the verbatim QC panel with
the QC_001-HUMAN seam as the ONLY correction route.

The viewer owns NO medical judgment, NO measurement, and NO display authority
over DICOM (no raw-DICOM display path).  Every displayed pixel is a render of
an accepted canonical-volume MPR artifact (Rule B).  Schemas live OUTSIDE
``core/`` (no ``core/enums.py`` change; ``ArtifactKind.MPR_VIEW`` already
exists and is the only artifact kind this package consumes).

Non-goals (v1, spec Part 0): no CPR/curved planar, no measurement, no
annotation, no reporting/export, no multi-user/networked mode, no raw-DICOM
display path, no model output display.  Custom interactive (continuous)
window/level is explicitly v2 (spec Part 4.2) and is NOT implemented here.
"""

from __future__ import annotations

VIEWER_PACKAGE_VERSION: str = "0.1.0"

__all__ = ["VIEWER_PACKAGE_VERSION"]
