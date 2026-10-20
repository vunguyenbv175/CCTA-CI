"""CCTA Studio canonical foundation (WS-00).

The canonical data model is the single center of the platform (Master Plan
§6.2 / Rule B). Every module reads and writes through the typed records in
this package:

- :mod:`core.configuration` -- lanes, enums, IDs, digests, errors.
- :mod:`core.study`         -- ``Study`` / ``Series`` containers.
- :mod:`core.dicom`         -- original DICOM read-only store contract.
- :mod:`core.coordinates`   -- voxel/world/mesh/centerline transforms.
- :mod:`core.volume`        -- canonical volume records.
- :mod:`core.provenance`    -- §44 provenance records + invalidation DAG.
- :mod:`core.artifacts`     -- derived artifacts, review lifecycle, gates.

Original DICOM data is immutable; everything else is a derived artifact with
mandatory provenance. State changes flow through an append-only event log;
there are no setters.
"""

from __future__ import annotations

__all__ = [
    "artifacts",
    "configuration",
    "coordinates",
    "dicom",
    "enums",
    "errors",
    "events",
    "provenance",
    "study",
    "volume",
]
